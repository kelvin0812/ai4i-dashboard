"""Model logic for the Failure Risk Prediction page (Kelvin).

Plain Python / pandas, no Streamlit, so it can be unit-tested. The page in
src/prediction.py wraps these with Streamlit caching and widgets.

Pipeline the app runs for one set of machine settings:

    physical-unit inputs --validate_inputs()--> build_model_frame()
        (type coded 0/1/2, five sensors scaled with ai4i_sensor_scaler.joblib)
    --> models/failure_model.joblib (adds engineered features internally)
    --> failure probability, compared with the threshold in models/model_card.json
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, SCALER_PATH, SENSOR_ORDER, TYPE_MAP

MODEL_PATH = MODELS_DIR / "failure_model.joblib"
CARD_PATH = MODELS_DIR / "model_card.json"
MODEL_INPUT_COLUMNS = ["type", *SENSOR_ORDER]
TYPE_CODE = {label: code for code, label in TYPE_MAP.items()}

# Hard limits the form accepts (wider than the data, so users can explore) and
# the range actually seen in the 10,000 records (outside it the model is
# extrapolating and its output should not be trusted).
INPUT_LIMITS = {
    "air_temperature_k": (280.0, 320.0),
    "process_temperature_k": (290.0, 330.0),
    "rotational_speed_rpm": (500, 3500),
    "torque_nm": (0.0, 120.0),
    "tool_wear_min": (0, 350),
}
DATA_RANGE = {
    "air_temperature_k": (295.3, 304.5),
    "process_temperature_k": (305.7, 313.8),
    "rotational_speed_rpm": (1168, 2886),
    "torque_nm": (3.8, 76.6),
    "tool_wear_min": (0, 253),
}

# Risk bands relative to the decision threshold t: < t/2 low, < t elevated, >= t high.
LOW, ELEVATED, HIGH = "Low", "Elevated", "High"

# Ready-made scenarios for the demo. Values are in physical units.
PRESETS = {
    "Typical operation": {"type": "M", "air_temperature_k": 300.0, "process_temperature_k": 310.0,
                          "rotational_speed_rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 100},
    "Worn tool, heavy load": {"type": "L", "air_temperature_k": 300.0, "process_temperature_k": 310.0,
                              "rotational_speed_rpm": 1400, "torque_nm": 55.0, "tool_wear_min": 215},
    "Poor heat dissipation": {"type": "M", "air_temperature_k": 302.5, "process_temperature_k": 310.5,
                              "rotational_speed_rpm": 1300, "torque_nm": 50.0, "tool_wear_min": 80},
    "Power too high": {"type": "H", "air_temperature_k": 299.0, "process_temperature_k": 309.5,
                       "rotational_speed_rpm": 1300, "torque_nm": 70.0, "tool_wear_min": 60},
    "Power too low": {"type": "M", "air_temperature_k": 299.0, "process_temperature_k": 309.5,
                      "rotational_speed_rpm": 2400, "torque_nm": 12.0, "tool_wear_min": 60},
}
DEFAULT_PRESET = "Typical operation"

# Overstrain rule threshold (tool wear x torque, min*Nm) by product type.
OSF_LIMIT = {"L": 11_000, "M": 12_000, "H": 13_000}


# --- Loading -------------------------------------------------------------------
def read_model(path: Path = MODEL_PATH):
    return joblib.load(path)


def read_card(path: Path = CARD_PATH) -> dict:
    return json.loads(Path(path).read_text())


def read_scaler(path: Path = SCALER_PATH):
    return joblib.load(path)


# --- Validation ------------------------------------------------------------------
@dataclass
class Validation:
    errors: list = field(default_factory=list)    # block the prediction
    warnings: list = field(default_factory=list)  # prediction shown, with a caveat

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_inputs(product_type, sensors: dict, labels: dict) -> Validation:
    """Check one set of inputs. `labels` maps column -> display label with unit."""
    v = Validation()
    if product_type not in TYPE_CODE:
        v.errors.append("Choose a product type (L, M or H).")
    for col in SENSOR_ORDER:
        value = sensors.get(col)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            v.errors.append(f"{labels[col]} is missing.")
            continue
        lo, hi = INPUT_LIMITS[col]
        if not lo <= value <= hi:
            v.errors.append(f"{labels[col]} must be between {lo:g} and {hi:g}.")
            continue
        dlo, dhi = DATA_RANGE[col]
        if not dlo <= value <= dhi:
            v.warnings.append(
                f"{labels[col]} = {value:g} is outside the training data ({dlo:g}–{dhi:g}). "
                "The model is extrapolating, so treat the result with caution."
            )
    air, proc = sensors.get("air_temperature_k"), sensors.get("process_temperature_k")
    if v.ok and proc <= air:
        v.warnings.append(
            "Process temperature is not above air temperature. Every record in the data has the "
            "process about 8–12 K warmer, so this combination was never seen in training."
        )
    return v


# --- Prediction ------------------------------------------------------------------
def build_model_frame(product_type: str, sensors: dict, scaler) -> pd.DataFrame:
    """One-row DataFrame in MODEL_INPUT_COLUMNS order with the five sensors scaled."""
    return build_model_frames(pd.DataFrame([{**sensors, "type": product_type}]), scaler)


def build_model_frames(physical: pd.DataFrame, scaler) -> pd.DataFrame:
    """Many rows at once. `physical` has a `type` column (L/M/H) and the sensors."""
    scaled = pd.DataFrame(scaler.transform(physical[SENSOR_ORDER]), columns=SENSOR_ORDER)
    scaled.insert(0, "type", physical["type"].map(TYPE_CODE).to_numpy())
    return scaled[MODEL_INPUT_COLUMNS]


def failure_probability(model, scaler, product_type: str, sensors: dict) -> float:
    return float(model.predict_proba(build_model_frame(product_type, sensors, scaler))[0, 1])


def risk_band(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return HIGH
    if probability >= threshold / 2:
        return ELEVATED
    return LOW


def sweep(model, scaler, product_type: str, sensors: dict, column: str, points: int = 120) -> pd.DataFrame:
    """Failure probability as `column` varies over the data range, others held fixed."""
    lo, hi = DATA_RANGE[column]
    values = np.linspace(lo, hi, points)
    grid = pd.DataFrame([{**sensors, "type": product_type}] * points)
    grid[column] = values
    proba = model.predict_proba(build_model_frames(grid, scaler))[:, 1]
    return pd.DataFrame({column: values, "probability": proba})


# --- Derived values and dataset generation rules -------------------------------------
def derived_values(product_type: str, sensors: dict) -> dict:
    diff = sensors["process_temperature_k"] - sensors["air_temperature_k"]
    power = sensors["torque_nm"] * sensors["rotational_speed_rpm"] * 2 * np.pi / 60
    strain = sensors["tool_wear_min"] * sensors["torque_nm"]
    return {"temperature_difference_k": diff, "calculated_power_w": power,
            "strain_min_nm": strain, "osf_limit": OSF_LIMIT.get(product_type)}


def generation_rules(product_type: str, sensors: dict) -> list[dict]:
    """Which published AI4I generation rules these inputs fall under.

    These are the rules the dataset authors used to CREATE the synthetic
    failures. They explain the model's behaviour; they are not safety limits.
    """
    d = derived_values(product_type, sensors)
    speed, wear = sensors["rotational_speed_rpm"], sensors["tool_wear_min"]
    return [
        {"rule": "Heat dissipation (HDF)",
         "condition": "temperature difference < 8.6 K and speed < 1,380 rpm",
         "value": f"{d['temperature_difference_k']:.1f} K, {speed:,.0f} rpm",
         "met": d["temperature_difference_k"] < 8.6 and speed < 1380},
        {"rule": "Power (PWF)",
         "condition": "power < 3,500 W or > 9,000 W",
         "value": f"{d['calculated_power_w']:,.0f} W",
         "met": d["calculated_power_w"] < 3500 or d["calculated_power_w"] > 9000},
        {"rule": "Overstrain (OSF)",
         "condition": f"tool wear × torque > {d['osf_limit']:,} min·Nm (type {product_type})",
         "value": f"{d['strain_min_nm']:,.0f} min·Nm",
         "met": d["osf_limit"] is not None and d["strain_min_nm"] > d["osf_limit"]},
        {"rule": "Tool wear (TWF)",
         "condition": "tool wear 200–240 min (tool replaced or fails at a random point here)",
         "value": f"{wear:,.0f} min",
         "met": 200 <= wear <= 240},
    ]
