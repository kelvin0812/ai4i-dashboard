"""Central constants: paths, column names, display labels, colours.

Everything that more than one module needs lives here so labels and colours
stay identical on every page.
"""

from pathlib import Path

# --- Paths (relative to this file, never hard-coded) -------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DATA_PATH = DATA_DIR / "ai4i_cleaned.csv"
SCALER_PATH = MODELS_DIR / "ai4i_sensor_scaler.joblib"

# --- Columns -----------------------------------------------------------------
TYPE_COL = "type"
TARGET_COL = "machine_failure"

# Order matters: this is the order the saved scaler expects.
SENSOR_ORDER = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

FAILURE_MODE_COLS = ["twf", "hdf", "pwf", "osf", "rnf"]

RAW_COLUMNS = [TYPE_COL, *SENSOR_ORDER, TARGET_COL, *FAILURE_MODE_COLS]

ENGINEERED_COLS = ["temperature_difference_k", "calculated_power_w"]

# Derived display columns added by src.data
TYPE_LABEL_COL = "type_label"
FAILURE_LABEL_COL = "failure_label"

# --- Product type ------------------------------------------------------------
TYPE_MAP = {0: "L", 1: "M", 2: "H"}
TYPE_ORDER = ["L", "M", "H"]
TYPE_DESCRIPTIONS = {"L": "Low quality", "M": "Medium quality", "H": "High quality"}

# --- Failure status ----------------------------------------------------------
NO_FAILURE = "No failure"
FAILURE = "Failure"
FAILURE_ORDER = [NO_FAILURE, FAILURE]

# One consistent colour pair on every page. Validated colour-blind-safe
# (protan dE 16.2, normal-vision dE 30.8): vibrant red = failure, steel blue
# = no failure, so the rare failures pop against the receding majority.
FAILURE_COLORS = {NO_FAILURE: "#4568a8", FAILURE: "#e8173f"}

# --- Display labels (always with units) --------------------------------------
LABELS = {
    "air_temperature_k": "Air temperature (K)",
    "process_temperature_k": "Process temperature (K)",
    "rotational_speed_rpm": "Rotational speed (rpm)",
    "torque_nm": "Torque (Nm)",
    "tool_wear_min": "Tool wear (min)",
    "temperature_difference_k": "Temperature difference (K)",
    "calculated_power_w": "Calculated power (W)",
    TYPE_LABEL_COL: "Product type",
    FAILURE_LABEL_COL: "Failure status",
    TARGET_COL: "Machine failure",
}

FAILURE_MODE_LABELS = {
    "twf": "Tool wear failure (TWF)",
    "hdf": "Heat dissipation failure (HDF)",
    "pwf": "Power failure (PWF)",
    "osf": "Overstrain failure (OSF)",
    "rnf": "Random failure (RNF)",
}

# --- Dataset generation rules (from the published AI4I rules) ----------------
# These are the rules used to *generate* the synthetic data. They are shown as
# reference lines only and must never be described as real safety limits.
GENERATION_RULES = {
    "temperature_difference_k": [
        (8.6, "HDF: < 8.6 K (with speed < 1,380 rpm)"),
    ],
    "rotational_speed_rpm": [
        (1380, "HDF: < 1,380 rpm (with temp. diff. < 8.6 K)"),
    ],
    "calculated_power_w": [
        (3500, "PWF: < 3,500 W"),
        (9000, "PWF: > 9,000 W"),
    ],
}
RULE_COLOR = "#4c0519"
