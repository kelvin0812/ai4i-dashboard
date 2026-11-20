"""Failure Risk Prediction page (Kelvin).

Called by app_pages/5_prediction.py. The model logic (validation, scaling,
prediction, what-if sweeps, generation-rule checks) lives in src/model.py so it
can be tested without Streamlit; this file is the UI.

Model contract (see models/README.md and models/model_card.json):
  inputs   type (L/M/H -> 0/1/2, not scaled) + the five sensors in SENSOR_ORDER,
           scaled with models/ai4i_sensor_scaler.joblib
  model    models/failure_model.joblib, an sklearn Pipeline that adds
           temperature difference and power internally (src/features.py)
  output   failure probability; "High risk" when >= the card's threshold
  never    twf/hdf/pwf/osf/rnf as inputs

Retrain with:  python training/train_model.py
"""

import pandas as pd
import streamlit as st

from src import model as m
from src.config import LABELS, ROOT_DIR, SENSOR_ORDER, TYPE_DESCRIPTIONS, TYPE_ORDER
from src.prediction_charts import confusion_chart, risk_gauge, sweep_chart
from src.style import callout, page_header

MODEL_PATH = m.MODEL_PATH
MODEL_INPUT_COLUMNS = m.MODEL_INPUT_COLUMNS
COMPARISON_PATH = ROOT_DIR / "reports" / "model_comparison.csv"

STORE_KEY = "pred_inputs"     # survives page switches (widget keys do not)
PRESET_KEY = "pred_preset"
SWEEP_KEY = "pred_sweep_col"
INTEGER_INPUTS = {"rotational_speed_rpm": 10, "tool_wear_min": 1}
FLOAT_STEP = {"air_temperature_k": 0.1, "process_temperature_k": 0.1, "torque_nm": 0.5}


# --- Cached resources ------------------------------------------------------------------
@st.cache_resource
def load_scaler():
    """The StandardScaler fitted on the five training sensors."""
    return m.read_scaler()


@st.cache_resource
def load_model():
    return m.read_model()


@st.cache_data
def load_card() -> dict:
    return m.read_card()


@st.cache_data
def load_comparison() -> pd.DataFrame | None:
    return pd.read_csv(COMPARISON_PATH) if COMPARISON_PATH.exists() else None


def build_input_frame(product_type: str, sensors: dict) -> pd.DataFrame:
    """One-row DataFrame in MODEL_INPUT_COLUMNS order, sensors scaled."""
    return m.build_model_frame(product_type, sensors, load_scaler())


# --- Input state -------------------------------------------------------------------------
def _key(col: str) -> str:
    return f"pred_{col}"


def _init_state() -> None:
    store = st.session_state.setdefault(STORE_KEY, dict(m.PRESETS[m.DEFAULT_PRESET]))
    for col in ["type", *SENSOR_ORDER]:
        if _key(col) not in st.session_state:
            st.session_state[_key(col)] = store[col]


def _load_preset() -> None:
    preset = m.PRESETS[st.session_state[PRESET_KEY]]
    for col, value in preset.items():
        st.session_state[_key(col)] = value


def _inputs_form() -> tuple[str, dict]:
    st.subheader("Machine settings")
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    c1.selectbox("Scenario", list(m.PRESETS), key=PRESET_KEY,
                 help="Ready-made settings for the demo. Loading one overwrites the inputs below.")
    c2.button("Load", on_click=_load_preset, width="stretch", key="pred_load")

    product_type = st.radio(
        "Product type", TYPE_ORDER, key=_key("type"), horizontal=True,
        format_func=lambda t: f"{t} · {TYPE_DESCRIPTIONS[t]}",
    )
    sensors = {}
    for col in SENSOR_ORDER:
        lo, hi = m.INPUT_LIMITS[col]
        dlo, dhi = m.DATA_RANGE[col]
        help_text = f"Seen in the data: {dlo:g}–{dhi:g}. Accepted: {lo:g}–{hi:g}."
        if col in INTEGER_INPUTS:
            sensors[col] = st.number_input(LABELS[col], min_value=int(lo), max_value=int(hi),
                                           step=INTEGER_INPUTS[col], key=_key(col), help=help_text)
        else:
            sensors[col] = st.number_input(LABELS[col], min_value=float(lo), max_value=float(hi),
                                           step=FLOAT_STEP[col], format="%.1f", key=_key(col),
                                           help=help_text)
    st.session_state[STORE_KEY] = {"type": product_type, **sensors}
    return product_type, sensors


# --- Result ------------------------------------------------------------------------------
_VERDICT = {
    m.HIGH: ("High failure risk", "The model flags these settings as likely to fail."),
    m.ELEVATED: ("Elevated failure risk", "Below the decision threshold, but at least half of it."),
    m.LOW: ("Low failure risk", "Well below the decision threshold."),
}


def _result(product_type: str, sensors: dict, card: dict) -> None:
    st.subheader("Predicted failure risk")
    check = m.validate_inputs(product_type, sensors, LABELS)
    if not check.ok:
        for msg in check.errors:
            st.error(msg, icon=":material/error:")
        st.info("Fix the inputs above to see a prediction.")
        return False

    try:
        proba = m.failure_probability(load_model(), load_scaler(), product_type, sensors)
    except Exception as exc:  # the page must never crash in a demo
        st.error(f"The model could not make a prediction: {exc}", icon=":material/error:")
        return False

    threshold = card["threshold"]
    band = m.risk_band(proba, threshold)
    title, detail = _VERDICT[band]
    {m.HIGH: st.error, m.ELEVATED: st.warning, m.LOW: st.success}[band](
        f"**{title}**: {detail}", icon=":material/warning:" if band != m.LOW else ":material/check_circle:")

    c1, c2 = st.columns(2)
    c1.metric("Failure probability", f"{proba * 100:.1f}%", border=True,
              help="Model score for these settings. At or above the threshold counts as a predicted failure.")
    c2.metric("Risk level", band, border=True,
              help=f"Low < {threshold * 50:.0f}% ≤ Elevated < {threshold * 100:.0f}% ≤ High.")
    st.plotly_chart(risk_gauge(proba, threshold), width="stretch", key="pred_gauge")
    st.caption(f"Dark line: decision threshold **{threshold * 100:.0f}%**, chosen on the training data "
               "to balance missed failures and false alarms (best failure-class F1). "
               "Bands: low (blue), elevated (amber), high (red).")

    for msg in check.warnings:
        st.warning(msg, icon=":material/info:")

    d = m.derived_values(product_type, sensors)
    c1, c2 = st.columns(2)
    c1.metric("Temp. difference", f"{d['temperature_difference_k']:.1f} K",
              help="Process − air temperature (K). Computed the same way inside the model.")
    c2.metric("Power", f"{d['calculated_power_w']:,.0f} W",
              help="Torque × speed × 2π ÷ 60 (W). Computed the same way inside the model.")

    rules = m.generation_rules(product_type, sensors)
    met = [r["rule"] for r in rules if r["met"]]
    with st.expander(f"Dataset generation rules: {len(met)} of {len(rules)} met"
                     + (f" ({', '.join(met)})" if met else "")):
        st.caption("These are the published rules the AI4I authors used to create the synthetic "
                   "failures. They help explain the model's output. They are not real safety limits.")
        st.dataframe(
            pd.DataFrame(rules).rename(columns={"rule": "Rule", "condition": "Failure condition",
                                                "value": "These settings", "met": "Met"}),
            hide_index=True, width="stretch",
        )
    return True


def _what_if(product_type: str, sensors: dict, card: dict) -> None:
    st.subheader("What-if: change one setting")
    st.caption("How the predicted risk changes as one input moves across the range seen in the data, "
               "with every other input held at the values above.")
    col = st.selectbox("Setting to vary", SENSOR_ORDER, format_func=LABELS.get,
                       index=SENSOR_ORDER.index("torque_nm"), key=SWEEP_KEY)
    curve = m.sweep(load_model(), load_scaler(), product_type, sensors, col)
    st.plotly_chart(sweep_chart(curve, col, sensors[col], card["threshold"]), width="stretch",
                    key="pred_sweep")
    above = curve[curve["probability"] >= card["threshold"]][col]
    if above.empty:
        st.caption(f"Across the whole range of {LABELS[col].lower()}, the risk stays below the threshold.")
    else:
        st.caption(
            f"The risk is above the threshold for {LABELS[col].lower()} between "
            f"{above.min():,.1f} and {above.max():,.1f} (possibly in several separate stretches; "
            "see the chart). This is an association learned from synthetic data, not proof that "
            "changing this setting alone prevents failure."
        )


def _performance(card: dict) -> None:
    st.subheader("How good is the model?")
    t = card["test_metrics"]
    st.caption(
        f"{card['model_name']}, evaluated once on the untouched test set "
        f"({card['test_rows']:,} records, {card['test_failures']} failures). "
        "Model and threshold were chosen with cross-validation on the training set only."
    )
    c = st.columns(4)
    c[0].metric("Recall", f"{t['recall'] * 100:.1f}%", border=True,
                help="Share of real failures the model caught.")
    c[1].metric("Precision", f"{t['precision'] * 100:.1f}%", border=True,
                help="Share of failure alarms that were real failures.")
    c[2].metric("F1-score", f"{t['f1'] * 100:.1f}%", border=True,
                help="Balance of precision and recall for the failure class.")
    c[3].metric("Accuracy", f"{t['accuracy'] * 100:.1f}%", border=True,
                help="Shown for completeness only: always predicting 'no failure' already scores ~96.6%.")
    left, right = st.columns([2, 3], vertical_alignment="center")
    with left:
        st.plotly_chart(confusion_chart(t), width="stretch", key="pred_confusion")
    with right:
        st.markdown(
            f"Of the **{t['tp'] + t['fn']}** real failures in the test set, the model caught "
            f"**{t['tp']}** and missed **{t['fn']}**. It raised **{t['fp']}** false alarms among "
            f"**{t['tn'] + t['fp']:,}** records that did not fail.\n\n"
            "A model that always predicts *no failure* would score about 96.6% accuracy while "
            "catching none, which is why recall, precision and F1 are the headline numbers."
        )
    comparison = load_comparison()
    if comparison is not None:
        with st.expander("All candidate models compared"):
            table = comparison[["model", "cv_f1", "test_precision", "test_recall", "test_f1"]].rename(
                columns={"model": "Model", "cv_f1": "CV F1", "test_precision": "Test precision",
                         "test_recall": "Test recall", "test_f1": "Test F1"})
            pct = st.column_config.NumberColumn(format="percent")
            st.dataframe(table, hide_index=True, width="stretch",
                         column_config={k: pct for k in table.columns if k != "Model"})
            st.caption("Every candidate tried. The model with the best CV F1 was selected; test "
                       "scores are listed for transparency only.")
    callout(
        "<b>Limits.</b> AI4I is synthetic. The engineered features match the rules used to generate "
        "its failures, so the model learning them is expected. Random failures (RNF) cannot be "
        "predicted. The data has no repair costs or downtime, so savings cannot be measured."
    )


# --- Entry point -------------------------------------------------------------------------
def render_prediction() -> None:
    """Entry point called by app_pages/5_prediction.py."""
    try:
        card = load_card()
        load_model()
        load_scaler()
    except FileNotFoundError as exc:
        page_header("Failure Risk Prediction", "Estimate the failure risk for one set of machine settings.",
                    eyebrow="Model · not available")
        st.error(f"Model files are missing ({exc.filename}). Run `python training/train_model.py` "
                 "from the project folder to create them.", icon=":material/error:")
        return
    except Exception as exc:
        page_header("Failure Risk Prediction", "Estimate the failure risk for one set of machine settings.",
                    eyebrow="Model · not available")
        st.error(f"The model could not be loaded: {exc}. Check that the package versions match "
                 "requirements.txt, or retrain with `python training/train_model.py`.",
                 icon=":material/error:")
        return

    page_header(
        "Failure Risk Prediction",
        "Enter machine settings in physical units to estimate the chance of a machine failure.",
        eyebrow=f"Model · {card['model_name'].lower()}",
    )
    st.caption("The sidebar filters do not apply to this page. They only affect the exploratory views.")

    _init_state()
    left, right = st.columns([1, 1.35], gap="large")
    with left:
        product_type, sensors = _inputs_form()
    with right:
        valid = _result(product_type, sensors, card)
    if valid:
        _what_if(product_type, sensors, card)
    st.divider()
    _performance(card)
