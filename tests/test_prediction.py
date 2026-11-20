"""Model and prediction-page tests (Kelvin)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from sklearn.metrics import f1_score, precision_score, recall_score
from streamlit.testing.v1 import AppTest

from src import model as m
from src.config import LABELS, SENSOR_ORDER

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
PAGE = "app_pages/5_prediction.py"
LEAKY = {"twf", "hdf", "pwf", "osf", "rnf", "machine_failure"}


@pytest.fixture(scope="module")
def model():
    return m.read_model()


@pytest.fixture(scope="module")
def scaler():
    return m.read_scaler()


@pytest.fixture(scope="module")
def card():
    return m.read_card()


def _split(preset):
    p = dict(m.PRESETS[preset])
    return p.pop("type"), p


# --- Model contract ------------------------------------------------------------------
def test_card_input_order(card, model):
    assert card["input_columns"] == ["type", *SENSOR_ORDER]
    assert list(model.feature_names_in_) == card["input_columns"]
    assert not LEAKY & set(card["input_columns"])


def test_card_metrics_reproduce_on_test_set(card, model):
    test = pd.read_csv(ROOT / "training" / "data" / "test_scaled.csv")
    pred = (model.predict_proba(test[card["input_columns"]])[:, 1] >= card["threshold"]).astype(int)
    y = test["machine_failure"]
    t = card["test_metrics"]
    assert f1_score(y, pred) == pytest.approx(t["f1"])
    assert precision_score(y, pred) == pytest.approx(t["precision"])
    assert recall_score(y, pred) == pytest.approx(t["recall"])
    assert len(test) == 2000 and y.sum() == 68


def test_app_scaling_matches_kiat_files(scaler):
    """Physical-unit rows scaled by the app equal Kiat's scaled training rows."""
    raw = pd.read_csv(ROOT / "data" / "ai4i_cleaned.csv")
    train = pd.read_csv(ROOT / "training" / "data" / "train_scaled.csv")
    first = train.iloc[0]
    # find the physical row whose scaled values match the first training row
    scaled_all = m.build_model_frames(raw.assign(type=raw["type"].map({0: "L", 1: "M", 2: "H"})), scaler)
    hits = np.isclose(scaled_all[SENSOR_ORDER].to_numpy(), first[SENSOR_ORDER].to_numpy().astype(float)).all(axis=1)
    assert hits.sum() >= 1
    assert (scaled_all.loc[hits, "type"] == first["type"]).all()


# --- Prediction behaviour ------------------------------------------------------------
def test_typical_is_low_risk(model, scaler, card):
    t, s = _split("Typical operation")
    assert m.risk_band(m.failure_probability(model, scaler, t, s), card["threshold"]) == m.LOW


@pytest.mark.parametrize("preset", [p for p in m.PRESETS if p != "Typical operation"])
def test_failure_scenarios_are_high_risk(model, scaler, card, preset):
    t, s = _split(preset)
    assert m.failure_probability(model, scaler, t, s) >= card["threshold"]
    assert any(r["met"] for r in m.generation_rules(t, s)), "scenario should trigger a generation rule"


@pytest.mark.parametrize("col", SENSOR_ORDER)
@pytest.mark.parametrize("end", [0, 1])
def test_extreme_values_give_valid_probability(model, scaler, col, end):
    t, s = _split("Typical operation")
    s[col] = m.INPUT_LIMITS[col][end]
    p = m.failure_probability(model, scaler, t, s)
    assert 0.0 <= p <= 1.0


def test_sweep_shape(model, scaler):
    t, s = _split("Typical operation")
    curve = m.sweep(model, scaler, t, s, "torque_nm", points=50)
    assert len(curve) == 50 and curve["probability"].between(0, 1).all()
    assert curve["torque_nm"].min() == m.DATA_RANGE["torque_nm"][0]


# --- Validation ----------------------------------------------------------------------
def test_validation_ok():
    t, s = _split("Typical operation")
    v = m.validate_inputs(t, s, LABELS)
    assert v.ok and not v.warnings


def test_validation_missing_and_bad_type():
    t, s = _split("Typical operation")
    s["torque_nm"] = None
    v = m.validate_inputs("X", s, LABELS)
    assert not v.ok and len(v.errors) == 2


def test_validation_out_of_limits():
    t, s = _split("Typical operation")
    s["rotational_speed_rpm"] = 10_000
    assert not m.validate_inputs(t, s, LABELS).ok


def test_validation_extrapolation_and_temperature_order():
    t, s = _split("Typical operation")
    s["torque_nm"] = 95.0
    s["process_temperature_k"] = 299.0  # below air temperature (300 K)
    v = m.validate_inputs(t, s, LABELS)
    assert v.ok
    assert any("outside the training data" in w for w in v.warnings)
    assert any("not above air temperature" in w for w in v.warnings)


# --- App (headless) ------------------------------------------------------------------
def _page() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.switch_page(PAGE).run()
    assert not at.exception, at.exception
    return at


def _metric(at, label):
    return next(x.value for x in at.metric if x.label == label)


def test_page_shows_low_risk_by_default():
    at = _page()
    assert _metric(at, "Risk level") == m.LOW
    assert any("Low failure risk" in s.value for s in at.success)
    assert len(at.get("plotly_chart")) == 3  # gauge, what-if, confusion matrix


def test_page_preset_changes_result():
    at = _page()
    at.selectbox(key="pred_preset").set_value("Power too high").run()
    at.button(key="pred_load").click().run()
    assert not at.exception, at.exception
    assert _metric(at, "Risk level") == m.HIGH
    assert at.number_input(key="pred_torque_nm").value == 70.0
    assert any("High failure risk" in e.value for e in at.error)


def test_page_input_change_and_extrapolation_warning():
    at = _page()
    before = _metric(at, "Failure probability")
    at.number_input(key="pred_torque_nm").set_value(90.0).run()
    assert not at.exception, at.exception
    assert _metric(at, "Failure probability") != before
    assert any("outside the training data" in w.value for w in at.warning)


def test_page_inputs_persist_across_pages():
    at = _page()
    at.number_input(key="pred_tool_wear_min").set_value(220).run()
    at.switch_page("app_pages/1_overview.py").run()
    at.switch_page(PAGE).run()
    assert at.number_input(key="pred_tool_wear_min").value == 220


def test_what_if_selector():
    at = _page()
    at.selectbox(key="pred_sweep_col").set_value("tool_wear_min").run()
    assert not at.exception, at.exception


def test_missing_model_shows_error(monkeypatch):
    def missing(*_a, **_k):
        raise FileNotFoundError(2, "No such file", "models/failure_model.joblib")

    st.cache_resource.clear()
    st.cache_data.clear()
    monkeypatch.setattr(m, "read_model", missing)
    try:
        at = _page()
        assert any("Model files are missing" in e.value for e in at.error)
        assert len(at.metric) == 0
    finally:
        monkeypatch.undo()
        st.cache_resource.clear()
        st.cache_data.clear()
