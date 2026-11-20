"""Sanity checks: the unfiltered data must match Ilhan's EDA numbers."""

import pytest

from src.config import FAILURE, NO_FAILURE, SENSOR_ORDER, TYPE_ORDER
from src.data import failure_rate_by_type, medians_by_failure, overall_kpis, read_data
from src.filters import (
    STATUS_FAILURE,
    STATUS_KEY,
    TYPES_KEY,
    apply_filters,
    default_filters,
)


@pytest.fixture(scope="module")
def df():
    return read_data()


def test_overall(df):
    k = overall_kpis(df)
    assert k["records"] == 10_000
    assert k["failures"] == 339
    assert round(k["failure_rate_pct"], 2) == 3.39


def test_by_type(df):
    t = failure_rate_by_type(df).set_index("type_label")
    expected = {"L": (6000, 235, 3.92), "M": (2997, 83, 2.77), "H": (1003, 21, 2.09)}
    for label, (records, failures, rate) in expected.items():
        assert t.loc[label, "records"] == records
        assert t.loc[label, "failures"] == failures
        assert round(t.loc[label, "failure_rate_pct"], 2) == rate


def test_medians(df):
    torque = medians_by_failure(df, "torque_nm")
    wear = medians_by_failure(df, "tool_wear_min")
    assert round(torque[FAILURE], 1) == 53.7
    assert round(torque[NO_FAILURE], 1) == 39.9
    assert wear[FAILURE] == 165
    assert wear[NO_FAILURE] == 107


def test_default_filters_keep_everything(df):
    assert len(apply_filters(df, default_filters(df))) == len(df)


def test_filters_narrow(df):
    f = default_filters(df)
    assert len(apply_filters(df, {**f, TYPES_KEY: ["H"]})) == 1003
    assert len(apply_filters(df, {**f, STATUS_KEY: STATUS_FAILURE})) == 339
    assert len(apply_filters(df, {**f, TYPES_KEY: []})) == 0
    f2 = {**f, "filter_range_torque_nm": (50.0, 60.0)}
    out = apply_filters(df, f2)
    assert out["torque_nm"].between(50, 60).all() and 0 < len(out) < len(df)


def test_type_labels(df):
    assert set(df["type_label"].astype(str)) == set(TYPE_ORDER)


def test_scaler_order():
    from src.prediction import load_scaler

    assert list(load_scaler().feature_names_in_) == SENSOR_ORDER
