"""Sidebar filters shared by every page.

The widgets are rendered once in app.py (before the page runs), and their
values live in st.session_state under the keys below, so the selection
survives navigation between pages. Pages call `get_filtered_data()`.
"""

import math

import pandas as pd
import streamlit as st

from src.config import (
    FAILURE,
    FAILURE_LABEL_COL,
    LABELS,
    NO_FAILURE,
    SENSOR_ORDER,
    TYPE_LABEL_COL,
    TYPE_ORDER,
)
from src.data import load_data

STATUS_ALL = "All records"
STATUS_FAILURE = "Failure only"
STATUS_NO_FAILURE = "No failure only"
STATUS_OPTIONS = [STATUS_ALL, STATUS_FAILURE, STATUS_NO_FAILURE]

TYPES_KEY = "filter_types"
STATUS_KEY = "filter_status"

# Integer sensors get integer sliders; the others step by 0.1.
INTEGER_SENSORS = {"rotational_speed_rpm", "tool_wear_min"}
_EPS = 1e-6  # guards against float rounding at slider ends


def _range_key(col: str) -> str:
    return f"filter_range_{col}"


def sensor_bounds(df: pd.DataFrame, col: str) -> tuple:
    lo, hi = df[col].min(), df[col].max()
    if col in INTEGER_SENSORS:
        return int(math.floor(lo)), int(math.ceil(hi))
    return math.floor(lo * 10) / 10, math.ceil(hi * 10) / 10


def default_filters(df: pd.DataFrame) -> dict:
    defaults = {TYPES_KEY: list(TYPE_ORDER), STATUS_KEY: STATUS_ALL}
    for col in SENSOR_ORDER:
        defaults[_range_key(col)] = sensor_bounds(df, col)
    return defaults


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Pure function: return the rows of df matching a filter dict."""
    mask = df[TYPE_LABEL_COL].isin(filters[TYPES_KEY])

    status = filters[STATUS_KEY]
    if status == STATUS_FAILURE:
        mask &= df[FAILURE_LABEL_COL] == FAILURE
    elif status == STATUS_NO_FAILURE:
        mask &= df[FAILURE_LABEL_COL] == NO_FAILURE

    for col in SENSOR_ORDER:
        lo, hi = filters[_range_key(col)]
        mask &= df[col].between(lo - _EPS, hi + _EPS)
    return df.loc[mask]


def current_filters() -> dict:
    df = load_data()
    defaults = default_filters(df)
    return {k: st.session_state.get(k, v) for k, v in defaults.items()}


def get_filtered_data() -> pd.DataFrame:
    return apply_filters(load_data(), current_filters())


def _reset_filters() -> None:
    for key, value in default_filters(load_data()).items():
        st.session_state[key] = value


def render_sidebar_filters() -> None:
    df = load_data()
    for key, value in default_filters(df).items():
        st.session_state.setdefault(key, value)

    with st.sidebar:
        st.header("Filters")
        st.caption("Apply to every page.")
        st.multiselect(
            "Product type",
            TYPE_ORDER,
            key=TYPES_KEY,
            help="L = low, M = medium, H = high quality variant.",
        )
        st.radio("Failure status", STATUS_OPTIONS, key=STATUS_KEY)

        st.markdown("**Sensor ranges**")
        for col in SENSOR_ORDER:
            lo, hi = sensor_bounds(df, col)
            step = 1 if col in INTEGER_SENSORS else 0.1
            st.slider(LABELS[col], min_value=lo, max_value=hi, step=step, key=_range_key(col))

        st.button("Reset filters", on_click=_reset_filters, width="stretch")

        n = len(get_filtered_data())
        st.caption(f"Showing **{n:,}** of {len(df):,} records")


def active_filter_summary() -> str:
    """Short text describing which filters differ from the defaults."""
    df = load_data()
    defaults = default_filters(df)
    f = current_filters()
    parts = []
    if list(f[TYPES_KEY]) != defaults[TYPES_KEY]:
        parts.append("Type: " + (", ".join(f[TYPES_KEY]) or "none"))
    if f[STATUS_KEY] != STATUS_ALL:
        parts.append(f[STATUS_KEY])
    for col in SENSOR_ORDER:
        key = _range_key(col)
        if tuple(f[key]) != tuple(defaults[key]):
            lo, hi = f[key]
            parts.append(f"{LABELS[col]}: {lo:g}–{hi:g}")
    return " · ".join(parts)


def show_filter_status(filtered: pd.DataFrame) -> None:
    """Caption under each page title; stops the page if nothing matches."""
    summary = active_filter_summary()
    total = len(load_data())
    if summary:
        st.caption(f"Filtered view: {len(filtered):,} of {total:,} records · {summary}")
    else:
        st.caption(f"All {total:,} records (no filters applied)")

    if filtered.empty:
        st.warning(
            "No data: no records match the current filters. "
            "Widen the sensor ranges, select a product type, or press **Reset filters**.",
            icon=":material/filter_alt_off:",
        )
        st.stop()
