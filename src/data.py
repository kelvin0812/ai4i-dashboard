"""Data loading and summary calculations.

`read_data` and the summary helpers are plain pandas (no Streamlit) so they can
be unit-tested; `load_data` wraps `read_data` with Streamlit's cache.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    DATA_PATH,
    FAILURE,
    FAILURE_LABEL_COL,
    FAILURE_MODE_COLS,
    FAILURE_ORDER,
    NO_FAILURE,
    RAW_COLUMNS,
    TARGET_COL,
    TYPE_COL,
    TYPE_LABEL_COL,
    TYPE_MAP,
    TYPE_ORDER,
)


def read_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the cleaned AI4I CSV and add display and engineered columns."""
    df = pd.read_csv(path)

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing}")

    df[TYPE_LABEL_COL] = pd.Categorical(
        df[TYPE_COL].map(TYPE_MAP), categories=TYPE_ORDER, ordered=True
    )
    df[FAILURE_LABEL_COL] = pd.Categorical(
        np.where(df[TARGET_COL] == 1, FAILURE, NO_FAILURE), categories=FAILURE_ORDER
    )

    # Engineered features (EDA only; not part of the saved scaler).
    df["temperature_difference_k"] = df["process_temperature_k"] - df["air_temperature_k"]
    df["calculated_power_w"] = df["torque_nm"] * df["rotational_speed_rpm"] * 2 * np.pi / 60
    return df


@st.cache_data(show_spinner="Loading data...")
def load_data() -> pd.DataFrame:
    """Cached full dataset (unfiltered)."""
    return read_data()


# --- Summaries ---------------------------------------------------------------
def overall_kpis(df: pd.DataFrame) -> dict:
    records = len(df)
    failures = int(df[TARGET_COL].sum())
    rate = failures / records * 100 if records else float("nan")
    return {"records": records, "failures": failures, "failure_rate_pct": rate}


def failure_rate_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Records, failures and failure rate (%) per product type present in df."""
    out = (
        df.groupby(TYPE_LABEL_COL, observed=True)[TARGET_COL]
        .agg(records="count", failures="sum")
        .reset_index()
    )
    out["failure_rate_pct"] = out["failures"] / out["records"] * 100
    return out


def medians_by_failure(df: pd.DataFrame, column: str) -> pd.Series:
    return df.groupby(FAILURE_LABEL_COL, observed=True)[column].median()


def failure_mode_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[FAILURE_MODE_COLS].sum().astype(int)
    out = counts.rename_axis("mode").reset_index(name="count")
    out["pct_of_records"] = out["count"] / len(df) * 100 if len(df) else np.nan
    return out
