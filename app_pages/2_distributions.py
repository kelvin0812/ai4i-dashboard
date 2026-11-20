import pandas as pd
import streamlit as st

from src.charts import box_chart, histogram_chart
from src.config import ENGINEERED_COLS, FAILURE_ORDER, GENERATION_RULES, LABELS, SENSOR_ORDER
from src.data import medians_by_failure
from src.filters import get_filtered_data, show_filter_status
from src.style import page_header

page_header(
    "Sensor distributions",
    "Compare each sensor reading for failed and non-failed records.",
)
df = get_filtered_data()
show_filter_status(df)

c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
with c3:
    include_engineered = st.toggle(
        "Engineered features", key="dist_engineered",
        help="Adds temperature difference (process − air) and calculated power (torque × speed × 2π ÷ 60).",
    )
options = SENSOR_ORDER + (ENGINEERED_COLS if include_engineered else [])
with c1:
    column = st.selectbox("Measurement", options, format_func=LABELS.get, key="dist_column")
with c2:
    normalise = st.toggle(
        "Normalise within group", value=True, key="dist_normalise",
        help="Shows each group as % of its own records. Without it, the ~3% failure group is almost invisible.",
    )

has_rules = column in GENERATION_RULES
show_rules = False
if has_rules:
    show_rules = st.checkbox(
        "Show dataset generation rules", value=True, key="dist_rules",
        help="Thresholds from the published rules used to generate the synthetic data. They are not real safety limits.",
    )

left, right = st.columns([3, 2])
with left:
    st.plotly_chart(histogram_chart(df, column, normalise, show_rules), width="stretch")
    st.caption(
        f"Distribution of {LABELS[column]} for failed and non-failed records "
        + ("(each group scaled to 100%)." if normalise else "(raw counts).")
    )
with right:
    st.plotly_chart(box_chart(df, column, show_rules), width="stretch")
    st.caption("Box = middle 50% of records; line = median; whiskers = 1.5 × IQR.")

if show_rules:
    st.caption(
        "Dashed lines are **dataset generation rules**: thresholds the AI4I authors used to create "
        "the synthetic failures. They are not real-world safety limits."
    )

medians = medians_by_failure(df, column).reindex(FAILURE_ORDER)
counts = df["failure_label"].value_counts().reindex(FAILURE_ORDER).fillna(0).astype(int)
st.subheader("Median by failure status")
st.dataframe(
    pd.DataFrame({"Failure status": FAILURE_ORDER, "Records": counts.values,
                  f"Median {LABELS[column]}": medians.values}),
    hide_index=True, width="content",
    column_config={
        "Records": st.column_config.NumberColumn(format="localized"),
        f"Median {LABELS[column]}": st.column_config.NumberColumn(format="%.1f"),
    },
)
