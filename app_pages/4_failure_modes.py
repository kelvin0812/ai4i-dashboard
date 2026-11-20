import pandas as pd
import streamlit as st

from src.charts import failure_mode_by_type_chart, failure_mode_chart
from src.config import FAILURE_MODE_COLS, FAILURE_MODE_LABELS, TARGET_COL
from src.data import failure_mode_counts
from src.filters import get_filtered_data, show_filter_status
from src.style import page_header

page_header(
    "Failure modes",
    "Which kinds of failure were recorded: tool wear, heat dissipation, power, overstrain or random.",
)
df = get_filtered_data()
show_filter_status(df)

st.caption(
    "Failure-mode labels describe recorded outcomes. They are shown here for exploration only "
    "and are **never** used as prediction inputs."
)

modes_per_row = df[FAILURE_MODE_COLS].sum(axis=1)
failed = df[TARGET_COL] == 1
c1, c2, c3 = st.columns(3)
c1.metric("Records with any failure mode", f"{int((modes_per_row > 0).sum()):,}", border=True)
c2.metric("Multi-mode failures", f"{int(((modes_per_row > 1) & failed).sum()):,}", border=True,
          help="Failures with more than one failure-mode flag set.")
c3.metric("Failures with no mode label", f"{int(((modes_per_row == 0) & failed).sum()):,}", border=True,
          help="Records labelled machine_failure = 1 but with no failure-mode flag set.")

left, right = st.columns(2)
with left:
    st.subheader("Count of each failure mode")
    st.plotly_chart(failure_mode_chart(df), width="stretch")
    st.caption("A record can have more than one mode, so these counts can add up to more than the number of failures.")
with right:
    st.subheader("Failure modes by product type")
    st.plotly_chart(failure_mode_by_type_chart(df), width="stretch")
    st.caption("Counts per product type (grey shades: L light → H dark). L is the most common type, so its counts are highest.")

with st.expander("Table: failure mode counts"):
    t = failure_mode_counts(df)
    t["mode"] = t["mode"].map(FAILURE_MODE_LABELS)
    st.dataframe(
        t.rename(columns={"mode": "Failure mode", "count": "Records", "pct_of_records": "% of filtered records"}),
        hide_index=True, width="content",
        column_config={"% of filtered records": st.column_config.NumberColumn(format="%.2f")},
    )

with st.expander("What the modes mean"):
    st.table(
        pd.DataFrame({
            "Mode": [FAILURE_MODE_LABELS[m] for m in FAILURE_MODE_COLS],
            "Description": [
                "Tool reaches its end of life through wear.",
                "Too little heat dissipation (small air/process temperature difference at low speed).",
                "Power (torque × speed) outside the allowed band.",
                "Tool wear × torque too high for the product type.",
                "Random failure, independent of the process parameters.",
            ],
        }).set_index("Mode")
    )
