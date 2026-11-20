import streamlit as st

from src.charts import failure_rate_by_type_chart, failure_split_chart
from src.config import TYPE_LABEL_COL
from src.data import failure_rate_by_type, load_data, overall_kpis
from src.filters import get_filtered_data, show_filter_status
from src.style import page_header

page_header(
    "Machine failure overview",
    "How often machines fail, and how that differs by product quality. Use the sidebar filters to slice the data.",
)
df = get_filtered_data()
show_filter_status(df)

# --- KPIs ----------------------------------------------------------------------
kpi = overall_kpis(df)
full = overall_kpis(load_data())
c1, c2, c3 = st.columns(3)
c1.metric("Records", f"{kpi['records']:,}", border=True,
          help=f"Rows matching the filters, out of {full['records']:,}.")
c2.metric("Failures", f"{kpi['failures']:,}", border=True,
          help="Records with machine_failure = 1.")
is_filtered = kpi["records"] != full["records"]
c3.metric(
    "Failure rate",
    f"{kpi['failure_rate_pct']:.2f}%",
    delta=f"{kpi['failure_rate_pct'] - full['failure_rate_pct']:+.2f} pp vs all data" if is_filtered else None,
    delta_color="inverse",
    border=True,
    help="Failures ÷ records. The delta compares with the unfiltered rate (3.39%).",
)

# --- By product type -------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Failure rate by product type")
    st.plotly_chart(failure_rate_by_type_chart(df), width="stretch")
    st.caption(
        "Share of records in each product type that recorded a machine failure. "
        "Hover a bar for the failure and record counts."
    )
with right:
    st.subheader("Records by product type and failure status")
    st.plotly_chart(failure_split_chart(df), width="stretch")
    st.caption("Failures are rare, so the red segments are thin. Use the rate chart to compare types.")

with st.expander("Table: failure rate by product type"):
    table = failure_rate_by_type(df).rename(
        columns={TYPE_LABEL_COL: "Product type", "records": "Records",
                 "failures": "Failures", "failure_rate_pct": "Failure rate (%)"}
    )
    st.dataframe(table, hide_index=True, width="content",
                 column_config={"Records": st.column_config.NumberColumn(format="localized"),
                                "Failure rate (%)": st.column_config.NumberColumn(format="%.2f")})

# --- About the data --------------------------------------------------------------
st.subheader("About the dataset")
st.markdown(
    """
The **AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository) is a
**synthetic** dataset designed to look like industrial milling-machine data. It has
10,000 records. Each record has a product quality variant (**L**ow, **M**edium, **H**igh),
five sensor readings in physical units, a binary `machine_failure` label, and five
failure-mode labels (TWF, HDF, PWF, OSF, RNF).

Failures are rare (about 3% of records), so accuracy alone is a poor guide to model
performance. Patterns shown here are **associations within synthetic data**. They are not
evidence that changing a single reading prevents failure. The dataset has no repair costs
or downtime, so real-world savings cannot be measured from it.
"""
)
st.caption("Source: AI4I 2020 Predictive Maintenance Dataset (2020), UCI ML Repository, DOI 10.24432/C5HS5C.")
