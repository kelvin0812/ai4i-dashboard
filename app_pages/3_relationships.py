import streamlit as st

from src.charts import correlation_heatmap, scatter_chart
from src.config import ENGINEERED_COLS, LABELS, SENSOR_ORDER, TARGET_COL
from src.filters import get_filtered_data, show_filter_status
from src.style import page_header

page_header(
    "Sensor relationships",
    "See how pairs of readings move together and where failures cluster.",
)
df = get_filtered_data()
show_filter_status(df)

PRESETS = {
    "Speed vs torque": ("rotational_speed_rpm", "torque_nm"),
    "Air vs process temperature": ("air_temperature_k", "process_temperature_k"),
    "Speed vs temperature difference": ("rotational_speed_rpm", "temperature_difference_k"),
    "Custom": None,
}

st.subheader("Scatter plot")
preset = st.segmented_control("View", list(PRESETS), default="Speed vs torque", key="rel_preset")
preset = preset or "Speed vs torque"
c2, c3, c4 = st.columns([1, 1, 1], vertical_alignment="bottom")
axis_options = SENSOR_ORDER + ENGINEERED_COLS
if PRESETS[preset]:
    x_col, y_col = PRESETS[preset]
else:
    with c2:
        x_col = st.selectbox("X axis", axis_options, format_func=LABELS.get, index=2, key="rel_x")
    with c3:
        y_col = st.selectbox("Y axis", axis_options, format_func=LABELS.get, index=3, key="rel_y")
with c4:
    show_rules = st.checkbox("Dataset generation rules", value=True, key="rel_rules",
                             help="Reference lines from the rules used to generate the synthetic data.")

st.plotly_chart(scatter_chart(df, x_col, y_col, show_rules), width="stretch")
captions = {
    "Speed vs torque": "Speed and torque are strongly negatively related. Failures cluster at the "
    "high-torque / low-speed end and at the extremes of the power band (dashed curves).",
    "Air vs process temperature": "Air and process temperature rise together. The process temperature "
    "in this synthetic dataset was generated from the air temperature.",
    "Speed vs temperature difference": "Records below both dashed lines match the heat dissipation "
    "failure (HDF) generation rule.",
}
st.caption(
    captions.get(preset, f"{LABELS[y_col]} against {LABELS[x_col]}.")
    + " Red points (failures) are drawn on top. Dashed lines are **dataset generation rules**, "
    "not real safety limits."
)

st.divider()
st.subheader("Correlation heatmap")
include_eng = st.toggle("Include engineered features", key="rel_heat_eng")
cols = SENSOR_ORDER + (ENGINEERED_COLS if include_eng else []) + [TARGET_COL]
# Drop columns with no variation in the filtered data (correlation is undefined).
constant = [c for c in cols if df[c].nunique() < 2]
cols = [c for c in cols if c not in constant]

if len(df) < 3 or len(cols) < 2:
    st.info("Not enough varied records under the current filters to compute correlations.")
else:
    st.plotly_chart(correlation_heatmap(df, cols), width="stretch")
    note = (
        "Pearson correlation (r) between measurements and machine failure (0/1). Red = move together, "
        "blue = move in opposite directions. Low r can still hide non-linear patterns, such as failures "
        "at both low and high speed."
    )
    if constant:
        note += " Not shown (constant under the current filters): " + ", ".join(LABELS.get(c, c) for c in constant) + "."
    st.caption(note)
