"""AI4I 2020 predictive maintenance dashboard — single entry point.

Run with:  streamlit run app.py

Page files live in app_pages/ (not pages/) on purpose: Streamlit treats a
folder named pages/ as legacy auto-discovered pages, which can clash with
st.navigation and skip this file's shared sidebar on reruns.
"""

import streamlit as st

from src.filters import render_sidebar_filters
from src.style import apply_theme

st.set_page_config(
    page_title="AI4I Predictive Maintenance",
    page_icon=":material/precision_manufacturing:",
    layout="wide",
)
apply_theme()

PAGES_DIR = "app_pages"
pages = {
    "Explore": [
        st.Page(f"{PAGES_DIR}/1_overview.py", title="Overview", icon=":material/dashboard:", default=True),  # served at "/"
        st.Page(f"{PAGES_DIR}/2_distributions.py", title="Sensor distributions", icon=":material/bar_chart:", url_path="distributions"),
        st.Page(f"{PAGES_DIR}/3_relationships.py", title="Sensor relationships", icon=":material/scatter_plot:", url_path="relationships"),
        st.Page(f"{PAGES_DIR}/4_failure_modes.py", title="Failure modes", icon=":material/report:", url_path="failure-modes"),
    ],
    "Model": [
        st.Page(f"{PAGES_DIR}/5_prediction.py", title="Failure Risk Prediction", icon=":material/online_prediction:", url_path="prediction"),
    ],
}

nav = st.navigation(pages)

# Shared filters: rendered on every run, before the page, so every page sees
# the same session_state selection.
render_sidebar_filters()

nav.run()
