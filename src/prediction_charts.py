"""Plotly figures for the prediction page (kept apart from Geo's src/charts.py)."""

import pandas as pd
import plotly.graph_objects as go

from src.charts import _style
from src.config import FAILURE_COLORS, FAILURE, LABELS, NO_FAILURE, RULE_COLOR

RED = FAILURE_COLORS[FAILURE]
BLUE = FAILURE_COLORS[NO_FAILURE]
AMBER = "#d97706"


def risk_gauge(probability: float, threshold: float) -> go.Figure:
    """Horizontal gauge: 0-100% failure probability with the decision threshold marked."""
    pct, thr = probability * 100, threshold * 100
    colour = RED if probability >= threshold else AMBER if probability >= threshold / 2 else BLUE
    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=pct,
        gauge={
            "shape": "bullet",
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": colour, "thickness": 0.55},
            "steps": [
                {"range": [0, thr / 2], "color": "#eef2f9"},
                {"range": [thr / 2, thr], "color": "#fdf3e3"},
                {"range": [thr, 100], "color": "#fde7eb"},
            ],
            "threshold": {"line": {"color": RULE_COLOR, "width": 3}, "thickness": 0.9, "value": thr},
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig = _style(fig, height=80)
    fig.update_layout(margin=dict(l=12, r=20, t=8, b=24), title_text="")
    return fig


def sweep_chart(curve: pd.DataFrame, column: str, current: float, threshold: float) -> go.Figure:
    """Predicted failure probability across one input, the others held fixed."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve[column], y=curve["probability"] * 100, mode="lines",
        line=dict(color=RED, width=3, shape="hv"), name="Failure probability",
        hovertemplate=f"{LABELS[column]}: %{{x:,.1f}}<br>Failure probability: %{{y:.1f}}%<extra></extra>",
    ))
    fig.add_hline(y=threshold * 100, line=dict(color=RULE_COLOR, dash="dash", width=1.5),
                  annotation_text=f"Decision threshold {threshold * 100:.0f}%",
                  annotation_position="top left")
    fig.add_vline(x=current, line=dict(color=BLUE, width=2),
                  annotation_text=f"Current: {current:,.1f}", annotation_position="bottom right",
                  annotation_font_color=BLUE)
    fig.update_xaxes(title_text=LABELS[column])
    fig.update_yaxes(title_text="Failure probability (%)", range=[0, 100])
    fig = _style(fig, height=380)
    fig.update_layout(showlegend=False, title_text="")
    return fig


def confusion_chart(m: dict) -> go.Figure:
    """2x2 test-set confusion matrix from the model card."""
    z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    text = [[f"{m['tn']:,}<br>correct: no failure", f"{m['fp']:,}<br>false alarms"],
            [f"{m['fn']:,}<br>missed failures", f"{m['tp']:,}<br>failures caught"]]
    fig = go.Figure(go.Heatmap(
        z=[[0, 1], [1, 0.6]], x=["Predicted no failure", "Predicted failure"],
        y=["Actual no failure", "Actual failure"], text=text, texttemplate="%{text}",
        textfont=dict(size=14), customdata=z,
        colorscale=[[0, "#eef2f9"], [0.6, "#f8c6cf"], [1, "#fde7eb"]], showscale=False,
        hovertemplate="%{y} / %{x}: %{customdata:,}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top")
    fig = _style(fig, height=300)
    fig.update_layout(title_text="")
    return fig
