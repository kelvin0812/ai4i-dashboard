"""Plotly figure builders. Every chart is computed from the (filtered) CSV."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    FAILURE,
    FAILURE_COLORS,
    FAILURE_LABEL_COL,
    FAILURE_MODE_COLS,
    FAILURE_MODE_LABELS,
    FAILURE_ORDER,
    GENERATION_RULES,
    LABELS,
    NO_FAILURE,
    RULE_COLOR,
    TARGET_COL,
    TYPE_LABEL_COL,
    TYPE_ORDER,
)
from src.data import failure_mode_counts, failure_rate_by_type

FONT = 'Inter, system-ui, -apple-system, "Segoe UI", sans-serif'
GRID = "#f3e3e6"
AXIS = "#e5c4ca"
INK_MUTED = "#6b5a5e"
TYPE_COLORS = {"L": "#c9b8bb", "M": "#8f7a7e", "H": "#4a3a3e"}  # warm grey ramp L -> H
CATEGORY_ORDERS = {FAILURE_LABEL_COL: FAILURE_ORDER, TYPE_LABEL_COL: TYPE_ORDER}

# Diverging steel blue <-> red with a neutral midpoint (for correlations).
DIVERGING = [
    [0.0, "#2c4a80"],
    [0.25, "#93a9d3"],
    [0.5, "#f6f1f2"],
    [0.75, "#f68a9e"],
    [1.0, "#9f1239"],
]


def _style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        font=dict(family=FONT, size=13, color="#1f1215"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""
        ),
        hoverlabel=dict(font_family=FONT),
        title=dict(font=dict(size=15), x=0, xanchor="left"),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS)
    return fig


def _rule_label_position(i: int) -> str:
    return "top right" if i % 2 == 0 else "bottom right"


def add_rule_lines(fig: go.Figure, column: str, axis: str = "x") -> go.Figure:
    """Dashed reference lines for the synthetic dataset's generation rules."""
    for i, (value, text) in enumerate(GENERATION_RULES.get(column, [])):
        kwargs = dict(
            line_dash="dash",
            line_color=RULE_COLOR,
            line_width=1.5,
            annotation_text=f"Dataset generation rule · {text}",
            annotation_font=dict(size=11, color=INK_MUTED),
        )
        if axis == "x":
            fig.add_vline(x=value, annotation_position=_rule_label_position(i), **kwargs)
        else:
            fig.add_hline(y=value, annotation_position="top left", **kwargs)
    return fig


# --- Overview ----------------------------------------------------------------
def failure_rate_by_type_chart(df: pd.DataFrame) -> go.Figure:
    summary = failure_rate_by_type(df)
    fig = go.Figure(
        go.Bar(
            x=summary[TYPE_LABEL_COL].astype(str),
            y=summary["failure_rate_pct"],
            marker_color=FAILURE_COLORS[FAILURE],
            text=[f"{r:.2f}%" for r in summary["failure_rate_pct"]],
            textposition="outside",
            customdata=summary[["failures", "records"]],
            hovertemplate=(
                "Type %{x}<br>Failure rate: %{y:.2f}%"
                "<br>Failures: %{customdata[0]:,} of %{customdata[1]:,}<extra></extra>"
            ),
            width=0.5,
        )
    )
    ymax = max(summary["failure_rate_pct"].max() * 1.25, 1) if len(summary) else 1
    fig.update_layout(xaxis_title="Product type", yaxis_title="Failure rate (%)")
    fig.update_yaxes(range=[0, ymax])
    fig.update_xaxes(categoryorder="array", categoryarray=TYPE_ORDER)
    return _style(fig, height=360)


def failure_split_chart(df: pd.DataFrame) -> go.Figure:
    """Records per product type, stacked by failure status."""
    counts = (
        df.groupby([TYPE_LABEL_COL, FAILURE_LABEL_COL], observed=True)
        .size()
        .reset_index(name="records")
    )
    fig = px.bar(
        counts,
        x=TYPE_LABEL_COL,
        y="records",
        color=FAILURE_LABEL_COL,
        color_discrete_map=FAILURE_COLORS,
        category_orders=CATEGORY_ORDERS,
        labels={**LABELS, "records": "Records"},
    )
    fig.update_traces(hovertemplate="Type %{x}<br>%{y:,} records<extra>%{fullData.name}</extra>")
    fig.update_layout(bargap=0.5)
    return _style(fig, height=360)


# --- Distributions -----------------------------------------------------------
def histogram_chart(df: pd.DataFrame, column: str, normalise: bool, show_rules: bool) -> go.Figure:
    fig = px.histogram(
        df,
        x=column,
        color=FAILURE_LABEL_COL,
        color_discrete_map=FAILURE_COLORS,
        category_orders=CATEGORY_ORDERS,
        # Grouped (side-by-side) bars: overlaid bars blend blue + red into a
        # third, misleading colour.
        barmode="group",
        nbins=30,
        histnorm="percent" if normalise else None,
        labels=LABELS,
    )
    fig.update_layout(
        yaxis_title="% of records in group" if normalise else "Records",
        bargap=0.12,
        bargroupgap=0.05,
    )
    if show_rules:
        add_rule_lines(fig, column, axis="x")
    return _style(fig)


def box_chart(df: pd.DataFrame, column: str, show_rules: bool) -> go.Figure:
    fig = px.box(
        df,
        x=FAILURE_LABEL_COL,
        y=column,
        color=FAILURE_LABEL_COL,
        color_discrete_map=FAILURE_COLORS,
        category_orders=CATEGORY_ORDERS,
        labels=LABELS,
        points=False,
    )
    fig.update_layout(showlegend=False)
    if show_rules:
        add_rule_lines(fig, column, axis="y")
    return _style(fig)


# --- Relationships -----------------------------------------------------------
def _power_curve(power_w: float, rpm: np.ndarray) -> np.ndarray:
    """Torque (Nm) that gives `power_w` at each speed: P = T * n * 2π / 60."""
    return power_w * 60 / (2 * np.pi * rpm)


def scatter_chart(df: pd.DataFrame, x: str, y: str, show_rules: bool) -> go.Figure:
    # Draw failures last so the rare class is not hidden under 9,000+ points.
    ordered = df.sort_values(TARGET_COL, kind="stable")
    fig = px.scatter(
        ordered,
        x=x,
        y=y,
        color=FAILURE_LABEL_COL,
        color_discrete_map=FAILURE_COLORS,
        category_orders=CATEGORY_ORDERS,
        labels=LABELS,
        hover_data={TYPE_LABEL_COL: True, FAILURE_LABEL_COL: False},
        render_mode="webgl",
    )
    fig.update_traces(marker=dict(size=6, opacity=0.6), selector=dict(name=NO_FAILURE))
    fig.update_traces(marker=dict(size=8, opacity=0.9), selector=dict(name=FAILURE))

    if show_rules:
        add_rule_lines(fig, x, axis="x")
        add_rule_lines(fig, y, axis="y")
        pair = {x, y}
        if pair == {"rotational_speed_rpm", "torque_nm"} and not df.empty:
            rpm = np.linspace(df["rotational_speed_rpm"].min(), df["rotational_speed_rpm"].max(), 200)
            for power, text in [(3500, "< 3,500 W"), (9000, "> 9,000 W")]:
                torque = _power_curve(power, rpm)
                xs, ys = (rpm, torque) if x == "rotational_speed_rpm" else (torque, rpm)
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color=RULE_COLOR, dash="dash", width=1.5),
                        name=f"Dataset generation rule · PWF {text}",
                        hoverinfo="skip",
                    )
                )
            # Keep the axes on the data, not the curve extremes.
            fig.update_xaxes(range=_padded(df[x]))
            fig.update_yaxes(range=_padded(df[y]))
    return _style(fig, height=520)


def _padded(series: pd.Series, pad: float = 0.05) -> list:
    lo, hi = float(series.min()), float(series.max())
    span = (hi - lo) or 1
    return [lo - span * pad, hi + span * pad]


def correlation_heatmap(df: pd.DataFrame, columns: list) -> go.Figure:
    corr = df[columns].corr()
    names = [LABELS.get(c, c) for c in columns]
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=names,
            y=names,
            zmin=-1,
            zmax=1,
            colorscale=DIVERGING,
            text=np.where(np.isnan(corr.values), "", np.round(corr.values, 2).astype(str)),
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}<br>r = %{z:.2f}<extra></extra>",
            colorbar=dict(title="r", thickness=12),
            xgap=2,
            ygap=2,
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickangle=-30)
    return _style(fig, height=520)


# --- Failure modes -----------------------------------------------------------
def failure_mode_chart(df: pd.DataFrame) -> go.Figure:
    counts = failure_mode_counts(df).sort_values("count")
    labels = [FAILURE_MODE_LABELS[m] for m in counts["mode"]]
    fig = go.Figure(
        go.Bar(
            x=counts["count"],
            y=labels,
            orientation="h",
            marker_color=FAILURE_COLORS[FAILURE],
            text=[f"{c:,}" for c in counts["count"]],
            textposition="outside",
            cliponaxis=False,
            customdata=counts["pct_of_records"],
            hovertemplate="%{y}<br>%{x:,} records (%{customdata:.2f}% of filtered records)<extra></extra>",
            width=0.55,
        )
    )
    fig.update_layout(xaxis_title="Records with this failure mode", yaxis_title="")
    fig.update_xaxes(gridcolor=GRID, showgrid=True)
    fig.update_yaxes(showgrid=False)
    return _style(fig, height=360)


def failure_mode_by_type_chart(df: pd.DataFrame) -> go.Figure:
    long = (
        df.groupby(TYPE_LABEL_COL, observed=True)[FAILURE_MODE_COLS]
        .sum()
        .reset_index()
        .melt(id_vars=TYPE_LABEL_COL, var_name="mode", value_name="count")
    )
    long["mode"] = long["mode"].str.upper()
    fig = px.bar(
        long,
        x="mode",
        y="count",
        color=TYPE_LABEL_COL,
        barmode="group",
        category_orders={**CATEGORY_ORDERS, "mode": [m.upper() for m in FAILURE_MODE_COLS]},
        # Product type is ordinal (L < M < H): a neutral grey ramp, so it never
        # competes with the blue/red failure colours.
        color_discrete_map=TYPE_COLORS,
        labels={"mode": "Failure mode", "count": "Records", TYPE_LABEL_COL: "Product type"},
    )
    fig.update_traces(hovertemplate="%{x}, type %{fullData.name}: %{y:,} records<extra></extra>")
    return _style(fig, height=360)
