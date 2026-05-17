from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from src.config import MODEL_COMPARISON

TEMPLATE = "plotly_dark"
ACCENT = "#10b981"
BLUE = "#38bdf8"
YELLOW = "#f59e0b"
RED = "#ef4444"


def metric_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 42, "color": "#f8fafc"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar": {"color": "#f8fafc", "thickness": 0.16},
                "bgcolor": "rgba(15,23,42,0.9)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(16,185,129,0.75)"},
                    {"range": [35, 65], "color": "rgba(245,158,11,0.78)"},
                    {"range": [65, 100], "color": "rgba(239,68,68,0.80)"},
                ],
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 4},
                    "thickness": 0.75,
                    "value": probability * 100,
                },
            },
        )
    )
    fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def contribution_bar(contributors: pd.DataFrame) -> go.Figure:
    data = contributors.sort_values("Contribution")
    colors = data["Contribution"].map(lambda value: RED if value > 0 else ACCENT)
    fig = go.Figure(
        go.Bar(
            x=data["Contribution"],
            y=data["Feature"],
            orientation="h",
            marker_color=colors,
            text=data["Direction"],
            hovertemplate="<b>%{y}</b><br>Contribution: %{x:.3f}<br>%{text}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(
        template=TEMPLATE,
        height=320,
        xaxis_title="Model contribution",
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.16)")
    return fig


def model_comparison_table() -> pd.DataFrame:
    return pd.DataFrame(MODEL_COMPARISON).T.reset_index(names="Model")


def model_comparison_bar() -> go.Figure:
    df = model_comparison_table().melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(
        df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        template=TEMPLATE,
        color_discrete_sequence=[ACCENT, BLUE, RED],
        text=df["Score"].map(lambda value: f"{value:.2f}"),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(range=[0, 1.08], gridcolor="rgba(148,163,184,0.16)")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def confusion_matrix_chart(y_true, y_pred) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Claims"),
        x=["Legitimate", "Fraud"],
        y=["Legitimate", "Fraud"],
        color_continuous_scale=["#0f172a", "#0ea5e9", "#10b981"],
        template=TEMPLATE,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def roc_curve_chart(y_true, probabilities) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Logistic Regression", line=dict(color=ACCENT, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#64748b", dash="dash")))
    fig.update_layout(
        template=TEMPLATE,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def precision_recall_chart(y_true, probabilities) -> go.Figure:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="Precision-Recall", line=dict(color=BLUE, width=3)))
    fig.update_layout(
        template=TEMPLATE,
        xaxis_title="Recall",
        yaxis_title="Precision",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def threshold_chart(y_true, probabilities) -> go.Figure:
    thresholds = np.linspace(0.05, 0.95, 91)
    rows = []
    for threshold in thresholds:
        pred = (probabilities >= threshold).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        rows.append({"Threshold": threshold, "Precision": precision, "Recall": recall})
    df = pd.DataFrame(rows)
    fig = px.line(
        df,
        x="Threshold",
        y=["Precision", "Recall"],
        template=TEMPLATE,
        color_discrete_sequence=[BLUE, ACCENT],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Metric",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def histogram(data: pd.DataFrame, column: str, color: str | None = None) -> go.Figure:
    fig = px.histogram(
        data,
        x=column,
        color=color,
        nbins=32,
        template=TEMPLATE,
        color_discrete_sequence=[ACCENT, RED, BLUE],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=25, b=10),
    )
    return fig


def box_chart(data: pd.DataFrame, x: str, y: str) -> go.Figure:
    fig = px.box(data, x=x, y=y, color=x, template=TEMPLATE, color_discrete_sequence=[ACCENT, RED])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=10, r=10, t=25, b=10),
    )
    return fig


def fraud_rate_bar(data: pd.DataFrame, column: str) -> go.Figure:
    grouped = (
        data.groupby(column, dropna=False)["fraud_reported"]
        .mean()
        .mul(100)
        .reset_index(name="Fraud Rate")
        .sort_values("Fraud Rate", ascending=False)
    )
    fig = px.bar(grouped, x=column, y="Fraud Rate", template=TEMPLATE, color="Fraud Rate", color_continuous_scale="Teal")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=25, b=10),
    )
    return fig


def savings_chart(monthly_claims: int, avg_claim: float, fraud_rate: float, recall: float, review_cost: float) -> go.Figure:
    expected_fraud = monthly_claims * fraud_rate
    caught = expected_fraud * recall
    gross_savings = caught * avg_claim
    review_spend = monthly_claims * review_cost
    net_savings = max(gross_savings - review_spend, 0)
    df = pd.DataFrame(
        {
            "Category": ["Expected Fraud Exposure", "Fraud Prevented", "Review Cost", "Estimated Net Savings"],
            "Amount": [expected_fraud * avg_claim, gross_savings, review_spend, net_savings],
        }
    )
    fig = px.bar(df, x="Category", y="Amount", text="Amount", template=TEMPLATE, color="Category", color_discrete_sequence=[RED, ACCENT, YELLOW, BLUE])
    fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=25, b=10),
    )
    return fig
