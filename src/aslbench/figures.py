"""Shared Plotly figure builders used by both the Dash app and the Quarto report.

Every figure accepts a list of scoring.ModelResult plus a color map so a model
keeps the same color across all figures. All builders render correctly for 1 to
6+ models.
"""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import scoring
from .config import CLASSES
from .scoring import ModelResult

_PALETTE = px.colors.qualitative.Safe


def assign_colors(model_labels: list[str]) -> dict[str, str]:
    """Assign a stable color to each model label."""
    return {label: _PALETTE[i % len(_PALETTE)] for i, label in enumerate(model_labels)}


def accuracy_bar(results: list[ModelResult], colors: dict[str, str]) -> go.Figure:
    """Grouped bars: overall accuracy and macro F1 per model, CI on accuracy."""
    fig = go.Figure()
    categories = ["Accuracy", "Macro F1"]
    for res in results:
        summ = scoring.compute_summary(res.df)
        if summ.get("n_items", 0) == 0:
            continue
        acc = summ["accuracy"]
        macro_f1 = summ["macro_f1"]
        ci = summ["accuracy_ci"]
        err_plus = [ci[1] - acc, 0]
        err_minus = [acc - ci[0], 0]
        fig.add_bar(
            name=res.model_label,
            x=categories,
            y=[acc, macro_f1],
            marker_color=colors.get(res.model_label),
            error_y=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus),
        )
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Score", range=[0, 1]),
        title="Overall accuracy and macro F1 (95% bootstrap CI on accuracy)",
        legend_title="Model",
    )
    return fig


def per_class_accuracy_bars(results: list[ModelResult], colors: dict[str, str]) -> go.Figure:
    """Grouped bars of per-class accuracy, one bar per model, classes on x."""
    fig = go.Figure()
    order = [c for c in CLASSES]
    for res in results:
        table = scoring.per_class_table(res.df)
        if table.empty:
            continue
        acc_map = dict(zip(table["true_char"], table["accuracy"]))
        present = [c for c in order if c in acc_map]
        fig.add_bar(
            name=res.model_label,
            x=present,
            y=[acc_map[c] for c in present],
            marker_color=colors.get(res.model_label),
        )
    fig.update_layout(
        barmode="group",
        xaxis=dict(title="Class", type="category"),
        yaxis=dict(title="Accuracy", range=[0, 1]),
        title="Per-class accuracy",
        legend_title="Model",
    )
    return fig


def confusion_heatmaps(results: list[ModelResult], colors: dict[str, str]) -> go.Figure:
    """Small-multiple confusion matrices (true on y, predicted on x)."""
    n = len(results)
    cols = min(n, 2)
    rows = int(np.ceil(n / cols)) if n else 1
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[r.model_label for r in results],
        horizontal_spacing=0.12,
        vertical_spacing=0.12,
    )
    for i, res in enumerate(results):
        r = i // cols + 1
        c = i % cols + 1
        grid = scoring.confusion_long(res.df)
        if grid.empty:
            continue
        true_classes = [cl for cl in CLASSES if cl in set(grid["true_char"])]
        pred_classes = [cl for cl in CLASSES if cl in set(grid["pred_char"])]
        if scoring.PARSE_FAIL_SYMBOL in set(grid["pred_char"]):
            pred_classes = pred_classes + [scoring.PARSE_FAIL_SYMBOL]
        pivot = (
            grid.pivot(index="true_char", columns="pred_char", values="count")
            .reindex(index=true_classes, columns=pred_classes)
            .fillna(0)
        )
        fig.add_heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            coloraxis="coloraxis",
            row=r,
            col=c,
        )
        fig.update_xaxes(title_text="Predicted", type="category", row=r, col=c)
        fig.update_yaxes(title_text="True", type="category", autorange="reversed", row=r, col=c)
    fig.update_layout(
        title="Confusion matrix (true vs predicted)",
        coloraxis=dict(colorscale="Blues"),
        height=350 * rows + 80,
    )
    return fig


def per_participant_bars(results: list[ModelResult], colors: dict[str, str]) -> go.Figure:
    """Grouped bars of accuracy per participant, one bar per model."""
    fig = go.Figure()
    for res in results:
        table = scoring.per_participant_table(res.df)
        if table.empty:
            continue
        fig.add_bar(
            name=res.model_label,
            x=table["participant"],
            y=table["accuracy"],
            marker_color=colors.get(res.model_label),
        )
    fig.update_layout(
        barmode="group",
        xaxis=dict(title="Participant", type="category"),
        yaxis=dict(title="Accuracy", range=[0, 1]),
        title="Accuracy per participant",
        legend_title="Model",
    )
    return fig


def pairwise_agreement_fig(results: list[ModelResult]) -> go.Figure | None:
    """Heatmap of the fraction of items where each pair were both correct."""
    if len(results) < 2:
        return None
    labels = [r.model_label for r in results]
    correctness = {
        r.model_label: r.df.set_index("item_id")["correct"].astype(bool) for r in results
    }
    z = np.zeros((len(labels), len(labels)))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            ca, cb = correctness[a], correctness[b]
            common = ca.index.intersection(cb.index)
            if len(common) == 0:
                z[i, j] = np.nan
            else:
                z[i, j] = float((ca.loc[common] & cb.loc[common]).mean())
    fig = go.Figure(data=go.Heatmap(z=z, x=labels, y=labels, colorscale="Greens", zmin=0, zmax=1))
    fig.update_layout(title="Pairwise both-correct fraction")
    return fig
