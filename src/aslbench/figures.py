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
from .config import CLASSES, N_CLASSES
from .scoring import ModelResult

_PALETTE = px.colors.qualitative.Safe

# Uniform-random accuracy for the 36-way task, drawn as a reference line.
_CHANCE = 1.0 / N_CLASSES

# Shared legend style applied to every categorically-coloured figure.
_LEGEND = dict(
    title=dict(text="Model", side="left"),
    orientation="h",
    yanchor="bottom",
    y=1.0,
    xanchor="left",
    x=0,
)
_MARGIN_T = dict(t=80, b=30)


def assign_colors(model_labels: list[str]) -> dict[str, str]:
    """Assign a stable color to each model label."""
    return {label: _PALETTE[i % len(_PALETTE)] for i, label in enumerate(model_labels)}


def accuracy_bar(results: list[ModelResult], colors: dict[str, str]) -> go.Figure:
    """Grouped bars: overall accuracy and macro F1 per model, with 95% CIs."""
    fig = go.Figure()
    categories = ["Accuracy", "Macro F1"]
    for res in results:
        summ = scoring.compute_summary(res.df)
        if summ.get("n_items", 0) == 0:
            continue
        acc = summ["accuracy"]
        macro_f1 = summ["macro_f1"]
        acc_ci = summ["accuracy_ci"]
        f1_ci = summ["macro_f1_ci"]
        err_plus = [acc_ci[1] - acc, f1_ci[1] - macro_f1]
        err_minus = [acc - acc_ci[0], macro_f1 - f1_ci[0]]
        fig.add_bar(
            name=res.model_label,
            x=categories,
            y=[acc, macro_f1],
            marker_color=colors.get(res.model_label),
            error_y=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus),
            customdata=[[acc_ci[0], acc_ci[1]], [f1_ci[0], f1_ci[1]]],
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "%{x}: %{y:.3f}<br>"
                "95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]"
                "<extra></extra>"
            ),
        )
    fig.add_hline(
        y=_CHANCE,
        line_dash="dash",
        line_color="grey",
        annotation_text=f"chance (1/{N_CLASSES})",
        annotation_position="top left",
    )
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Score", range=[0, 1]),
        title=dict(text="Overall accuracy and macro F1 (95% bootstrap CI)", pad=dict(t=15, b=15)),
        legend=_LEGEND,
        margin=_MARGIN_T,
    )
    return fig


def confusion_heatmaps(
    results: list[ModelResult],
    colors: dict[str, str],
) -> go.Figure:
    """Small-multiple confusion matrices (true on y, predicted on x).

    Each matrix is row-normalized so cell values are per-true-class recall in
    [0, 1] — the standard, readable view for a 36-way task. The diagonal is the
    model's recall for each character; bright off-diagonal cells are systematic
    confusions.
    """
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
        row_sums = pivot.sum(axis=1).replace(0, np.nan)
        pivot = pivot.div(row_sums, axis=0).fillna(0)
        hovertemplate = "True: %{y}<br>Predicted: %{x}<br>Recall: %{z:.3f}<extra></extra>"
        fig.add_heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            coloraxis="coloraxis",
            hovertemplate=hovertemplate,
            row=r,
            col=c,
        )
        fig.update_xaxes(title_text="Predicted", type="category", row=r, col=c)
        fig.update_yaxes(title_text="True", type="category", autorange="reversed", row=r, col=c)
    coloraxis = dict(colorscale="Blues", cmin=0, cmax=1)
    title = "Confusion matrix (row-normalized, recall)"
    fig.update_layout(
        title=dict(text=title, pad=dict(t=15, b=15)),
        coloraxis=coloraxis,
        height=350 * rows + 120,
        margin=dict(t=80),
    )
    return fig


def per_class_diff_bars(results: list[ModelResult], colors: dict[str, str]) -> go.Figure | None:
    """Per-class accuracy difference bars for every model pair.

    Only produced when 2+ models are present. For each pair (A, B) a bar chart
    shows (accuracy_A − accuracy_B) per class, sorted from most-A-favoured on
    the left to most-B-favoured on the right. Bars above zero use model A's
    colour; bars below zero use model B's colour. The zero line is parity.

    For more than one pair the chart uses small-multiple subplots, one per pair.
    """
    if len(results) < 2:
        return None

    labels = [r.model_label for r in results]
    acc_by_model: dict[str, dict[str, float]] = {}
    for res in results:
        table = scoring.per_class_table(res.df)
        acc_by_model[res.model_label] = dict(zip(table["true_char"], table["accuracy"]))

    pairs = [
        (labels[i], labels[j])
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
    ]
    n_pairs = len(pairs)
    n_cols = min(n_pairs, 2)
    n_rows = int(np.ceil(n_pairs / n_cols))

    subplot_titles = [f"{a} \u2212 {b}" for a, b in pairs]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for idx, (a, b) in enumerate(pairs):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        acc_a = acc_by_model.get(a, {})
        acc_b = acc_by_model.get(b, {})
        classes = sorted(set(acc_a) & set(acc_b))
        diffs = {c: acc_a[c] - acc_b[c] for c in classes}
        sorted_classes = sorted(classes, key=lambda c: diffs[c], reverse=True)
        y_vals = [diffs[c] for c in sorted_classes]
        bar_colors = [colors.get(a) if d >= 0 else colors.get(b) for d in y_vals]
        fig.add_bar(
            x=sorted_classes,
            y=y_vals,
            marker_color=bar_colors,
            showlegend=False,
            hovertemplate="Class: %{x}<br>Difference: %{y:.3f}<extra></extra>",
            row=row,
            col=col,
        )
        fig.add_hline(y=0, line_color="black", line_width=0.8, row=row, col=col)
        fig.update_xaxes(title_text="Class", type="category", tickangle=0, row=row, col=col)
        fig.update_yaxes(
            title_text="Accuracy difference", range=[-1, 1], row=row, col=col
        )

    # Invisible scatter traces used only to populate the legend (one entry per model).
    for label in labels:
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=12, color=colors.get(label), symbol="square"),
            name=f"{label} better",
            showlegend=True,
            hoverinfo="skip",
        )

    height = max(350 * n_rows + 80, 400)
    # Grow the top margin with the number of models so the legend always fits
    # between the figure title and the first subplot row.
    margin_t = max(100, 45 + 20 * len(labels))
    # Place the legend bottom a fixed 15 px above the plot-area top, expressed
    # in paper coordinates (1.0 = top of plot area).
    plot_h = height - margin_t - 80
    legend_y = 1.0 + 15.0 / plot_h
    fig.update_layout(
        title=dict(text="Per-class accuracy difference (sorted by advantage)", pad=dict(t=15, b=15)),
        height=height,
        margin=dict(t=margin_t),
        legend={**_LEGEND, "y": legend_y},
    )
    return fig
