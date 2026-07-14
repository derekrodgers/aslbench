"""Dash layout: sidebar controls and main tabbed area.

The benchmark uses a single fixed dataset, so there is no dataset picker; the
only subset control is how many images to sample per class. That control
defaults to blank (no valid default); the prompt template, model, and run
controls stay disabled until a number is chosen.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from ..prompts import list_templates


def dataset_summary():
    """A short description of the processed dataset, or a setup hint."""
    from ..dataset import dataset_stats

    try:
        stats = dataset_stats()
    except Exception:
        return dbc.Alert(
            "No processed dataset found. Run `python scripts/subset_dataset.py` "
            "to build data/processed/ from the raw ASL-HG dataset.",
            color="warning",
        )
    return html.Small(
        [
            f"Source dataset: {stats.n_classes} classes × "
            f"{stats.min_per_class} images × "
            f"{stats.n_participants} participants "
            f"= {stats.n_items:,} images total. ",
            "Each run samples 1–10 images per class.",
        ],
        className="text-muted",
    )


def subset_options() -> list[dict]:
    return [{"label": str(i), "value": i} for i in range(1, 11)]


def template_options() -> list[dict]:
    return [{"label": t["label"], "value": t["id"]} for t in list_templates()]


def build_sidebar() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Configure run", className="mb-2"),
                dataset_summary(),
                html.Hr(),
                html.Label("Images per class"),
                dcc.Dropdown(
                    id="subset-size",
                    options=subset_options(),
                    value=None,
                    placeholder="Choose a number (1-10)",
                ),
                html.Div(id="total-images", className="small text-muted mt-1"),
                html.Hr(),
                html.Label("Prompt template"),
                dcc.Dropdown(id="template-picker", options=template_options(), value=None,
                             placeholder="Select a template", disabled=True),
                dbc.Button("Preview template", id="preview-toggle", size="sm",
                           color="link", className="p-0 mt-1"),
                dbc.Collapse(
                    html.Pre(id="template-preview", className="small bg-light p-2 mt-1"),
                    id="template-preview-collapse",
                    is_open=False,
                ),
                html.Hr(),
                html.Label("Models"),
                html.Div(id="model-cards", children=[]),
                dbc.Button("+ Add model", id="add-model", size="sm", color="secondary",
                           className="mt-2", disabled=True),
                html.Div(id="model-warning", className="text-danger small mt-2"),
                html.Hr(),
                dbc.Input(id="run-note", placeholder="Optional run note", className="mb-2"),
                dbc.Button("Run benchmark", id="run-button", color="primary",
                           disabled=True, className="w-100"),
                html.Div(id="run-start-msg", className="small mt-2"),
                # Bookkeeping stores.
                dcc.Store(id="card-index", data=0),
                dcc.Store(id="active-run-slug", data=None),
            ]
        ),
        className="h-100",
    )


def build_run_tab() -> html.Div:
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(html.H4("Run progress", className="mt-3"), width="auto"),
                    dbc.Col(
                        dbc.Button("Stop run", id="stop-button", color="danger",
                                   size="sm", disabled=True, className="mt-3"),
                        width="auto",
                    ),
                ],
                align="center",
                justify="between",
            ),
            html.Div(id="stop-msg", className="small text-danger"),
            html.Div(id="run-progress"),
            html.Hr(),
            html.H5("Recent completions"),
            html.Div(id="run-recent"),
            html.Div(id="run-done-banner", className="mt-3"),
            dcc.Interval(id="progress-interval", interval=1000, disabled=True),
        ]
    )


def build_results_tab() -> html.Div:
    return html.Div(
        [
            html.H4("Results", className="mt-3"),
            html.Label("Run"),
            dcc.Dropdown(id="results-run-picker", options=[], value=None),
            dcc.Loading(
                html.Div(id="results-body", style={"minHeight": "200px"}),
                className="mt-3",
            ),
            dcc.Store(id="_scroll-sink"),
        ]
    )


def build_history_tab() -> html.Div:
    return html.Div(
        [
            html.H4("History", className="mt-3"),
            dbc.Button("Refresh", id="history-refresh", size="sm", color="secondary",
                       className="mb-2"),
            html.Div(id="history-body"),
            # Confirm-delete modal
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Delete run?")),
                    dbc.ModalBody(html.P(id="delete-run-confirm-msg")),
                    dbc.ModalFooter(
                        [
                            dbc.Button("Delete", id="delete-run-confirm",
                                       color="danger", className="me-2"),
                            dbc.Button("Cancel", id="delete-run-cancel",
                                       color="secondary"),
                        ]
                    ),
                ],
                id="delete-run-modal",
                is_open=False,
            ),
            dcc.Store(id="delete-run-slug", data=None),
        ]
    )


def build_export_tab() -> html.Div:
    return html.Div(
        [
            html.H4("Export", className="mt-3"),
            html.Label("Run"),
            dcc.Dropdown(id="export-run-picker", options=[], value=None),
            html.Label("Format", className="mt-2"),
            dcc.RadioItems(
                id="export-format",
                options=[{"label": " PDF", "value": "pdf"}, {"label": " HTML", "value": "html"}],
                value="html",
                inline=True,
            ),
            html.Br(),
            dbc.Button("Export", id="export-button", color="primary", className="mt-2"),
            dcc.Loading(html.Div(id="export-result", className="mt-3")),
        ]
    )


def build_layout() -> dbc.Container:
    return dbc.Container(
        [
            html.H2("aslbench", className="my-3"),
            html.P("Frontier VLM recognition of ASL fingerspelling handshapes.",
                   className="text-muted"),
            dbc.Row(
                [
                    dbc.Col(build_sidebar(), width=3),
                    dbc.Col(
                        dbc.Tabs(
                            [
                                dbc.Tab(build_run_tab(), label="Run", tab_id="tab-run"),
                                dbc.Tab(build_results_tab(), label="Results", tab_id="tab-results"),
                                dbc.Tab(build_history_tab(), label="History", tab_id="tab-history"),
                                dbc.Tab(build_export_tab(), label="Export", tab_id="tab-export"),
                            ],
                            id="main-tabs",
                            active_tab="tab-run",
                        ),
                        width=9,
                    ),
                ]
            ),
        ],
        fluid=True,
    )
