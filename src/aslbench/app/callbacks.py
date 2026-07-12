"""Dash callbacks.

Registered via ``register(app)`` from the app factory. Dynamic model cards use
pattern-matching component ids. Long work runs in the runner's background
thread; progress is polled from atomic state files via dcc.Interval.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import ALL, MATCH, Input, Output, State, callback_context, dcc, html, no_update
from dash.dash_table import DataTable

from .. import figures, runner, scoring
from ..config import load_providers
from ..dataset import available_classes
from ..prompts import render_prompt
from ..providers import get_provider


def _provider_options() -> list[dict]:
    opts = []
    for p in load_providers():
        configured = p.credential_present()
        label = p.label if configured else f"{p.label} (missing {p.api_key_env})"
        opts.append({"label": label, "value": p.id, "disabled": not configured})
    return opts


def _make_card(index: int) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Small(f"Model {index + 1}"), width=8),
                        dbc.Col(
                            dbc.Button(
                                "x",
                                id={"type": "card-remove", "index": index},
                                size="sm",
                                color="link",
                                className="text-danger p-0 float-end",
                            ),
                            width=4,
                        ),
                    ]
                ),
                dcc.Dropdown(
                    id={"type": "card-provider", "index": index},
                    options=_provider_options(),
                    placeholder="Provider",
                    className="mb-1",
                ),
                dcc.Loading(
                    dcc.Dropdown(
                        id={"type": "card-model", "index": index},
                        options=[],
                        placeholder="Model",
                    ),
                    type="dot",
                ),
                html.Div(id={"type": "card-error", "index": index}, className="text-danger small"),
            ]
        ),
        className="mb-2",
        id={"type": "card", "index": index},
    )


def _n_classes() -> int:
    try:
        return len(available_classes())
    except Exception:
        return 36


def register(app: dash.Dash) -> None:  # noqa: C901 - a single cohesive registration block
    # -- Subset summary + gating of downstream controls --------------------
    @app.callback(
        Output("total-images", "children"),
        Output("template-picker", "disabled"),
        Output("add-model", "disabled"),
        Input("subset-size", "value"),
    )
    def _subset_summary(n_per_class):
        if not n_per_class:
            return "Total images: choose a number to continue.", True, True
        n_classes = _n_classes()
        total = int(n_per_class) * n_classes
        return (
            f"Total images: {total}  ({n_per_class} x {n_classes} classes)",
            False,
            False,
        )

    # -- Template preview ---------------------------------------------------
    @app.callback(
        Output("template-preview-collapse", "is_open"),
        Output("template-preview", "children"),
        Input("preview-toggle", "n_clicks"),
        Input("template-picker", "value"),
        State("template-preview-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def _preview(n, template_id, is_open):
        trigger = callback_context.triggered_id
        if not template_id:
            return False, ""
        text = render_prompt(template_id)
        if trigger == "preview-toggle":
            return (not is_open), text
        return is_open, text

    # -- Add / remove model cards ------------------------------------------
    @app.callback(
        Output("model-cards", "children"),
        Output("card-index", "data"),
        Input("add-model", "n_clicks"),
        Input({"type": "card-remove", "index": ALL}, "n_clicks"),
        State("model-cards", "children"),
        State("card-index", "data"),
        prevent_initial_call=True,
    )
    def _manage_cards(add_clicks, remove_clicks, children, next_index):
        children = children or []
        trigger = callback_context.triggered_id
        if trigger == "add-model":
            children = children + [_make_card(next_index)]
            return children, next_index + 1
        if isinstance(trigger, dict) and trigger.get("type") == "card-remove":
            rm_index = trigger["index"]
            children = [
                c for c in children
                if c.get("props", {}).get("id", {}).get("index") != rm_index
            ]
            return children, next_index
        return no_update, no_update

    # -- Populate model dropdown live when a provider is chosen ------------
    @app.callback(
        Output({"type": "card-model", "index": MATCH}, "options"),
        Output({"type": "card-error", "index": MATCH}, "children"),
        Input({"type": "card-provider", "index": MATCH}, "value"),
        prevent_initial_call=True,
    )
    def _list_models(provider_id):
        if not provider_id:
            return [], ""
        providers = {p.id: p for p in load_providers()}
        pcfg = providers.get(provider_id)
        if pcfg is None:
            return [], "Unknown provider"
        try:
            provider = get_provider(pcfg)
            models = provider.list_models()
        except Exception as exc:
            return [], f"Could not list models: {exc}"
        options = []
        for m in models:
            if m.vision is False:
                continue
            label = m.label + (" (vision untested)" if m.vision is None else "")
            options.append({"label": label, "value": m.id})
        if not options:
            return [], "No vision-capable models found"
        return options, ""

    # -- Enable Run button + duplicate warning -----------------------------
    @app.callback(
        Output("run-button", "disabled"),
        Output("model-warning", "children"),
        Input("subset-size", "value"),
        Input("template-picker", "value"),
        Input({"type": "card-provider", "index": ALL}, "value"),
        Input({"type": "card-model", "index": ALL}, "value"),
        Input("active-run-slug", "data"),
    )
    def _validate(subset_size, template_id, providers_vals, models_vals, active_slug):
        warning = ""
        if runner.is_run_active():
            return True, "A run is currently active."
        if not subset_size or not template_id:
            return True, warning
        if not providers_vals:
            return True, "Add at least one model."
        pairs = []
        half_complete = False
        for pv, mv in zip(providers_vals, models_vals):
            if pv and mv:
                pairs.append((pv, mv))
            elif pv or mv:
                half_complete = True
        if half_complete:
            return True, "A model card is incomplete."
        if not pairs:
            return True, "Select provider and model in at least one card."
        if len(pairs) != len(set(pairs)):
            return True, "Duplicate provider+model pairs are not allowed."
        return False, warning

    # -- Start a run --------------------------------------------------------
    @app.callback(
        Output("active-run-slug", "data"),
        Output("progress-interval", "disabled"),
        Output("run-start-msg", "children"),
        Output("main-tabs", "active_tab"),
        Input("run-button", "n_clicks"),
        State("subset-size", "value"),
        State("template-picker", "value"),
        State({"type": "card-provider", "index": ALL}, "value"),
        State({"type": "card-model", "index": ALL}, "value"),
        State("run-note", "value"),
        prevent_initial_call=True,
    )
    def _start(n, subset_size, template_id, providers_vals, models_vals, note):
        if not n:
            return no_update, no_update, no_update, no_update
        specs = [
            {"provider_id": pv, "model_id": mv}
            for pv, mv in zip(providers_vals, models_vals)
            if pv and mv
        ]
        try:
            slug = runner.start_run(
                model_specs=specs,
                template_id=template_id,
                n_per_class=int(subset_size),
                run_note=note or "",
            )
        except Exception as exc:
            return no_update, no_update, html.Span(f"Failed to start: {exc}",
                                                   className="text-danger"), no_update
        return slug, False, f"Started run {slug}", "tab-run"

    # -- Stop the active run ------------------------------------------------
    @app.callback(
        Output("stop-msg", "children"),
        Output("stop-button", "disabled", allow_duplicate=True),
        Input("stop-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def _stop(n):
        if not n:
            return no_update, no_update
        cancelled = runner.cancel_run()
        if cancelled:
            return "Stopping after each model finishes its current image...", True
        return "No active run to stop.", True

    # -- Progress polling ---------------------------------------------------
    @app.callback(
        Output("run-progress", "children"),
        Output("run-recent", "children"),
        Output("run-done-banner", "children"),
        Output("progress-interval", "disabled", allow_duplicate=True),
        Output("stop-button", "disabled"),
        Input("progress-interval", "n_intervals"),
        State("active-run-slug", "data"),
        prevent_initial_call=True,
    )
    def _poll(_n, slug):
        if not slug:
            return "", "", "", True, True
        try:
            cfg = runner.load_run_config(slug)
            state = runner.load_run_state(slug)
        except Exception:
            return "", "", "", True, True

        bars = []
        overall_completed = 0
        overall_total = 0
        recent_rows = []
        for m in cfg["models"]:
            ms = runner.load_model_state(slug, m["model_slug"])
            completed = ms.get("completed", 0)
            total = ms.get("total", 0) or cfg["n_items"]
            overall_completed += completed
            overall_total += total
            pct = int(100 * completed / total) if total else 0
            status = ms.get("status", "pending")
            bars.append(
                html.Div(
                    [
                        html.Small(f"{m['provider_label']} / {m['model_label']} "
                                   f"({completed}/{total}) [{status}]"),
                        dbc.Progress(
                            value=pct,
                            label=f"{pct}%",
                            color="success" if status == "done"
                                  else "danger" if status == "error"
                                  else "info",
                        ),
                    ],
                    className="mb-2",
                )
            )
            df = runner.load_model_results(slug, m["model_slug"])
            if not df.empty:
                tail = df.tail(3)
                for _, r in tail.iterrows():
                    err = str(r.get("provider_error") or "")
                    short_err = (err[:80] + "…") if len(err) > 80 else err
                    recent_rows.append(
                        {
                            "model": m["model_label"],
                            "item_id": r["item_id"],
                            "true": r["true_char"],
                            "predicted": r.get("predicted_char") or "",
                            "correct": bool(r["correct"]),
                            "error": short_err,
                        }
                    )
        overall_pct = int(100 * overall_completed / overall_total) if overall_total else 0
        overall = html.Div(
            [
                html.Small(f"Overall ({overall_completed}/{overall_total})"),
                dbc.Progress(value=overall_pct, label=f"{overall_pct}%", color="primary"),
            ],
            className="mb-3",
        )
        progress = [overall] + bars

        recent_table = ""
        if recent_rows:
            recent_table = DataTable(
                data=recent_rows,
                columns=[{"name": c, "id": c} for c in
                         ["model", "item_id", "true", "predicted", "correct", "error"]],
                style_cell={"fontSize": 12, "textAlign": "left"},
                style_data_conditional=[
                    {
                        "if": {"filter_query": '{error} != ""'},
                        "backgroundColor": "#f8d7da",
                        "color": "#721c24",
                    }
                ],
                page_size=15,
            )

        done = state.get("status") in ("done", "error", "cancelled")
        banner = ""
        if done:
            status = state.get("status")
            color = {"done": "success", "cancelled": "secondary"}.get(status, "warning")
            banner = dbc.Alert(
                [
                    f"Run {status}. ",
                    dbc.Button("View results", id="goto-results", color="primary", size="sm"),
                ],
                color=color,
            )
        return progress, recent_table, banner, done, done

    @app.callback(
        Output("main-tabs", "active_tab", allow_duplicate=True),
        Output("results-run-picker", "value"),
        Input("goto-results", "n_clicks"),
        State("active-run-slug", "data"),
        prevent_initial_call=True,
    )
    def _goto_results(n, slug):
        if not n:
            return no_update, no_update
        return "tab-results", slug

    # -- Results run picker options ----------------------------------------
    @app.callback(
        Output("results-run-picker", "options"),
        Output("export-run-picker", "options"),
        Input("main-tabs", "active_tab"),
    )
    def _run_options(_tab):
        runs = runner.list_runs()
        opts = [{"label": f"{r['run_slug']} [{r['status']}]", "value": r["run_slug"]} for r in runs]
        return opts, opts

    # -- Results rendering --------------------------------------------------
    @app.callback(
        Output("results-body", "children"),
        Input("results-run-picker", "value"),
    )
    def _render_results(slug):
        if not slug:
            return html.P("Select a run.", className="text-muted")
        try:
            return _build_results_view(slug)
        except Exception as exc:
            return html.Div(f"Could not render results: {exc}", className="text-danger")

    # -- Item explorer detail ----------------------------------------------
    @app.callback(
        Output("item-detail", "children"),
        Input("item-table", "active_cell"),
        State("item-table", "data"),
        State("results-run-picker", "value"),
        prevent_initial_call=True,
    )
    def _item_detail(active_cell, data, slug):
        if not active_cell or not slug:
            return ""
        row = data[active_cell["row"]]
        item_id = row["item_id"]
        cfg = runner.load_run_config(slug)
        img = html.Img(src=f"/image/{item_id}", style={"maxWidth": "100%"})
        panes = []
        for m in cfg["models"]:
            df = runner.load_model_results(slug, m["model_slug"])
            match = df[df["item_id"] == item_id]
            resp = match.iloc[0]["raw_response"] if not match.empty else "(no response)"
            panes.append(
                dbc.AccordionItem(html.Pre(resp, className="small"), title=m["model_label"])
            )
        return dbc.Row(
            [
                dbc.Col(img, width=5),
                dbc.Col(dbc.Accordion(panes, start_collapsed=True), width=7),
            ]
        )

    # -- History ------------------------------------------------------------
    def _render_history() -> html.Div:
        runs = runner.list_runs()
        if not runs:
            return html.P("No runs yet.", className="text-muted")
        _STATUS_COLOR = {
            "done": "success", "error": "danger", "cancelled": "secondary",
            "interrupted": "warning", "running": "info",
        }
        items = []
        for r in runs:
            acc = "; ".join(
                f"{k}: {v:.3f}" if v is not None else f"{k}: -"
                for k, v in r["accuracy"].items()
            )
            badge_color = _STATUS_COLOR.get(r["status"], "secondary")
            items.append(
                dbc.ListGroupItem(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Span(r["run_slug"], className="fw-bold me-2"),
                                    dbc.Badge(r["status"], color=badge_color,
                                              className="me-2"),
                                    html.Small(r["started_at"], className="text-muted"),
                                    html.Br(),
                                    html.Small(
                                        f"Models: {', '.join(r['models'])} | "
                                        f"Images: {r['n_items']} ({r['n_per_class']}/class) | "
                                        f"Accuracy: {acc or '—'}"
                                    ),
                                ],
                                width=10,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Delete",
                                    id={"type": "delete-run-btn", "slug": r["run_slug"]},
                                    color="outline-danger",
                                    size="sm",
                                ),
                                width=2,
                                className="d-flex align-items-center justify-content-end",
                            ),
                        ],
                        align="center",
                    ),
                    className="py-2",
                )
            )
        return dbc.ListGroup(items, flush=True)

    @app.callback(
        Output("history-body", "children"),
        Input("history-refresh", "n_clicks"),
        Input("main-tabs", "active_tab"),
    )
    def _history(_n, tab):
        return _render_history()

    # -- Delete run: open confirm modal ------------------------------------
    @app.callback(
        Output("delete-run-modal", "is_open"),
        Output("delete-run-confirm-msg", "children"),
        Output("delete-run-slug", "data"),
        Input({"type": "delete-run-btn", "slug": ALL}, "n_clicks"),
        Input("delete-run-cancel", "n_clicks"),
        prevent_initial_call=True,
    )
    def _open_delete_modal(delete_clicks, _cancel):
        trigger = callback_context.triggered_id
        if trigger == "delete-run-cancel" or not any(c for c in delete_clicks if c):
            return False, "", None
        if isinstance(trigger, dict) and trigger.get("type") == "delete-run-btn":
            slug = trigger["slug"]
            return (
                True,
                f"Permanently delete all data for run {slug!r}? This cannot be undone.",
                slug,
            )
        return no_update, no_update, no_update

    # -- Delete run: execute and refresh -----------------------------------
    @app.callback(
        Output("delete-run-modal", "is_open", allow_duplicate=True),
        Output("history-body", "children", allow_duplicate=True),
        Input("delete-run-confirm", "n_clicks"),
        State("delete-run-slug", "data"),
        prevent_initial_call=True,
    )
    def _execute_delete(n, slug):
        if not n or not slug:
            return no_update, no_update
        try:
            runner.delete_run(slug)
        except Exception as exc:
            return False, dbc.Alert(f"Could not delete run: {exc}", color="danger")
        return False, _render_history()

    # -- Export -------------------------------------------------------------
    @app.callback(
        Output("export-result", "children"),
        Input("export-button", "n_clicks"),
        State("export-run-picker", "value"),
        State("export-format", "value"),
        prevent_initial_call=True,
    )
    def _export(n, slug, fmt):
        if not n or not slug:
            return no_update
        from ..export import ExportError, export_run

        try:
            out = export_run(slug, fmt)
        except ExportError as exc:
            return dbc.Alert(html.Pre(str(exc), className="small mb-0"), color="danger")
        except Exception as exc:
            return dbc.Alert(f"Export failed: {exc}", color="danger")
        return dbc.Alert(f"Exported to {out}", color="success")


def _build_results_view(slug: str) -> html.Div:
    cfg = runner.load_run_config(slug)
    results = runner.model_results_list(slug)
    nonempty = [r for r in results if not r.df.empty]
    if not nonempty:
        return html.P("No results recorded for this run yet.")
    labels = [r.model_label for r in nonempty]
    colors = figures.assign_colors(labels)

    # Metadata block.
    meta = dbc.Card(
        dbc.CardBody(
            [
                html.H5("Run metadata"),
                html.Small(
                    f"{cfg['n_items']} images | {cfg.get('n_per_class')} per class | "
                    f"{cfg.get('n_classes')} classes | seed {cfg['sample_seed']} | "
                    f"template {cfg['template_id']}"
                ),
                html.Br(),
                html.Small(f"Note: {cfg.get('run_note') or '(none)'}"),
            ]
        ),
        className="mb-3",
    )

    # Comparison table.
    comp = scoring.comparison_table(nonempty)
    comp_disp = comp.copy()
    for col in comp_disp.columns:
        if col != "metric":
            comp_disp[col] = comp_disp[col].apply(
                lambda v: f"{v:.3f}" if isinstance(v, (int, float)) and v == v else "-"
            )
    comp_table = DataTable(
        data=comp_disp.to_dict("records"),
        columns=[{"name": c, "id": c} for c in comp_disp.columns],
        style_cell={"fontSize": 12, "textAlign": "left"},
    )

    figs = [
        dcc.Graph(figure=figures.accuracy_bar(nonempty, colors)),
        dcc.Graph(figure=figures.per_class_accuracy_bars(nonempty, colors)),
        dcc.Graph(figure=figures.confusion_heatmaps(nonempty, colors)),
        dcc.Graph(figure=figures.per_participant_bars(nonempty, colors)),
    ]
    pair_fig = figures.pairwise_agreement_fig(nonempty)
    if pair_fig is not None:
        figs.append(dcc.Graph(figure=pair_fig))

    # Most-confused pairs, one small table per model.
    confused_blocks = [html.H5("Most-confused pairs", className="mt-3")]
    for res in nonempty:
        mc = scoring.most_confused(res.df, top_n=8)
        if mc.empty:
            continue
        mc_disp = mc.rename(columns={"true_char": "true", "pred_char": "predicted"})
        confused_blocks.append(html.Small(res.model_label, className="fw-bold"))
        confused_blocks.append(
            DataTable(
                data=mc_disp.to_dict("records"),
                columns=[{"name": c, "id": c} for c in ["true", "predicted", "count"]],
                style_cell={"fontSize": 12, "textAlign": "left"},
                page_size=8,
            )
        )

    # Item explorer.
    matrix = scoring.outcome_matrix(nonempty)
    explorer_cols = ["item_id", "true_char"] + labels
    item_table = DataTable(
        id="item-table",
        data=matrix.to_dict("records"),
        columns=[{"name": c, "id": c} for c in explorer_cols if c in matrix.columns],
        style_cell={"fontSize": 12, "textAlign": "left"},
        page_size=15,
        style_data_conditional=[
            {"if": {"filter_query": f'{{{lab}}} = "correct"', "column_id": lab},
             "backgroundColor": "#d4edda"}
            for lab in labels
        ] + [
            {"if": {"filter_query": f'{{{lab}}} = "incorrect"', "column_id": lab},
             "backgroundColor": "#f8d7da"}
            for lab in labels
        ],
    )

    return html.Div(
        [
            meta,
            html.H5("Comparison"),
            comp_table,
            *figs,
            *confused_blocks,
            html.H5("Item explorer", className="mt-3"),
            html.Small("Click a row to see the image and each model's response."),
            item_table,
            html.Div(id="item-detail", className="mt-2"),
        ]
    )
