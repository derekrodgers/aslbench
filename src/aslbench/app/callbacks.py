"""Dash callbacks.

Registered via ``register(app)`` from the app factory. Dynamic model cards use
pattern-matching component ids. Long work runs in the runner's background
thread; progress is polled from atomic state files via dcc.Interval.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
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


def _item_table_base_styles(labels: list[str]) -> list[dict]:
    """Conditional styles for the item-explorer DataTable (colour by outcome)."""
    return (
        [{"if": {"filter_query": f'{{{lab}}} = "correct"', "column_id": lab},
          "backgroundColor": "#d4edda"} for lab in labels]
        + [{"if": {"filter_query": f'{{{lab}}} = "incorrect"', "column_id": lab},
            "backgroundColor": "#f8d7da"} for lab in labels]
        + [{"if": {"filter_query": f'{{{lab}}} = "parse-failure"', "column_id": lab},
            "backgroundColor": "#e2e3e5", "color": "#6c757d"} for lab in labels]
        + [{"if": {"filter_query": f'{{{lab}}} = "error"', "column_id": lab},
            "backgroundColor": "#fff3cd", "color": "#856404"} for lab in labels]
    )


# Applied to every DataTable to suppress the default focused-cell ring.
_NO_CELL_FOCUS = [{
    "selector": "td.focused",
    "rule": (
        "-webkit-box-shadow: none !important;"
        " box-shadow: none !important;"
        " outline: none !important;"
        " background-color: inherit !important;"
    ),
}]


# Layperson-friendly definitions of the comparison-table metrics, shown as a
# bulleted list beneath the table (mirrors the report).
_METRIC_DEFINITIONS = [
    ("Accuracy", "the share of images the model labelled correctly, from 0 to 1 "
     "; simply how often it is right."),
    ("Macro F1", "the model's reliability averaged across all 36 characters, "
     "giving each character equal weight and blending how often its guesses are "
     "right (precision) with how often it finds each character (recall). It "
     "rewards models that do well on every character, not just the easy ones."),
    ("MCC", "the Matthews correlation coefficient: a single overall quality "
     "score from -1 to +1 that accounts for every kind of mistake at once. "
     "0 means no better than random guessing and 1 is perfect."),
    ("Parse failure rate", "how often the model's reply could not be read as a "
     "valid answer (it did not clearly name one of the 36 characters)."),
    ("Provider error rate", "how often the model's API failed to return any "
     "response at all (timeouts or errors)."),
    ("Chance (1/36)", "a reference column showing what each score would be for a "
     "model that guesses at random."),
]

_MCNEMAR_NOTE = (
    "What this answers: whether two models have genuinely different accuracy, "
    "judged only on the images where they disagree. Because every model saw the "
    "exact same images, we compare them image-by-image. The test looks only at "
    "discordant items, that is, images where one model was right and the other was "
    "wrong. Images they both got right or both got wrong say nothing about which "
    "is better, so they are set aside. If those disagreements are far more "
    "lopsided toward one model than a coin flip would explain, the difference is "
    "judged real. In the table, only_a_correct and only_b_correct count the "
    "discordant images each model won, better names the model ahead on them, and "
    "a p_value below 0.05 means the gap is statistically significant (unlikely "
    "to be luck)."
)


def register(app: dash.Dash) -> None:  # noqa: C901 - a single cohesive registration block
    # -- Auto-scroll to item detail when a row is clicked ------------------
    app.clientside_callback(
        """
        function(children) {
            if (children) {
                window.setTimeout(function() {
                    var el = document.getElementById('item-detail');
                    if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
                }, 150);
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("_scroll-sink", "data"),
        Input("item-detail", "children"),
        prevent_initial_call=True,
    )

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
                tail = df.tail(12)
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
        Output("item-table", "style_data_conditional"),
        Input("item-table", "active_cell"),
        State("item-table", "data"),
        State("results-run-picker", "value"),
        prevent_initial_call=True,
    )
    def _item_detail(active_cell, data, slug):
        _all_cols = list(data[0].keys()) if data else []
        _labels = [c for c in _all_cols if c not in ("item_id", "true_char")]
        row_styles = _item_table_base_styles(_labels)
        if active_cell:
            r = active_cell["row"]
            row_styles = row_styles + [
                {"if": {"row_index": r},
                 "borderTop": "2px solid #0d6efd",
                 "borderBottom": "2px solid #0d6efd"},
            ]
            if _all_cols:
                row_styles.append({"if": {"row_index": r, "column_id": _all_cols[0]},
                                   "borderLeft": "2px solid #0d6efd"})
                row_styles.append({"if": {"row_index": r, "column_id": _all_cols[-1]},
                                   "borderRight": "2px solid #0d6efd"})
        if not active_cell or not slug:
            return "", row_styles
        row = data[active_cell["row"]]
        item_id = row["item_id"]
        cfg = runner.load_run_config(slug)
        img = html.Img(src=f"/image/{item_id}", style={"maxWidth": "100%"})
        panes = []
        for m in cfg["models"]:
            df = runner.load_model_results(slug, m["model_slug"])
            match = df[df["item_id"] == item_id]
            if match.empty:
                summary = html.Small("(no data)", className="text-muted")
                thinking_text = ""
                resp = ""
            else:
                r = match.iloc[0]
                predicted = r.get("predicted_char")
                parse_ok = bool(r.get("parse_ok", False))
                correct = bool(r.get("correct", False))
                err = str(r.get("provider_error") or "")
                if err:
                    badge = dbc.Badge("error", color="warning", className="me-1")
                    detail = html.Small(err[:120], className="text-warning")
                elif not parse_ok:
                    badge = dbc.Badge("parse-failure", color="secondary", className="me-1")
                    detail = html.Small("No valid ANSWER: X found", className="text-muted")
                elif correct:
                    badge = dbc.Badge(f"✓ {predicted}", color="success", className="me-1")
                    detail = html.Small(f"Correct — predicted {predicted!r}", className="text-success")
                else:
                    badge = dbc.Badge(f"✗ {predicted}", color="danger", className="me-1")
                    detail = html.Small(
                        f"Wrong — predicted {predicted!r}, true was {r['true_char']!r}",
                        className="text-danger",
                    )
                in_tok = r.get("input_tokens")
                out_tok = r.get("output_tokens")
                tok_info = ""
                if in_tok is not None and out_tok is not None:
                    tok_info = html.Small(
                        f" · {int(in_tok)} in / {int(out_tok)} out tokens",
                        className="text-muted ms-2",
                    )
                summary = html.Div([badge, detail, tok_info], className="mb-1")

                # Thinking trace — stored separately by newer runs or extracted
                # from <think>…</think> tags for servers that embed them inline.
                thinking_text = str(r.get("thinking") or "").strip()
                resp = str(r.get("raw_response") or "")
                # Fallback for old runs: extract <think> block from raw_response.
                if not thinking_text and resp.lstrip().startswith("<think>"):
                    import re as _re
                    _m = _re.match(r"<think>(.*?)</think>\s*", resp.lstrip(),
                                   _re.DOTALL | _re.IGNORECASE)
                    if _m:
                        thinking_text = _m.group(1).strip()
                        resp = resp.lstrip()[_m.end():].strip()

            _pre_style = {"maxHeight": "600px", "overflowY": "auto",
                          "whiteSpace": "pre-wrap", "wordBreak": "break-word"}
            body_parts: list = [summary]
            if thinking_text:
                body_parts.append(
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                html.Pre(thinking_text, className="small",
                                         style={**_pre_style, "maxHeight": "400px"}),
                                title="💭 Thinking",
                                item_id="thinking",
                            )
                        ],
                        # active_item omitted → collapsed by default
                        className="mb-2",
                        always_open=False,
                    )
                )
            body_parts.append(html.Pre(resp, className="small mt-1", style=_pre_style))
            panes.append(
                dbc.AccordionItem(
                    html.Div(body_parts),
                    title=m["model_label"],
                    item_id=f"item-{len(panes)}",
                )
            )
        # Build list of item IDs so all panes start open.
        open_ids = [f"item-{i}" for i in range(len(panes))]
        return dbc.Row(
            [
                dbc.Col(img, width=5),
                dbc.Col(
                    dbc.Accordion(panes, active_item=open_ids, always_open=True),
                    width=7,
                ),
            ]
        ), row_styles

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
                html.Ul(
                    [
                        html.Li([html.B("Run: "), cfg['run_slug']]),
                        html.Li([html.B("Started: "), cfg['started_at']]),
                        html.Li([html.B("Images per class: "), cfg.get('n_per_class')]),
                        html.Li([html.B("Classes: "), cfg.get('n_classes')]),
                        html.Li([html.B("Total images: "), cfg['n_items']]),
                        html.Li([html.B("Sample seed: "), cfg['sample_seed']]),
                        html.Li([html.B("Prompt template: "), cfg['template_id']]),
                        html.Li([html.B("Note: "), cfg.get('run_note') or '(none)']),
                    ],
                    className="mb-0",
                ),
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
        css=_NO_CELL_FOCUS,
    )
    metric_defs = html.Ul(
        [html.Li([html.B(f"{term}: "), text]) for term, text in _METRIC_DEFINITIONS],
        className="small text-muted mt-2",
    )

    figs = [
        dcc.Graph(figure=figures.accuracy_bar(nonempty, colors)),
    ]

    # Lay out figures two-per-row.
    fig_rows = []
    for i in range(0, len(figs), 2):
        pair = figs[i : i + 2]
        cols = [dbc.Col(f, width=6) for f in pair]
        if len(cols) == 1:
            cols[0] = dbc.Col(pair[0], width=12)
        fig_rows.append(dbc.Row(cols, className="mb-2"))

    # Per-class accuracy difference chart (only shown with 2+ models).
    diff_fig = figures.per_class_diff_bars(nonempty, colors) if len(nonempty) >= 2 else None
    diff_block = (
        [dcc.Graph(figure=diff_fig, className="mt-2")] if diff_fig is not None else []
    )

    # Per-class accuracy bars (comes after diff).
    per_class_block = [dcc.Graph(figure=figures.per_class_accuracy_bars(nonempty, colors))]

    # Confusion matrix (row-normalized recall; classes are always balanced).
    confusion_block = html.Div(
        [
            html.H5("Confusion matrix", className="mt-3"),
            dcc.Graph(figure=figures.confusion_heatmaps(nonempty, colors)),
            html.Small(
                "How to read this: each row is the true character and each "
                "column is what the model guessed. Cells are row-normalized, so "
                "a value is the fraction of that character's images that received the correct "
                "guess; the diagonal is the model's recall for each character. A "
                "bright diagonal means accurate recognition, while bright "
                "off-diagonal cells show characters the model routinely mixes up "
                "(e.g. reading 'O' as '0').",
                className="text-muted",
            ),
        ]
    )

    # McNemar test: pairwise significance of accuracy differences on shared items.
    mcnemar_block = []
    if len(nonempty) >= 2:
        mt = scoring.mcnemar_table(nonempty)
        if not mt.empty:
            mt_disp = mt.copy()
            mt_disp["p_value"] = mt_disp["p_value"].apply(lambda v: f"{v:.4f}")
            mcnemar_block = [
                html.H5("McNemar test", className="mt-3"),
                DataTable(
                    data=mt_disp.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in mt_disp.columns],
                    style_cell={"fontSize": 12, "textAlign": "left"},
                    css=_NO_CELL_FOCUS,
                ),
                html.Small(
                    _MCNEMAR_NOTE,
                    className="text-muted d-block mt-1",
                ),
            ]

    # Most-confused pairs, two per row.
    confused_items = []
    for res in nonempty:
        mc = scoring.most_confused(res.df, top_n=8)
        if mc.empty:
            continue
        mc_disp = mc.rename(columns={"true_char": "true", "pred_char": "predicted"})
        confused_items.append(html.Div(
            [
                html.Small(res.model_label, className="fw-bold"),
                DataTable(
                    data=mc_disp.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in ["true", "predicted", "count"]],
                    style_cell={"fontSize": 12, "textAlign": "left"},
                    page_size=8,
                    css=_NO_CELL_FOCUS,
                ),
            ]
        ))

    confused_rows = []
    for i in range(0, len(confused_items), 2):
        pair = confused_items[i : i + 2]
        cols = [dbc.Col(pair[0], width=6)]
        if len(pair) == 2:
            cols.append(dbc.Col(pair[1], width=6))
        confused_rows.append(dbc.Row(cols, className="mb-2"))

    confused_blocks = [html.H5("Most-confused pairs", className="mt-3")] + confused_rows

    # Item explorer.
    matrix = scoring.outcome_matrix(nonempty)
    explorer_cols = ["item_id", "true_char"] + labels
    item_table = DataTable(
        id="item-table",
        data=matrix.to_dict("records"),
        columns=[{"name": c, "id": c} for c in explorer_cols if c in matrix.columns],
        style_cell={"fontSize": 12, "textAlign": "left"},
        page_size=15,
        css=_NO_CELL_FOCUS,
        style_data_conditional=_item_table_base_styles(labels),
    )

    return html.Div(
        [
            meta,
            html.H5("Comparison"),
            comp_table,
            metric_defs,
            *fig_rows,
            *diff_block,
            *per_class_block,
            confusion_block,
            *mcnemar_block,
            *confused_blocks,
            html.H5("Item explorer", className="mt-3"),
            html.Small("Click a row to see the image and each model's response."),
            item_table,
            html.Div(id="item-detail", className="mt-2"),
        ]
    )
