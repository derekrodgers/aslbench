"""Dash app factory and entry point.

Run with: python -m aslbench.app
"""

from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from flask import Response, abort

from .. import config, runner
from ..dataset import item_image_path
from .layout import build_layout


def create_app() -> dash.Dash:
    config.ensure_dirs()
    # Clear any stale run lock left by a previous process.
    runner.clear_stale_lock()

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.SANDSTONE],
        suppress_callback_exceptions=True,
        title="aslbench",
    )
    app.layout = build_layout()

    _register_image_route(app)

    from . import callbacks  # noqa: F401  (registers callbacks on import)

    callbacks.register(app)
    return app


def _register_image_route(app: dash.Dash) -> None:
    """Serve dataset images by item id, streaming from data/processed/."""

    @app.server.route("/image/<item_id>")
    def serve_image(item_id: str):  # noqa: ANN202
        # Basic path-traversal guard: reject separators.
        if "/" in item_id or ".." in item_id:
            abort(400)
        try:
            path = item_image_path(item_id)
        except Exception:
            abort(404)
        if path is None or not path.exists():
            abort(404)
        # Ensure the resolved file stays within the processed dataset folder.
        try:
            path.resolve().relative_to(config.PROCESSED_DIR.resolve())
        except ValueError:
            abort(400)
        return Response(path.read_bytes(), mimetype=config.IMAGE_MEDIA_TYPE)


def main() -> None:
    app = create_app()
    app.run(debug=False)


if __name__ == "__main__":
    main()
