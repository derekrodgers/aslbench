"""Quarto render invocation and output file naming.

The run folder is the intermediate: report.qmd reads it via the ``run_dir``
parameter at render time. Rendered output is moved into exports/.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Literal

from . import config
from .runner import run_dir


class ExportError(RuntimeError):
    """Raised when quarto render fails; carries the render log."""


def export_run(run_slug: str, fmt: Literal["pdf", "html"]) -> Path:
    """Render a run report to PDF or HTML and return the output path."""
    config.ensure_dirs()
    rdir = run_dir(run_slug)
    if not rdir.exists():
        raise FileNotFoundError(f"No run folder for {run_slug}")

    to = "typst" if fmt == "pdf" else "html"
    ext = "pdf" if fmt == "pdf" else "html"
    out_name = f"{run_slug}.{ext}"
    dest = config.EXPORTS_DIR / out_name

    # Run Quarto from EXPORTS_DIR so that any CSS/JS support files (report_files/)
    # are written adjacent to the HTML output rather than being lost when the file
    # is moved.  The --output flag is a bare filename so Quarto writes it into the
    # working directory (EXPORTS_DIR).
    cmd = [
        "quarto",
        "render",
        str(config.REPORT_QMD),
        "--to",
        to,
        "-P",
        f"run_dir:{rdir.resolve()}",
        "--output",
        out_name,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=config.EXPORTS_DIR)
    except FileNotFoundError as exc:
        raise ExportError("quarto executable not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        log = (exc.stdout or "") + "\n" + (exc.stderr or "")
        if fmt == "pdf" and "typst" in log.lower():
            log += (
                "\nPDF export uses Quarto's bundled Typst engine. If Typst is "
                "unavailable, run `quarto install tinytex` and retry."
            )
        raise ExportError(log.strip()) from exc

    # Quarto writes to cwd (EXPORTS_DIR) with a bare --output filename; also check
    # the QMD directory as a fallback for older Quarto behaviour.
    if dest.exists():
        return dest
    fallback = config.REPORT_QMD.parent / out_name
    if fallback.exists():
        shutil.move(str(fallback), str(dest))
        return dest
    raise ExportError(f"Quarto reported success but {out_name} was not found")
