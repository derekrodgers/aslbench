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

    # Run Quarto from the report/ directory (where the QMD lives) so that the
    # CSS/JS support tree (report_files/) is written alongside the output file.
    # embed-resources then resolves those paths correctly and inlines everything
    # into the HTML.  Afterwards we move the result to exports/.
    report_dir = config.REPORT_QMD.parent
    cmd = [
        "quarto",
        "render",
        str(config.REPORT_QMD),
        "--to",
        to,
        "-P",
        f"run_dir:{rdir.resolve()}",
        "-P",
        f"output_fmt:{to}",
        "--output",
        out_name,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=report_dir)
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

    # Move the rendered file from report/ to exports/.
    tmp_out = report_dir / out_name
    if tmp_out.exists():
        shutil.move(str(tmp_out), str(dest))
    elif not dest.exists():
        raise ExportError(f"Quarto reported success but {out_name} was not found")

    # Remove the support-files directory; embed-resources has inlined everything.
    report_files = report_dir / "report_files"
    if report_files.exists():
        shutil.rmtree(report_files)

    return dest
