"""Prompt template loading for aslbench.

Templates are Markdown files in this directory. Each instructs the model to
classify a single ASL fingerspelling image and to end its reply with the exact
line ``ANSWER: <single character>``. The images are passed as attachments only;
the filename (which encodes the class) is never sent to the model.

Templates are fully static (no per-item placeholders), so ``render_prompt``
simply returns the template body; the function is kept for a stable API.
"""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent

TEMPLATES: dict[str, dict] = {
    "v1_zeroshot": {
        "file": "v1_zeroshot.md",
        "label": "v1: minimal zero-shot",
    },
    "v2_class_list": {
        "file": "v2_class_list.md",
        "label": "v2: class list + O/0 caveat",
    },
    "v3_reasoning": {
        "file": "v3_reasoning.md",
        "label": "v3: explicit reasoning steps",
    },
}


def list_templates() -> list[dict]:
    """Return template metadata for UI pickers."""
    return [{"id": tid, "label": meta["label"]} for tid, meta in TEMPLATES.items()]


def load_template(template_id: str) -> str:
    """Return the raw Markdown body of a template."""
    if template_id not in TEMPLATES:
        raise KeyError(f"Unknown template id: {template_id}")
    path = PROMPTS_DIR / TEMPLATES[template_id]["file"]
    return path.read_text(encoding="utf-8")


def render_prompt(template_id: str) -> str:
    """Render a template to the final prompt string sent to a model."""
    return load_template(template_id)
