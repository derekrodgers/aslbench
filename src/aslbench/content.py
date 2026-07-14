"""Shared explanatory text for the dashboard and the Quarto report.

Strings are stored as Markdown (``**bold**``, ``*italic*``). Use the raw
strings directly with ``Markdown(...)`` in the report. In the dashboard call
``to_dash_children(text)`` to convert to a flat list of Dash inline components
(``html.B``, ``html.Em``, plain strings) that can be passed as ``children`` to
any Dash HTML container (``html.Small``, ``html.P``, etc.).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Canonical text (single source of truth)
# ---------------------------------------------------------------------------

METRIC_DEFINITIONS: list[str] = [
    "**Accuracy** \u2014 the share of images the model labelled correctly, from "
    "0 to 1; simply how often it is right.",
    "**Macro F1** \u2014 the model's reliability averaged across all 36 "
    "characters, giving each character equal weight and blending how often "
    "its guesses are right (precision) with how often it finds each "
    "character (recall). It rewards models that do well on every character, "
    "not just the easy ones.",
    "**Parse failure rate** \u2014 how often the model's reply could not be "
    "read as a valid answer (it did not clearly name one of the 36 characters).",
    "**Provider error rate** \u2014 how often the model's API failed to return "
    "any response at all (timeouts or errors).",
    "**Chance (1/36)** \u2014 a reference column showing what each score would "
    "be for a model that guesses at random.",
]

CONFUSION_NOTE = (
    "**How to read this:** each row is the true character and each column is "
    "what the model guessed. Cells are row-normalized, so a value is the "
    "fraction of that character's images that received the correct guess; the "
    "diagonal is the model's recall for each character. A bright diagonal "
    "means accurate recognition, while bright off-diagonal cells show "
    "characters the model routinely mixes up (e.g. reading \u2018O\u2019 as "
    "\u20180\u2019)."
)

MCNEMAR_NOTE = (
    "**What this answers:** whether two models have genuinely different "
    "accuracy, judged only on the images where they disagree. Because every "
    "model saw the exact same images, we compare them image-by-image. The test "
    "looks only at *discordant items*, that is, images where one model was "
    "right and the other was wrong. Images they both got right or both got "
    "wrong say nothing about which is better, so they are set aside. If those "
    "disagreements are far more lopsided toward one model than a coin flip "
    "would explain, the difference is judged real. In the table, **better** "
    "names the model ahead on the discordant images, **only_a_correct** and "
    "**only_b_correct** count how many each won, **p_value** is the raw test "
    "result, and **p_holm** is that p-value after a Holm-Bonferroni correction "
    "that accounts for comparing every pair of models at once. **significant** "
    "is yes when p_holm is below 0.05. Rows are sorted most-significant first."
)

# ---------------------------------------------------------------------------
# Rendering helper for Dash
# ---------------------------------------------------------------------------

_SEGMENT_RE = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*")


def to_dash_children(text: str) -> list:
    """Convert a Markdown-inline string to a list of Dash inline components.

    Supports ``**bold**`` and ``*italic*``. Returns a flat list that can be
    passed as ``children`` to any Dash HTML container. Requires Dash to be
    installed (import is deferred so the module can be imported in non-Dash
    contexts such as the Quarto report without pulling in Dash).
    """
    from dash import html  # deferred to avoid hard Dash dependency at import time

    parts: list = []
    pos = 0
    for m in _SEGMENT_RE.finditer(text):
        if m.start() > pos:
            parts.append(text[pos : m.start()])
        if m.group(1) is not None:  # **bold**
            parts.append(html.B(m.group(1)))
        else:  # *italic*
            parts.append(html.Em(m.group(2)))
        pos = m.end()
    if pos < len(text):
        parts.append(text[pos:])
    return parts
