"""Parsing, per-model metrics, bootstrap CIs, and cross-model comparison.

The task is 36-way single-character classification (ASL fingerspelling). All
metric functions are pure: they accept a DataFrame of per-item results and
return plain dicts / DataFrames. No I/O happens here.

Expected per-item columns: item_id, true_char, participant, predicted_char,
parse_ok, correct, provider_error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CLASSES

# Valid output characters, uppercased.
_VALID = set(CLASSES)

# Match an explicit "ANSWER: X" contract line. The LAST match in a response
# wins. Also allows an optional surrounding backtick or quote around the char.
_ANSWER_RE = re.compile(r"ANSWER:\s*[`'\"]?\s*([0-9A-Za-z])", re.IGNORECASE)

# Symbol used in the confusion matrix for a parse failure (no valid prediction).
PARSE_FAIL_SYMBOL = "\u2205"  # empty-set symbol

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 12345


def parse_prediction(text: str) -> str | None:
    """Extract the predicted character from a model response.

    Prefers the last ``ANSWER: X`` contract line. Falls back to a bare
    single-character response. Letters are upper-cased. Returns None when no
    valid class character can be found.
    """
    if not text:
        return None
    matches = _ANSWER_RE.findall(text)
    if matches:
        cand = matches[-1].upper()
        return cand if cand in _VALID else None
    # Fallback: the whole reply is a single character.
    stripped = text.strip().strip("`'\"").strip()
    if len(stripped) == 1:
        cand = stripped.upper()
        return cand if cand in _VALID else None
    return None


def score_item(predicted_char: str | None, true_char: str) -> dict:
    """Score a single prediction against the ground-truth class."""
    parse_ok = predicted_char is not None
    return {
        "parse_ok": parse_ok,
        "predicted_char": predicted_char,
        "correct": bool(parse_ok and predicted_char == true_char),
    }


def _safe_nanmean(series: pd.Series) -> float:
    """Mean ignoring NaNs; returns NaN for an all-NaN input without warning."""
    arr = series.to_numpy(dtype=float)
    mask = ~np.isnan(arr)
    return float(arr[mask].mean()) if mask.any() else float("nan")


def _bootstrap_ci(
    values: np.ndarray,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean of a 0/1 array."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, n)
        means[i] = values[idx].mean()
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return (lo, hi)


def _per_class_prf(df: pd.DataFrame) -> pd.DataFrame:
    """Precision, recall, F1, accuracy, and support per true class.

    Parse failures count as an incorrect prediction (a false negative for the
    true class, but not a false positive for any class).
    """
    classes = sorted(df["true_char"].unique())
    rows = []
    for c in classes:
        support = int((df["true_char"] == c).sum())
        tp = int(((df["true_char"] == c) & (df["predicted_char"] == c)).sum())
        fp = int(((df["true_char"] != c) & (df["predicted_char"] == c)).sum())
        fn = support - tp
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        if precision == precision and recall == recall and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0 if support else float("nan")
        rows.append(
            {
                "true_char": c,
                "support": support,
                "n_correct": tp,
                "accuracy": tp / support if support else float("nan"),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def compute_summary(df: pd.DataFrame) -> dict:
    """Compute all per-model metrics for a results DataFrame."""
    n = len(df)
    if n == 0:
        return {"n_items": 0}

    correct = df["correct"].to_numpy(dtype=float)
    acc = float(np.mean(correct))
    acc_lo, acc_hi = _bootstrap_ci(correct)

    parse_failures = int((~df["parse_ok"]).sum())
    provider_errors = (
        int(df["provider_error"].fillna("").astype(bool).sum()) if "provider_error" in df else 0
    )

    prf = _per_class_prf(df)
    macro_precision = _safe_nanmean(prf["precision"]) if not prf.empty else float("nan")
    macro_recall = _safe_nanmean(prf["recall"]) if not prf.empty else float("nan")
    macro_f1 = _safe_nanmean(prf["f1"]) if not prf.empty else float("nan")

    return {
        "n_items": n,
        "n_parsed": int(df["parse_ok"].sum()),
        "n_classes": int(df["true_char"].nunique()),
        "accuracy": acc,
        "accuracy_ci": [acc_lo, acc_hi],
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "parse_failure_rate": parse_failures / n,
        "provider_error_rate": provider_errors / n,
    }


def per_class_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class accuracy, precision, recall, F1, and support."""
    if df.empty:
        return pd.DataFrame(
            columns=["true_char", "support", "n_correct", "accuracy", "precision", "recall", "f1"]
        )
    return _per_class_prf(df)


def per_participant_table(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy stratified by participant (analogous to a per-subject table)."""
    if "participant" not in df.columns or df.empty:
        return pd.DataFrame(columns=["participant", "n", "accuracy"])
    rows = []
    for participant, group in df.groupby("participant"):
        rows.append(
            {
                "participant": participant,
                "n": len(group),
                "accuracy": float(group["correct"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("participant").reset_index(drop=True)


def confusion_long(df: pd.DataFrame) -> pd.DataFrame:
    """Confusion counts as long-form rows: true_char, pred_char, count.

    Predictions that failed to parse are bucketed under PARSE_FAIL_SYMBOL.
    """
    if df.empty:
        return pd.DataFrame(columns=["true_char", "pred_char", "count"])
    tmp = df.copy()
    tmp["pred_char"] = tmp["predicted_char"].where(tmp["parse_ok"], PARSE_FAIL_SYMBOL)
    grid = tmp.groupby(["true_char", "pred_char"], observed=True).size().reset_index(name="count")
    return grid


def most_confused(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Top off-diagonal (true != predicted) confusion pairs, most frequent first."""
    grid = confusion_long(df)
    if grid.empty:
        return grid
    off = grid[grid["true_char"] != grid["pred_char"]]
    return off.sort_values("count", ascending=False).head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    """A model's identity plus its scored per-item DataFrame."""

    model_slug: str
    model_label: str
    df: pd.DataFrame


_COMPARISON_METRICS = [
    "accuracy",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "parse_failure_rate",
    "provider_error_rate",
]


def comparison_table(results: list[ModelResult]) -> pd.DataFrame:
    """Metrics as rows, one column per model."""
    data: dict[str, list] = {"metric": _COMPARISON_METRICS}
    for res in results:
        summ = compute_summary(res.df)
        data[res.model_label] = [summ.get(m) for m in _COMPARISON_METRICS]
    return pd.DataFrame(data)


def outcome_matrix(results: list[ModelResult]) -> pd.DataFrame:
    """One row per item; columns are true_char plus each model's outcome.

    Outcomes are "correct", "incorrect", or "parse-failure".
    """
    if not results:
        return pd.DataFrame()
    base = results[0].df[["item_id", "true_char"]].copy().set_index("item_id")
    for res in results:
        col = []
        for _, row in res.df.iterrows():
            if not row["parse_ok"]:
                col.append("parse-failure")
            elif row["correct"]:
                col.append("correct")
            else:
                col.append("incorrect")
        base[res.model_label] = pd.Series(col, index=res.df["item_id"])
    return base.reset_index()


def pairwise_agreement(results: list[ModelResult]) -> pd.DataFrame:
    """McNemar-style discordance counts for each model pair."""
    rows = []
    correctness = {}
    for res in results:
        correctness[res.model_label] = res.df.set_index("item_id")["correct"].astype(bool)
    labels = [r.model_label for r in results]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a = correctness[labels[i]]
            b = correctness[labels[j]]
            common = a.index.intersection(b.index)
            a2, b2 = a.loc[common], b.loc[common]
            rows.append(
                {
                    "model_a": labels[i],
                    "model_b": labels[j],
                    "both_correct": int((a2 & b2).sum()),
                    "both_incorrect": int((~a2 & ~b2).sum()),
                    "only_a_correct": int((a2 & ~b2).sum()),
                    "only_b_correct": int((~a2 & b2).sum()),
                }
            )
    return pd.DataFrame(rows)


def hardest_easiest(results: list[ModelResult]) -> dict:
    """Item ids all models missed, and item ids all models got."""
    matrix = outcome_matrix(results)
    if matrix.empty:
        return {"hardest": [], "easiest": []}
    model_cols = [r.model_label for r in results]
    correct_counts = (matrix[model_cols] == "correct").sum(axis=1)
    n_models = len(model_cols)
    easiest = matrix.loc[correct_counts == n_models, "item_id"].tolist()
    hardest = matrix.loc[correct_counts == 0, "item_id"].tolist()
    return {"hardest": hardest, "easiest": easiest}


def hardest_classes(results: list[ModelResult]) -> pd.DataFrame:
    """Mean per-class accuracy averaged across models, worst first."""
    if not results:
        return pd.DataFrame(columns=["true_char", "mean_accuracy"])
    per = []
    for res in results:
        t = per_class_table(res.df)[["true_char", "accuracy"]]
        per.append(t.set_index("true_char")["accuracy"])
    combined = pd.concat(per, axis=1)
    mean_acc = combined.mean(axis=1)
    out = mean_acc.reset_index()
    out.columns = ["true_char", "mean_accuracy"]
    return out.sort_values("mean_accuracy").reset_index(drop=True)


def compare_models(results: list[ModelResult]) -> dict:
    """Bundle all cross-model comparison artifacts."""
    return {
        "comparison_table": comparison_table(results),
        "outcome_matrix": outcome_matrix(results),
        "pairwise_agreement": pairwise_agreement(results) if len(results) >= 2 else pd.DataFrame(),
        "hardest_easiest": hardest_easiest(results),
        "hardest_classes": hardest_classes(results),
    }
