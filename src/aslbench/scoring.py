"""Parsing, per-model metrics, bootstrap CIs, and cross-model comparison.

The task is 36-way single-character classification (ASL fingerspelling). All
metric functions are pure: they accept a DataFrame of per-item results and
return plain dicts / DataFrames. No I/O happens here.

Expected per-item columns: item_id, true_char, participant, predicted_char,
parse_ok, correct, provider_error.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from .config import CLASSES, N_CLASSES

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


def _y_true_pred(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned (y_true, y_pred) arrays for sklearn metrics.

    Parse failures are mapped to PARSE_FAIL_SYMBOL so they count as a wrong
    prediction that is never a false positive for any real class.
    """
    y_true = df["true_char"].to_numpy()
    y_pred = df["predicted_char"].where(df["parse_ok"], PARSE_FAIL_SYMBOL).to_numpy()
    return y_true, y_pred


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


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


def _bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% percentile bootstrap CI for an arbitrary (y_true, y_pred) metric."""
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = np.empty(resamples)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(resamples):
            idx = rng.integers(0, n, n)
            vals[i] = metric(y_true[idx], y_pred[idx])
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def _per_class_prf(df: pd.DataFrame) -> pd.DataFrame:
    """Precision, recall, F1, accuracy, support, and Wilson recall CI per class.

    Computed with scikit-learn. Parse failures count as an incorrect prediction
    (a false negative for the true class, never a false positive for any class).
    Per-class accuracy equals recall in single-label classification.
    """
    classes = sorted(df["true_char"].unique())
    y_true, y_pred = _y_true_pred(df)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    rows = []
    for i, c in enumerate(classes):
        support_c = int(support[i])
        tp = int(((y_true == c) & (y_pred == c)).sum())
        lo, hi = wilson_interval(tp, support_c)
        rows.append(
            {
                "true_char": c,
                "support": support_c,
                "n_correct": tp,
                "accuracy": float(recall[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "recall_lo": lo,
                "recall_hi": hi,
                "f1": float(f1[i]),
            }
        )
    return pd.DataFrame(rows)


def compute_summary(df: pd.DataFrame) -> dict:
    """Compute all per-model metrics for a results DataFrame (via scikit-learn)."""
    n = len(df)
    if n == 0:
        return {"n_items": 0}

    y_true, y_pred = _y_true_pred(df)
    classes = sorted(df["true_char"].unique())
    correct = df["correct"].to_numpy(dtype=float)

    acc = float(accuracy_score(y_true, y_pred))
    acc_lo, acc_hi = _bootstrap_ci(correct)

    parse_failures = int((~df["parse_ok"]).sum())
    provider_errors = (
        int(df["provider_error"].fillna("").astype(bool).sum()) if "provider_error" in df else 0
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        )
        balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
        mcc = float(matthews_corrcoef(y_true, y_pred))
        kappa = float(cohen_kappa_score(y_true, y_pred))

    def _macro_f1(yt, yp):
        return f1_score(yt, yp, labels=classes, average="macro", zero_division=0)

    macro_f1_ci = _bootstrap_metric_ci(y_true, y_pred, _macro_f1)
    mcc_ci = _bootstrap_metric_ci(y_true, y_pred, matthews_corrcoef)

    return {
        "n_items": n,
        "n_parsed": int(df["parse_ok"].sum()),
        "n_classes": int(df["true_char"].nunique()),
        "accuracy": acc,
        "accuracy_ci": [acc_lo, acc_hi],
        "balanced_accuracy": balanced_acc,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "macro_f1_ci": [macro_f1_ci[0], macro_f1_ci[1]],
        "mcc": mcc,
        "mcc_ci": [mcc_ci[0], mcc_ci[1]],
        "cohen_kappa": kappa,
        "parse_failure_rate": parse_failures / n,
        "provider_error_rate": provider_errors / n,
    }


def per_class_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class accuracy, precision, recall (+Wilson CI), F1, and support."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "true_char", "support", "n_correct", "accuracy",
                "precision", "recall", "recall_lo", "recall_hi", "f1",
            ]
        )
    return _per_class_prf(df)


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
    "balanced_accuracy",
    "macro_f1",
    "mcc",
    "cohen_kappa",
    "macro_precision",
    "macro_recall",
    "parse_failure_rate",
    "provider_error_rate",
]

# Expected metric values for a uniform-random classifier over N_CLASSES balanced
# classes: accuracy, balanced accuracy, and (approx) macro P/R/F1 all equal 1/C,
# while MCC and Cohen's kappa are 0 (no better than chance). Shown as a baseline
# column so absolute scores are interpretable against random guessing.
_CHANCE_LABEL = f"Chance (1/{N_CLASSES})"


def _chance_baseline() -> dict:
    p = 1.0 / N_CLASSES
    return {
        "accuracy": p,
        "balanced_accuracy": p,
        "macro_f1": p,
        "mcc": 0.0,
        "cohen_kappa": 0.0,
        "macro_precision": p,
        "macro_recall": p,
        "parse_failure_rate": 0.0,
        "provider_error_rate": 0.0,
    }


def comparison_table(results: list[ModelResult]) -> pd.DataFrame:
    """Metrics as rows, one column per model, plus a random-chance baseline."""
    data: dict[str, list] = {"metric": _COMPARISON_METRICS}
    for res in results:
        summ = compute_summary(res.df)
        data[res.model_label] = [summ.get(m) for m in _COMPARISON_METRICS]
    baseline = _chance_baseline()
    data[_CHANCE_LABEL] = [baseline[m] for m in _COMPARISON_METRICS]
    return pd.DataFrame(data)


def outcome_matrix(results: list[ModelResult]) -> pd.DataFrame:
    """One row per item; columns are true_char plus each model's outcome.

    Outcomes are "correct", "incorrect", "parse-failure", or "error".
    """
    if not results:
        return pd.DataFrame()
    base = results[0].df[["item_id", "true_char"]].copy().set_index("item_id")
    for res in results:
        col = []
        for _, row in res.df.iterrows():
            if row.get("provider_error"):
                col.append("error")
            elif not row["parse_ok"]:
                col.append("parse-failure")
            elif row["correct"]:
                col.append("correct")
            else:
                col.append("incorrect")
        base[res.model_label] = pd.Series(col, index=res.df["item_id"])
    return base.reset_index()


def _mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b and c.

    Under H0 (equal error rates), each discordant item is a fair coin flip, so
    the smaller count follows Binomial(b + c, 0.5). No SciPy dependency: the
    binomial tail is summed directly with math.comb.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar_table(results: list[ModelResult]) -> pd.DataFrame:
    """Exact two-sided McNemar test for every model pair on shared items.

    All models see the identical images, so correctness is paired per item. For
    each pair we report the discordant counts (only-A-correct, only-B-correct),
    the exact p-value, and which model did better on discordant items.
    """
    cols = [
        "model_a", "model_b", "only_a_correct", "only_b_correct",
        "n_discordant", "p_value", "better",
    ]
    if len(results) < 2:
        return pd.DataFrame(columns=cols)
    correctness = {
        r.model_label: r.df.set_index("item_id")["correct"].astype(bool) for r in results
    }
    labels = [r.model_label for r in results]
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a = correctness[labels[i]]
            b_series = correctness[labels[j]]
            common = a.index.intersection(b_series.index)
            a2, b2 = a.loc[common], b_series.loc[common]
            b = int((a2 & ~b2).sum())  # only model_a correct
            c = int((~a2 & b2).sum())  # only model_b correct
            p = _mcnemar_exact_p(b, c)
            if b > c:
                better = labels[i]
            elif c > b:
                better = labels[j]
            else:
                better = "tie"
            rows.append(
                {
                    "model_a": labels[i],
                    "model_b": labels[j],
                    "only_a_correct": b,
                    "only_b_correct": c,
                    "n_discordant": b + c,
                    "p_value": p,
                    "better": better,
                }
            )
    return pd.DataFrame(rows, columns=cols)
