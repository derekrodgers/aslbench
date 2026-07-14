import pandas as pd

from aslbench import figures, scoring
from aslbench.scoring import ModelResult


def _scored_df(true_chars, preds):
    rows = []
    for i, (t, p) in enumerate(zip(true_chars, preds)):
        sc = scoring.score_item(p, t)
        rows.append(
            {
                "item_id": f"i{i}",
                "true_char": t,
                "participant": "P1",
                "predicted_char": sc["predicted_char"],
                "parse_ok": sc["parse_ok"],
                "correct": sc["correct"],
                "provider_error": None,
            }
        )
    return pd.DataFrame(rows)


def _results():
    truth = ["A", "B", "C", "D"]
    a = ModelResult("a", "Model A", _scored_df(truth, ["A", "B", "X", None]))
    b = ModelResult("b", "Model B", _scored_df(truth, ["A", "X", "C", "D"]))
    return [a, b]


def test_comparison_table_columns():
    table = scoring.comparison_table(_results())
    assert "metric" in table.columns
    assert "Model A" in table.columns and "Model B" in table.columns
    # Random-chance baseline column is always present.
    assert any(c.startswith("Chance") for c in table.columns)
    metrics = list(table["metric"])
    for m in ["accuracy", "macro_f1", "mcc"]:
        assert m in metrics


def test_comparison_table_chance_baseline():
    table = scoring.comparison_table(_results()).set_index("metric")
    chance_col = next(c for c in table.columns if c.startswith("Chance"))
    p = 1.0 / scoring.N_CLASSES
    assert table.loc["accuracy", chance_col] == p
    assert table.loc["mcc", chance_col] == 0.0


def test_outcome_matrix():
    matrix = scoring.outcome_matrix(_results()).set_index("item_id")
    a = matrix["Model A"].to_dict()
    assert a["i0"] == "correct"
    assert a["i2"] == "incorrect"
    assert a["i3"] == "parse-failure"
    assert "true_char" in matrix.columns


def test_mcnemar_table():
    mt = scoring.mcnemar_table(_results())
    row = mt.iloc[0]
    # A correct: i0,i1 ; B correct: i0,i2,i3 -> discordant: A-only {i1}, B-only {i2,i3}
    assert row["only_a_correct"] == 1
    assert row["only_b_correct"] == 2
    assert row["n_discordant"] == 3
    assert row["better"] == "Model B"
    assert 0.0 <= row["p_value"] <= 1.0


def test_mcnemar_single_model_empty():
    mt = scoring.mcnemar_table([_results()[0]])
    assert mt.empty
    assert "p_value" in mt.columns


def test_mcnemar_exact_p_symmetric():
    # No discordance -> p = 1.0; fully one-sided small counts -> small p.
    assert scoring._mcnemar_exact_p(0, 0) == 1.0
    assert scoring._mcnemar_exact_p(5, 5) == 1.0
    assert scoring._mcnemar_exact_p(10, 0) < 0.05


def test_per_class_diff_bars_two_models():
    colors = figures.assign_colors(["Model A", "Model B"])
    fig = figures.per_class_diff_bars(_results(), colors)
    assert fig is not None
    # One bar trace per pair (1 pair for 2 models).
    bar_traces = [t for t in fig.data if t.type == "bar"]
    assert len(bar_traces) == 1
    # Classes covered: A, B, C, D
    assert set(bar_traces[0].x) == {"A", "B", "C", "D"}
    # Sorted descending: A diff = 1-1=0, B diff = 1-0=1, C diff = 0-1=-1, D diff = 0-1=-1
    # B should be leftmost (highest diff = +1).
    assert bar_traces[0].x[0] == "B"


def test_per_class_diff_bars_single_model_returns_none():
    colors = figures.assign_colors(["Model A"])
    assert figures.per_class_diff_bars([_results()[0]], colors) is None
