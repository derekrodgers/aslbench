import pandas as pd

from aslbench import scoring
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
    assert "accuracy" in list(table["metric"])


def test_outcome_matrix():
    matrix = scoring.outcome_matrix(_results()).set_index("item_id")
    a = matrix["Model A"].to_dict()
    assert a["i0"] == "correct"
    assert a["i2"] == "incorrect"
    assert a["i3"] == "parse-failure"
    assert "true_char" in matrix.columns


def test_pairwise_agreement_counts():
    agree = scoring.pairwise_agreement(_results())
    row = agree.iloc[0]
    # A correct: i0,i1 ; B correct: i0,i2,i3
    assert row["both_correct"] == 1  # i0
    assert row["only_a_correct"] == 1  # i1
    assert row["only_b_correct"] == 2  # i2, i3
    assert row["both_incorrect"] == 0


def test_hardest_easiest():
    he = scoring.hardest_easiest(_results())
    assert "i0" in he["easiest"]  # both correct


def test_hardest_classes():
    hc = scoring.hardest_classes(_results())
    assert set(hc["true_char"]) == {"A", "B", "C", "D"}
    assert hc["mean_accuracy"].iloc[0] <= hc["mean_accuracy"].iloc[-1]


def test_single_model_degrades():
    out = scoring.compare_models([_results()[0]])
    assert out["pairwise_agreement"].empty
    assert "Model A" in out["comparison_table"].columns
    assert not out["outcome_matrix"].empty
