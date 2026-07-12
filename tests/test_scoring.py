import pandas as pd

from aslbench import scoring


def _scored_row(item_id, true_char, predicted, participant="P1"):
    sc = scoring.score_item(predicted, true_char)
    return {
        "item_id": item_id,
        "true_char": true_char,
        "participant": participant,
        "predicted_char": sc["predicted_char"],
        "parse_ok": sc["parse_ok"],
        "correct": sc["correct"],
        "provider_error": None,
    }


def test_score_item_correct():
    r = scoring.score_item("A", "A")
    assert r["correct"] is True
    assert r["parse_ok"] is True
    assert r["predicted_char"] == "A"


def test_score_item_incorrect():
    r = scoring.score_item("B", "A")
    assert r["correct"] is False
    assert r["parse_ok"] is True


def test_score_item_parse_failure():
    r = scoring.score_item(None, "A")
    assert r["parse_ok"] is False
    assert r["correct"] is False
    assert r["predicted_char"] is None


def _fixture():
    # true = A,A,A,B,B ; preds = A(correct),B(wrong),None(fail),B(correct),A(wrong)
    rows = [
        _scored_row("i0", "A", "A"),
        _scored_row("i1", "A", "B"),
        _scored_row("i2", "A", None),
        _scored_row("i3", "B", "B", participant="P2"),
        _scored_row("i4", "B", "A", participant="P2"),
    ]
    return pd.DataFrame(rows)


def test_compute_summary_accuracy():
    summ = scoring.compute_summary(_fixture())
    assert summ["n_items"] == 5
    assert summ["accuracy"] == 2 / 5  # i0 and i3
    assert summ["parse_failure_rate"] == 1 / 5
    assert summ["n_classes"] == 2


def test_bootstrap_ci_determinism():
    df = _fixture()
    s1 = scoring.compute_summary(df)
    s2 = scoring.compute_summary(df)
    assert s1["accuracy_ci"] == s2["accuracy_ci"]
    lo, hi = s1["accuracy_ci"]
    assert lo <= s1["accuracy"] <= hi


def test_per_class_table():
    t = scoring.per_class_table(_fixture()).set_index("true_char")
    assert t.loc["A", "support"] == 3
    assert t.loc["A", "n_correct"] == 1
    assert t.loc["B", "support"] == 2
    assert t.loc["B", "n_correct"] == 1


def test_per_participant_table():
    t = scoring.per_participant_table(_fixture()).set_index("participant")
    assert t.loc["P1", "n"] == 3
    assert t.loc["P1", "accuracy"] == 1 / 3
    assert t.loc["P2", "accuracy"] == 1 / 2


def test_confusion_long_includes_parse_failures():
    grid = scoring.confusion_long(_fixture())
    lookup = {(r.true_char, r.pred_char): r.count for r in grid.itertuples()}
    assert lookup[("A", "A")] == 1
    assert lookup[("A", "B")] == 1
    assert lookup[("A", scoring.PARSE_FAIL_SYMBOL)] == 1


def test_most_confused_excludes_diagonal():
    mc = scoring.most_confused(_fixture())
    assert not ((mc["true_char"] == mc["pred_char"]).any())
    assert mc["count"].iloc[0] >= mc["count"].iloc[-1]


def test_macro_f1_bounds():
    summ = scoring.compute_summary(_fixture())
    assert 0.0 <= summ["macro_f1"] <= 1.0
