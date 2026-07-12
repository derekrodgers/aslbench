from aslbench.scoring import parse_prediction


def test_answer_line_letter():
    assert parse_prediction("The sign is a fist.\nANSWER: A") == "A"


def test_answer_line_digit():
    assert parse_prediction("ANSWER: 5") == "5"


def test_lowercase_uppercased():
    assert parse_prediction("answer: a") == "A"


def test_last_match_wins():
    assert parse_prediction("ANSWER: B\nwait\nANSWER: C") == "C"


def test_backtick_wrapped():
    assert parse_prediction("ANSWER: `A`") == "A"


def test_bare_single_char():
    assert parse_prediction("A") == "A"
    assert parse_prediction("  7 ") == "7"


def test_missing_answer_multichar_returns_none():
    assert parse_prediction("I think it is the letter A somewhere.") is None


def test_invalid_char_returns_none():
    assert parse_prediction("ANSWER: !") is None


def test_empty_string():
    assert parse_prediction("") is None


def test_digit_zero_and_letter_o_distinct():
    assert parse_prediction("ANSWER: 0") == "0"
    assert parse_prediction("ANSWER: O") == "O"
