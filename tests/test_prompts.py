from aslbench import prompts


def test_all_templates_end_with_answer_contract():
    for tid in prompts.TEMPLATES:
        rendered = prompts.render_prompt(tid)
        last = rendered.strip().splitlines()[-1]
        assert last == "ANSWER: <single character>"


def test_v1_is_minimal():
    rendered = prompts.render_prompt("v1_zeroshot")
    assert "ANSWER: <single character>" in rendered


def test_v2_lists_classes_and_o_zero_caveat():
    rendered = prompts.render_prompt("v2_class_list")
    assert "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z" in rendered
    assert "0 1 2 3 4 5 6 7 8 9" in rendered
    assert '"0"' in rendered and '"O"' in rendered


def test_v3_has_reasoning_steps():
    rendered = prompts.render_prompt("v3_reasoning")
    assert "one hand or two hands" in rendered
    assert "ANSWER: <single character>" in rendered


def test_list_templates():
    tpls = prompts.list_templates()
    ids = {t["id"] for t in tpls}
    assert ids == {"v1_zeroshot", "v2_class_list", "v3_reasoning"}


def test_unknown_template_raises():
    import pytest

    with pytest.raises(KeyError):
        prompts.render_prompt("nope")
