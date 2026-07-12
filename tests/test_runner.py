import time

import pandas as pd
import pytest

from aslbench import config, runner
from aslbench.config import ProviderConfig
from aslbench.providers import CompletionResult


class FakeProvider:
    """Returns a canned ANSWER response, or raises for a chosen model."""

    def __init__(self, cfg, responses=None, raise_for=None):
        self.id = cfg.id
        self.label = cfg.label
        self._responses = responses or {}
        self._raise_for = raise_for

    def is_configured(self):
        return True

    def list_models(self):
        return []

    def complete(self, model, prompt, image_path):
        if model == self._raise_for:
            raise RuntimeError("boom")
        text = self._responses.get(model, "ANSWER: A")
        return CompletionResult(text=text, latency_s=0.01, input_tokens=10, output_tokens=5)


def _fixture_subset():
    rows = []
    for i in range(6):
        rows.append(
            {
                "item_id": f"P1_A_{i}",
                "true_char": "A",
                "participant": "P1",
                "image_abs_path": f"/tmp/fake-{i}.jpg",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def wired(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "ACTIVE_LOCK", tmp_path / "runs" / "ACTIVE")
    monkeypatch.setattr(runner.config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(runner.config, "ACTIVE_LOCK", tmp_path / "runs" / "ACTIVE")
    monkeypatch.setattr(runner, "build_subset", lambda n, seed: _fixture_subset())
    monkeypatch.setattr(runner, "providers_hash", lambda: "deadbeef")

    providers = {
        "pa": ProviderConfig(id="pa", type="fake", label="Prov A"),
        "pb": ProviderConfig(id="pb", type="fake", label="Prov B"),
    }
    monkeypatch.setattr(runner, "load_providers", lambda: list(providers.values()))
    return monkeypatch


def _wait_done(run_slug, timeout=10):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        state = runner.load_run_state(run_slug)
        if state["status"] in ("done", "error", "cancelled"):
            return state
        time.sleep(0.05)
    raise AssertionError("run did not finish in time")


def test_two_model_run(wired):
    monkeypatch = wired

    def fake_get_provider(cfg):
        return FakeProvider(cfg, responses={"model-x": "ANSWER: A", "model-y": "ANSWER: B"})

    monkeypatch.setattr(runner, "get_provider", fake_get_provider)

    slug = runner.start_run(
        model_specs=[
            {"provider_id": "pa", "model_id": "model-x"},
            {"provider_id": "pb", "model_id": "model-y"},
        ],
        template_id="v2_class_list",
        n_per_class=6,
    )
    state = _wait_done(slug)
    assert state["status"] == "done"

    cfg = runner.load_run_config(slug)
    assert cfg["n_items"] == 6
    assert len(cfg["item_ids"]) == 6

    res = runner.load_run(slug)
    slugs = [m["model_slug"] for m in cfg["models"]]
    ids_a = list(res.results[slugs[0]]["item_id"])
    ids_b = list(res.results[slugs[1]]["item_id"])
    assert ids_a == ids_b == cfg["item_ids"]

    # model-x always answers A (all true labels are A) -> accuracy 1.0.
    assert res.summaries[slugs[0]]["accuracy"] == 1.0
    assert res.summaries[slugs[1]]["accuracy"] == 0.0


def test_identical_items_across_models(wired):
    monkeypatch = wired
    monkeypatch.setattr(runner, "get_provider", lambda cfg: FakeProvider(cfg))
    slug = runner.start_run(
        model_specs=[
            {"provider_id": "pa", "model_id": "m1"},
            {"provider_id": "pa", "model_id": "m2"},
        ],
        template_id="v2_class_list",
        n_per_class=6,
    )
    _wait_done(slug)
    cfg = runner.load_run_config(slug)
    res = runner.load_run(slug)
    slugs = [m["model_slug"] for m in cfg["models"]]
    assert list(res.results[slugs[0]]["item_id"]) == list(res.results[slugs[1]]["item_id"])


def test_one_model_errors_other_completes(wired):
    monkeypatch = wired

    def fake_get_provider(cfg):
        return FakeProvider(cfg, raise_for="bad-model")

    monkeypatch.setattr(runner, "get_provider", fake_get_provider)
    slug = runner.start_run(
        model_specs=[
            {"provider_id": "pa", "model_id": "bad-model"},
            {"provider_id": "pb", "model_id": "good-model"},
        ],
        template_id="v2_class_list",
        n_per_class=6,
    )
    state = _wait_done(slug)
    cfg = runner.load_run_config(slug)
    by_model = {m["model_id"]: m["model_slug"] for m in cfg["models"]}
    assert runner.load_model_state(slug, by_model["bad-model"])["status"] == "error"
    assert runner.load_model_state(slug, by_model["good-model"])["status"] == "done"
    assert state["status"] == "error"


def test_duplicate_pair_rejected(wired):
    monkeypatch = wired
    monkeypatch.setattr(runner, "get_provider", lambda cfg: FakeProvider(cfg))
    with pytest.raises(ValueError):
        runner.start_run(
            model_specs=[
                {"provider_id": "pa", "model_id": "m1"},
                {"provider_id": "pa", "model_id": "m1"},
            ],
            template_id="v2_class_list",
            n_per_class=6,
        )


def test_cancel_run_stops_early(wired):
    monkeypatch = wired

    class CancellingProvider(FakeProvider):
        def complete(self, model, prompt, image_path):
            # Trigger cancellation after the first item completes.
            runner.cancel_run()
            return super().complete(model, prompt, image_path)

    monkeypatch.setattr(runner, "get_provider", lambda cfg: CancellingProvider(cfg))
    slug = runner.start_run(
        model_specs=[{"provider_id": "pa", "model_id": "m1"}],
        template_id="v2_class_list",
        n_per_class=6,
    )
    state = _wait_done(slug)
    assert state["status"] == "cancelled"
    cfg = runner.load_run_config(slug)
    mslug = cfg["models"][0]["model_slug"]
    mstate = runner.load_model_state(slug, mslug)
    assert mstate["status"] == "cancelled"
    # At least the first item ran, but not the full subset of 6.
    assert 1 <= mstate["completed"] < 6
    # The active lock is released once the run reaches a terminal state.
    assert not runner.is_run_active()
