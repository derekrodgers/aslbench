"""Run orchestration: background threads, atomic state files, per-model results.

A run evaluates one subset of the ASL fingerspelling dataset with one prompt
template against one or more models. Every model sees the identical sampled item
list so results are directly comparable. Each model runs in its own thread
writing only inside its own subfolder; shared state files are updated under a
lock with atomic replace.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import __version__, config
from . import prompts as prompts_mod
from . import scoring
from .config import ProviderConfig, load_providers, providers_hash
from .dataset import build_subset
from .providers import ERROR_SENTINEL, get_provider

_STATE_LOCK = threading.Lock()
_ACTIVE_HANDLE: dict | None = None
_CANCEL_EVENT: threading.Event | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _model_slug(provider_id: str, model_id: str, existing: set[str]) -> str:
    base = _slugify(f"{provider_id}-{model_id}")
    slug = base
    i = 2
    while slug in existing:
        slug = f"{base}-{i}"
        i += 1
    existing.add(slug)
    return slug


def run_dir(run_slug: str) -> Path:
    return config.RUNS_DIR / run_slug


def clear_stale_lock() -> None:
    """Remove the ACTIVE lock on app start (single-user local tool)."""
    if config.ACTIVE_LOCK.exists():
        config.ACTIVE_LOCK.unlink()


def is_run_active() -> bool:
    return config.ACTIVE_LOCK.exists()


# ---------------------------------------------------------------------------
# Run start
# ---------------------------------------------------------------------------


def start_run(
    model_specs: list[dict],
    template_id: str,
    n_per_class: int,
    run_note: str = "",
) -> str:
    """Validate inputs, create the run folder, spawn threads, return the slug.

    ``n_per_class`` images are sampled from every class (all classes are always
    included) with a freshly generated, recorded seed; every model iterates the
    identical sampled list.
    """
    global _ACTIVE_HANDLE, _CANCEL_EVENT

    if not model_specs:
        raise ValueError("model_specs must be non-empty")
    if is_run_active():
        raise RuntimeError("A run is already active")
    if n_per_class < 1:
        raise ValueError("n_per_class must be at least 1")

    sample_seed = random.randrange(2**31)
    subset = build_subset(n_per_class, seed=sample_seed)
    item_ids = subset["item_id"].tolist()
    n_items = len(item_ids)
    n_classes = int(subset["true_char"].nunique())

    providers = {p.id: p for p in load_providers()}

    # Build model records with unique slugs.
    slugs: set[str] = set()
    models: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for spec in model_specs:
        pid, mid = spec["provider_id"], spec["model_id"]
        if (pid, mid) in seen_pairs:
            raise ValueError(f"Duplicate provider+model pair: {pid}/{mid}")
        seen_pairs.add((pid, mid))
        pcfg = providers.get(pid)
        if pcfg is None:
            raise ValueError(f"Unknown provider: {pid}")
        slug = _model_slug(pid, mid, slugs)
        models.append(
            {
                "provider_id": pid,
                "provider_label": pcfg.label,
                "model_id": mid,
                "model_label": mid,
                "model_slug": slug,
            }
        )

    run_slug = f"aslbench-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    rdir = run_dir(run_slug)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "models").mkdir(exist_ok=True)

    config_obj = {
        "run_slug": run_slug,
        "template_id": template_id,
        "item_ids": item_ids,
        "sample_seed": sample_seed,
        "n_per_class": n_per_class,
        "n_items": n_items,
        "n_classes": n_classes,
        "run_note": run_note,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "package_version": __version__,
        "providers_hash": providers_hash(),
        "models": models,
    }
    _atomic_write_json(rdir / "config.json", config_obj)

    # Root state.
    root_state = {
        "status": "running",
        "started_at": config_obj["started_at"],
        "finished_at": None,
        "models": {m["model_slug"]: "running" for m in models},
    }
    _atomic_write_json(rdir / "state.json", root_state)

    # Acquire the active lock.
    config.ACTIVE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    config.ACTIVE_LOCK.write_text(run_slug, encoding="utf-8")

    cancel_event = threading.Event()
    threads = []
    for m in models:
        pcfg = providers[m["provider_id"]]
        t = threading.Thread(
            target=_run_model,
            args=(run_slug, rdir, m, pcfg, template_id, subset, cancel_event),
            daemon=True,
            name=f"run-{m['model_slug']}",
        )
        threads.append(t)

    _ACTIVE_HANDLE = {"run_slug": run_slug, "threads": threads, "n_models": len(models)}
    _CANCEL_EVENT = cancel_event
    for t in threads:
        t.start()

    return run_slug


def cancel_run() -> bool:
    """Signal the active run to stop after each model's current in-flight item.

    Returns True if a cancellation was signalled, False if no run was active.
    An in-flight provider call is allowed to finish (or time out); each model
    thread stops before starting its next item and its state becomes
    "cancelled".
    """
    if _CANCEL_EVENT is not None and not _CANCEL_EVENT.is_set():
        _CANCEL_EVENT.set()
        return True
    return False


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _update_root_state(rdir: Path, model_slug: str, status: str) -> None:
    """Atomically update one model's status in the root state file."""
    with _STATE_LOCK:
        state_path = rdir / "state.json"
        with open(state_path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        state["models"][model_slug] = status
        terminal = {"done", "error", "cancelled"}
        if all(s in terminal for s in state["models"].values()):
            statuses = set(state["models"].values())
            if statuses == {"done"}:
                state["status"] = "done"
            elif "error" in statuses:
                state["status"] = "error"
            else:
                state["status"] = "cancelled"
            state["finished_at"] = datetime.now().isoformat(timespec="seconds")
            # Release the active lock when the whole run finishes.
            if config.ACTIVE_LOCK.exists():
                config.ACTIVE_LOCK.unlink()
        _atomic_write_json(state_path, state)


def _run_model(
    run_slug: str,
    rdir: Path,
    model: dict,
    pcfg: ProviderConfig,
    template_id: str,
    subset: pd.DataFrame,
    cancel_event: threading.Event,
) -> None:
    slug = model["model_slug"]
    mdir = rdir / "models" / slug
    mdir.mkdir(parents=True, exist_ok=True)
    items_jsonl = mdir / "items.jsonl"

    total = len(subset)
    mstate = {
        "status": "running",
        "completed": 0,
        "total": total,
        "current_item_id": None,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "error": None,
    }
    _atomic_write_json(mdir / "state.json", mstate)

    prompt = prompts_mod.render_prompt(template_id)

    rows: list[dict] = []
    cancelled = False
    try:
        provider = get_provider(pcfg)
        with open(items_jsonl, "w", encoding="utf-8") as jf:
            for _, item in subset.iterrows():
                if cancel_event.is_set():
                    cancelled = True
                    break
                mstate["current_item_id"] = item["item_id"]
                _atomic_write_json(mdir / "state.json", mstate)

                true_char = str(item["true_char"])
                image_path = Path(item["image_abs_path"])

                result = provider.complete(model["model_id"], prompt, image_path)
                provider_error = result.error
                text = result.text
                predicted = scoring.parse_prediction(text)
                if provider_error is not None and text.startswith(ERROR_SENTINEL):
                    predicted = None
                sc = scoring.score_item(predicted, true_char)

                row = {
                    "item_id": item["item_id"],
                    "true_char": true_char,
                    "participant": item.get("participant", ""),
                    "raw_response": text,
                    "thinking": result.thinking or "",
                    "predicted_char": sc["predicted_char"],
                    "parse_ok": sc["parse_ok"],
                    "correct": sc["correct"],
                    "latency_s": result.latency_s,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "provider_error": provider_error,
                }
                rows.append(row)
                jf.write(json.dumps(row, default=str) + "\n")
                jf.flush()

                mstate["completed"] += 1
                _atomic_write_json(mdir / "state.json", mstate)

        df = pd.DataFrame(rows)
        df.to_parquet(mdir / "items.parquet", index=False)
        summary = scoring.compute_summary(df)
        _atomic_write_json(mdir / "summary.json", summary)

        final_status = "cancelled" if cancelled else "done"
        mstate["status"] = final_status
        mstate["current_item_id"] = None
        mstate["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _atomic_write_json(mdir / "state.json", mstate)
        _update_root_state(rdir, slug, final_status)
    except Exception as exc:  # noqa: BLE001 - isolate one model's failure
        if rows:
            try:
                pd.DataFrame(rows).to_parquet(mdir / "items.parquet", index=False)
            except Exception:
                pass
        mstate["status"] = "error"
        mstate["error"] = str(exc)
        mstate["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _atomic_write_json(mdir / "state.json", mstate)
        _update_root_state(rdir, slug, "error")


# ---------------------------------------------------------------------------
# Loading runs
# ---------------------------------------------------------------------------


def load_run_config(run_slug: str) -> dict:
    with open(run_dir(run_slug) / "config.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_run_state(run_slug: str) -> dict:
    path = run_dir(run_slug) / "state.json"
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_model_state(run_slug: str, model_slug: str) -> dict:
    path = run_dir(run_slug) / "models" / model_slug / "state.json"
    if not path.exists():
        return {"status": "pending", "completed": 0, "total": 0}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_model_results(run_slug: str, model_slug: str) -> pd.DataFrame:
    mdir = run_dir(run_slug) / "models" / model_slug
    pq = mdir / "items.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    jl = mdir / "items.jsonl"
    if jl.exists():
        records = [json.loads(line) for line in jl.read_text().splitlines() if line.strip()]
        return pd.DataFrame(records)
    return pd.DataFrame()


def load_model_summary(run_slug: str, model_slug: str) -> dict:
    path = run_dir(run_slug) / "models" / model_slug / "summary.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class LoadedRun:
    config: dict
    state: dict
    results: dict  # model_slug -> DataFrame
    summaries: dict  # model_slug -> dict


def load_run(run_slug: str) -> LoadedRun:
    cfg = load_run_config(run_slug)
    state = load_run_state(run_slug)
    results = {}
    summaries = {}
    for m in cfg["models"]:
        slug = m["model_slug"]
        results[slug] = load_model_results(run_slug, slug)
        summaries[slug] = load_model_summary(run_slug, slug)
    return LoadedRun(config=cfg, state=state, results=results, summaries=summaries)


def model_results_list(run_slug: str) -> list[scoring.ModelResult]:
    """Build scoring.ModelResult objects for comparison functions."""
    cfg = load_run_config(run_slug)
    out = []
    for m in cfg["models"]:
        df = load_model_results(run_slug, m["model_slug"])
        out.append(scoring.ModelResult(model_slug=m["model_slug"], model_label=m["model_label"], df=df))
    return out


def list_runs() -> list[dict]:
    """Summaries of all runs for the History tab, newest first."""
    if not config.RUNS_DIR.exists():
        return []
    # Slug of the currently-active run (if any).
    active_slug = ""
    if config.ACTIVE_LOCK.exists():
        try:
            active_slug = config.ACTIVE_LOCK.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    runs = []
    for child in sorted(config.RUNS_DIR.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        cfg_path = child / "config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
            state = json.loads((child / "state.json").read_text()) if (child / "state.json").exists() else {}
        except Exception:
            continue
        raw_status = state.get("status", "unknown")
        # A run stuck as 'running' but not held by the active lock was interrupted
        # (e.g. the app was killed).  Try to infer the real outcome from model states.
        if raw_status == "running" and child.name != active_slug:
            model_statuses = []
            for m in cfg.get("models", []):
                ms = load_model_state(child.name, m["model_slug"])
                model_statuses.append(ms.get("status", "unknown"))
            terminal = {"done", "error", "cancelled"}
            if model_statuses and all(s in terminal for s in model_statuses):
                statuses = set(model_statuses)
                if statuses == {"done"}:
                    raw_status = "done"
                elif "error" in statuses:
                    raw_status = "error"
                else:
                    raw_status = "cancelled"
            else:
                raw_status = "interrupted"
        per_model_acc = {}
        for m in cfg.get("models", []):
            summ = load_model_summary(child.name, m["model_slug"])
            per_model_acc[m["model_label"]] = summ.get("accuracy")
        runs.append(
            {
                "run_slug": cfg["run_slug"],
                "n_per_class": cfg.get("n_per_class"),
                "n_items": cfg["n_items"],
                "n_classes": cfg.get("n_classes"),
                "template_id": cfg["template_id"],
                "started_at": cfg["started_at"],
                "status": raw_status,
                "models": [m["model_label"] for m in cfg["models"]],
                "accuracy": per_model_acc,
            }
        )
    return runs


def delete_run(run_slug: str) -> None:
    """Permanently delete all files for a run.

    Raises RuntimeError if the run is currently active.
    """
    if config.ACTIVE_LOCK.exists():
        try:
            active = config.ACTIVE_LOCK.read_text(encoding="utf-8").strip()
        except Exception:
            active = ""
        if active == run_slug:
            raise RuntimeError(f"Cannot delete the currently-active run {run_slug!r}")
    rdir = run_dir(run_slug)
    if rdir.exists():
        shutil.rmtree(rdir)
