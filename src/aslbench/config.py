"""Paths, constants, and providers.yaml loading for aslbench.

All filesystem locations are anchored on the repository root so the package
behaves identically whether it is imported from the app, a script, the tests,
or the Quarto report.
"""

from __future__ import annotations

import hashlib
import os
import string
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Repository root is three levels up from this file:
# src/aslbench/config.py -> src/aslbench -> src -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "asl_hg_dataset"
PROCESSED_DIR = DATA_DIR / "processed"
RUNS_DIR = REPO_ROOT / "runs"
EXPORTS_DIR = REPO_ROOT / "exports"
REPORT_QMD = REPO_ROOT / "report" / "report.qmd"
PROVIDERS_YAML = REPO_ROOT / "providers.yaml"

ACTIVE_LOCK = RUNS_DIR / "ACTIVE"

# The 36 ASL fingerspelling classes, in a stable display order: digits 0-9 then
# letters A-Z. The folder names under data/processed/ are exactly these.
CLASSES: list[str] = list(string.digits) + list(string.ascii_uppercase)
N_CLASSES = len(CLASSES)

# Image files in the dataset are JPEGs.
IMAGE_SUFFIX = ".jpg"
IMAGE_MEDIA_TYPE = "image/jpeg"


def ensure_dirs() -> None:
    """Create the top-level data/runs/exports folders if missing."""
    for path in (DATA_DIR, PROCESSED_DIR, RUNS_DIR, EXPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


@dataclass
class ProviderConfig:
    id: str
    type: str
    label: str
    base_url: str | None = None
    api_key_env: str | None = None
    extra: dict = field(default_factory=dict)

    def resolve_api_key(self) -> str | None:
        """Return the credential for this provider, or None when unresolved.

        Local OpenAI-compatible servers (api_key_env is null) get a placeholder
        key since they accept any value.
        """
        if self.api_key_env is None:
            return "not-needed"
        value = os.environ.get(self.api_key_env, "")
        return value if value else None

    def credential_present(self) -> bool:
        """Whether a credential is available without probing the provider.

        For copilot_sdk this returns True as a static heuristic; the provider
        performs the real CLI auth probe in is_configured().
        """
        if self.type == "copilot_sdk":
            return True
        if self.api_key_env is None:
            return True
        return bool(os.environ.get(self.api_key_env, ""))


def load_providers(path: Path | None = None) -> list[ProviderConfig]:
    """Parse providers.yaml into ProviderConfig objects."""
    path = path or PROVIDERS_YAML
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    providers: list[ProviderConfig] = []
    for entry in raw.get("providers", []):
        known = {"id", "type", "label", "base_url", "api_key_env"}
        extra = {k: v for k, v in entry.items() if k not in known}
        providers.append(
            ProviderConfig(
                id=entry["id"],
                type=entry["type"],
                label=entry.get("label", entry["id"]),
                base_url=entry.get("base_url"),
                api_key_env=entry.get("api_key_env"),
                extra=extra,
            )
        )
    return providers


def providers_hash(path: Path | None = None) -> str:
    """Stable hash of providers.yaml bytes, recorded in run config."""
    path = path or PROVIDERS_YAML
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]
