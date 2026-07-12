"""Dataset loading and subset selection for aslbench.

The benchmark uses a single fixed dataset that lives under data/processed/, with
one folder per class (the 36 ASL fingerspelling characters 0-9 and A-Z) and the
original image filenames retained. A filename such as ``P1_A_5.jpg`` means
"participant 1, class A, image 5".

There is no dataset generation here; data/processed/ is produced once by
scripts/subset_dataset.py from the raw ASL-HG dataset. At run time the operator
chooses how many images per class to sample; build_subset performs that seeded
sampling.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config

# Filename pattern: P<participant>_<class>_<image number>.jpg
_FILENAME_RE = re.compile(r"^(P\d+)_([0-9A-Za-z])_(\d+)$")


def processed_dir(base: Path | None = None) -> Path:
    return base or config.PROCESSED_DIR


def parse_filename(stem: str) -> tuple[str, str, int] | None:
    """Return (participant, class_char, image_number) from a filename stem.

    Returns None when the stem does not match the expected pattern.
    """
    m = _FILENAME_RE.match(stem)
    if not m:
        return None
    participant, cls, num = m.group(1), m.group(2), int(m.group(3))
    return participant, cls, num


def available_classes(base: Path | None = None) -> list[str]:
    """Classes (in canonical order) that have a folder under processed/."""
    root = processed_dir(base)
    return [c for c in config.CLASSES if (root / c).is_dir()]


def load_items(base: Path | None = None) -> pd.DataFrame:
    """Load every image in the processed dataset as a row.

    Columns: item_id, true_char, participant, image_number, filename, rel_path,
    image_abs_path. item_id is the filename stem, which is unique across the
    dataset. The true class is the containing folder name, which is also encoded
    in the filename; the folder name is authoritative.
    """
    root = processed_dir(base)
    if not root.exists():
        raise FileNotFoundError(
            f"No processed dataset at {root}. Run scripts/subset_dataset.py first."
        )
    rows: list[dict] = []
    for cls in available_classes(base):
        cdir = root / cls
        for path in sorted(cdir.glob(f"*{config.IMAGE_SUFFIX}")):
            parsed = parse_filename(path.stem)
            participant = parsed[0] if parsed else ""
            image_number = parsed[2] if parsed else 0
            rows.append(
                {
                    "item_id": path.stem,
                    "true_char": cls,
                    "participant": participant,
                    "image_number": image_number,
                    "filename": path.name,
                    "rel_path": f"{cls}/{path.name}",
                    "image_abs_path": str(path.resolve()),
                }
            )
    if not rows:
        raise FileNotFoundError(
            f"Processed dataset at {root} contains no images. "
            "Run scripts/subset_dataset.py first."
        )
    return pd.DataFrame(rows)


@dataclass
class DatasetStats:
    n_classes: int
    n_participants: int
    n_items: int
    per_class: dict[str, int]
    min_per_class: int


def dataset_stats(base: Path | None = None) -> DatasetStats:
    """Summary counts for the processed dataset, used by the app sidebar."""
    df = load_items(base)
    per_class = df.groupby("true_char").size().to_dict()
    return DatasetStats(
        n_classes=int(df["true_char"].nunique()),
        n_participants=int(df["participant"].nunique()),
        n_items=int(len(df)),
        per_class={c: int(per_class.get(c, 0)) for c in available_classes(base)},
        min_per_class=int(min(per_class.values())) if per_class else 0,
    )


def build_subset(n_per_class: int, seed: int, base: Path | None = None) -> pd.DataFrame:
    """Sample ``n_per_class`` images from every class, without replacement.

    All classes are always included, so the returned frame has
    ``n_per_class * n_classes`` rows (fewer only if some class holds fewer than
    ``n_per_class`` images, which should not happen for the shipped dataset).
    Sampling is deterministic under ``seed``.
    """
    if n_per_class < 1:
        raise ValueError("n_per_class must be at least 1")
    df = load_items(base)
    rng = random.Random(seed)
    parts: list[pd.DataFrame] = []
    for cls in available_classes(base):
        group = df[df["true_char"] == cls]
        ids = group["item_id"].tolist()
        take = min(n_per_class, len(ids))
        chosen = rng.sample(ids, take)
        parts.append(group[group["item_id"].isin(chosen)])
    subset = pd.concat(parts, ignore_index=True)
    return subset.reset_index(drop=True)


def item_image_path(item_id: str, base: Path | None = None) -> Path | None:
    """Resolve an item id to its image path, or None if not found."""
    df = load_items(base)
    match = df[df["item_id"] == item_id]
    if match.empty:
        return None
    return Path(match.iloc[0]["image_abs_path"])
