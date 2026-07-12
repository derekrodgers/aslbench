"""Subset the raw ASL-HG dataset into data/processed/.

The raw dataset holds 36 class folders (0-9, A-Z). Each class contains 100
images per participant from 10 participants (P1-P10), so 1000 images per class.
A filename such as ``P1_A_5.jpg`` means "participant 1, class A, image 5".

This script keeps, for every class and every participant, a random subset of
``--per-participant`` images (default 10), preserving the original filenames. The
result is data/processed/<class>/ folders that mirror the raw layout but are far
smaller (10 participants x 10 images x 36 classes = 3600 images by default).

Sampling is seeded and the seed is recorded, so the subset is reproducible. The
raw dataset is not tracked in git (it is large); download it from
https://data.mendeley.com/datasets/j4y5w2c8w9/1 and place it at
data/raw/asl_hg_dataset/ to rebuild.

Usage:
    python scripts/subset_dataset.py                    # default: 10 per participant
    python scripts/subset_dataset.py --per-participant 5 --seed 42
    python scripts/subset_dataset.py --force            # overwrite an existing subset
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from aslbench import config

_FILENAME_RE = re.compile(r"^(P\d+)_([0-9A-Za-z])_(\d+)$")


def _group_by_participant(class_dir: Path) -> dict[str, list[Path]]:
    """Map participant id -> list of image paths for one class folder."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in class_dir.glob(f"*{config.IMAGE_SUFFIX}"):
        m = _FILENAME_RE.match(path.stem)
        if not m:
            continue
        groups[m.group(1)].append(path)
    return groups


def subset_dataset(
    per_participant: int = 10,
    seed: int | None = None,
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
    force: bool = False,
) -> dict:
    """Build data/processed/ from the raw dataset. Returns a summary dict."""
    raw_dir = raw_dir or config.RAW_DIR
    out_dir = out_dir or config.PROCESSED_DIR
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_dir}. Download it from "
            "https://data.mendeley.com/datasets/j4y5w2c8w9/1 and place it there."
        )
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise FileExistsError(
            f"{out_dir} already exists and is not empty. Pass --force to overwrite."
        )

    if seed is None:
        seed = int(time.time())
    rng = random.Random(seed)

    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = [c for c in config.CLASSES if (raw_dir / c).is_dir()]
    total_copied = 0
    per_class_counts: dict[str, int] = {}

    for cls in classes:
        src_dir = raw_dir / cls
        dst_dir = out_dir / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        groups = _group_by_participant(src_dir)
        copied = 0
        for participant in sorted(groups):
            images = sorted(groups[participant])
            take = min(per_participant, len(images))
            chosen = rng.sample(images, take)
            for path in chosen:
                shutil.copy2(path, dst_dir / path.name)
                copied += 1
        per_class_counts[cls] = copied
        total_copied += copied
        print(f"  class {cls}: {len(groups)} participants -> {copied} images", flush=True)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "per_participant": per_participant,
        "raw_dir": str(raw_dir),
        "n_classes": len(classes),
        "n_participants": max((len(_group_by_participant(raw_dir / c)) for c in classes), default=0),
        "n_images": total_copied,
        "per_class_counts": per_class_counts,
    }
    (out_dir / "subset_info.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"\nDone. {total_copied} images across {len(classes)} classes written to {out_dir} "
        f"(seed {seed}).",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Subset the raw ASL-HG dataset into data/processed/")
    parser.add_argument("--per-participant", type=int, default=10,
                        help="images to keep per participant per class (default 10)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed; when omitted, derived from the clock and recorded")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing data/processed/ folder")
    args = parser.parse_args()
    subset_dataset(per_participant=args.per_participant, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
