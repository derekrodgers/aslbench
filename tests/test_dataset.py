import pytest

from aslbench import dataset


def _make_processed(tmp_path, classes=("0", "A"), participants=("P1", "P2"), per=5):
    """Create a fake processed dataset with images per class/participant."""
    for cls in classes:
        cdir = tmp_path / cls
        cdir.mkdir(parents=True)
        for p in participants:
            for n in range(1, per + 1):
                (cdir / f"{p}_{cls}_{n}.jpg").write_bytes(b"\xff\xd8\xff")  # tiny jpeg header
    return tmp_path


def test_parse_filename():
    assert dataset.parse_filename("P1_A_5") == ("P1", "A", 5)
    assert dataset.parse_filename("P10_0_100") == ("P10", "0", 100)
    assert dataset.parse_filename("garbage") is None


def test_load_items(tmp_path):
    base = _make_processed(tmp_path)
    df = dataset.load_items(base)
    # 2 classes x 2 participants x 5 = 20
    assert len(df) == 20
    assert set(df["true_char"]) == {"0", "A"}
    assert set(df["participant"]) == {"P1", "P2"}
    assert df["item_id"].is_unique


def test_available_classes_order(tmp_path):
    base = _make_processed(tmp_path, classes=("A", "0"))
    # canonical order puts digits before letters
    assert dataset.available_classes(base) == ["0", "A"]


def test_dataset_stats(tmp_path):
    base = _make_processed(tmp_path)
    stats = dataset.dataset_stats(base)
    assert stats.n_classes == 2
    assert stats.n_participants == 2
    assert stats.n_items == 20
    assert stats.min_per_class == 10


def test_build_subset_size_and_no_duplicates(tmp_path):
    base = _make_processed(tmp_path)
    subset = dataset.build_subset(3, seed=1, base=base)
    # 3 per class x 2 classes
    assert len(subset) == 6
    assert subset["item_id"].is_unique
    counts = subset.groupby("true_char").size().to_dict()
    assert counts == {"0": 3, "A": 3}


def test_build_subset_deterministic(tmp_path):
    base = _make_processed(tmp_path)
    a = dataset.build_subset(4, seed=42, base=base)
    b = dataset.build_subset(4, seed=42, base=base)
    assert list(a["item_id"]) == list(b["item_id"])


def test_build_subset_all_classes_included(tmp_path):
    base = _make_processed(tmp_path)
    subset = dataset.build_subset(2, seed=7, base=base)
    assert set(subset["true_char"]) == {"0", "A"}


def test_load_items_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset.load_items(tmp_path / "does-not-exist")
