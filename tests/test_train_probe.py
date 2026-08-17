"""Cache-invalidation tests for training/train_probe.py.

No real images or CLIP here -- these exercise only the fingerprint and cache
logic, over a tiny temporary tree of throwaway files, so they belong in the
fast suite rather than behind `-m slow`.
"""

import numpy as np

from training import train_probe
from training.train_probe import _fingerprint_tree, _load_cache


def _write_tree(root, files):
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _write_cache(cache_path, fingerprint=None, n=2):
    features = np.zeros((n, 4))
    labels = np.array(["CLEAR"] * n)
    if fingerprint is None:
        # Mimics a cache written by the pre-fingerprint code.
        np.savez(cache_path, features=features, labels=labels)
    else:
        np.savez(cache_path, features=features, labels=labels, fingerprint=fingerprint)


def test_load_cache_hits_when_the_tree_is_unchanged(tmp_path):
    source = tmp_path / "source"
    _write_tree(source, {"a/1.jpg": b"one", "a/2.jpg": b"two"})
    cache_path = tmp_path / "cache.npz"
    _write_cache(cache_path, fingerprint=_fingerprint_tree(source))

    result = _load_cache(source, cache_path)

    assert result is not None
    features, labels = result
    assert len(labels) == 2


def test_load_cache_rejects_a_cache_once_a_file_is_added(tmp_path, capsys):
    source = tmp_path / "source"
    _write_tree(source, {"a/1.jpg": b"one"})
    cache_path = tmp_path / "cache.npz"
    _write_cache(cache_path, fingerprint=_fingerprint_tree(source))

    (source / "a" / "2.jpg").write_bytes(b"two")

    result = _load_cache(source, cache_path)

    assert result is None
    assert "stale" in capsys.readouterr().out


def test_load_cache_rejects_a_cache_once_a_file_is_removed(tmp_path):
    source = tmp_path / "source"
    _write_tree(source, {"a/1.jpg": b"one", "a/2.jpg": b"two"})
    cache_path = tmp_path / "cache.npz"
    _write_cache(cache_path, fingerprint=_fingerprint_tree(source))

    (source / "a" / "2.jpg").unlink()

    assert _load_cache(source, cache_path) is None


def test_load_cache_rejects_a_cache_once_a_file_is_replaced(tmp_path):
    source = tmp_path / "source"
    _write_tree(source, {"a/1.jpg": b"one"})
    cache_path = tmp_path / "cache.npz"
    _write_cache(cache_path, fingerprint=_fingerprint_tree(source))

    # Different length guarantees the fingerprint changes regardless of the
    # filesystem's mtime resolution.
    (source / "a" / "1.jpg").write_bytes(b"a much longer replacement payload")

    assert _load_cache(source, cache_path) is None


def test_load_cache_treats_a_fingerprint_less_cache_as_a_miss_not_a_crash(tmp_path, capsys):
    source = tmp_path / "source"
    _write_tree(source, {"a/1.jpg": b"one"})
    cache_path = tmp_path / "cache.npz"
    _write_cache(cache_path, fingerprint=None)  # old-format cache, no fingerprint field

    result = _load_cache(source, cache_path)

    assert result is None
    assert "stale" in capsys.readouterr().out


def test_fingerprint_is_stable_for_an_unchanged_tree(tmp_path):
    source = tmp_path / "source"
    _write_tree(source, {"a/1.jpg": b"one", "b/2.jpg": b"two"})

    assert _fingerprint_tree(source) == _fingerprint_tree(source)


def test_fingerprint_changes_when_image_suffixes_change(tmp_path, monkeypatch):
    """IMAGE_SUFFIXES decides which files `embed_folder` treats as candidates
    at all, so it belongs in the fingerprint alongside SOURCE_TO_GROUP,
    MODEL_ARCHITECTURE and MODEL_WEIGHTS -- otherwise widening or narrowing
    it (adding `.webp`, say) changes what a cache hit stands in for while the
    tree's file listing stays byte-identical, and a stale cache would be
    reused silently."""
    source = tmp_path / "source"
    _write_tree(source, {"a/1.jpg": b"one", "a/2.png": b"two"})

    before = _fingerprint_tree(source)
    monkeypatch.setattr(train_probe, "IMAGE_SUFFIXES", {".jpg", ".jpeg", ".png", ".webp"})
    after = _fingerprint_tree(source)

    assert before != after
