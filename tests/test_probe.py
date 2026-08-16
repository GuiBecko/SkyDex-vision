import numpy as np
import pytest

from app.probe import Probe, load_probe
from app.prompts import GROUP_ORDER


def a_probe(dimensions: int = 4) -> Probe:
    # Two groups, hand-built so the expected output is calculable by hand.
    return Probe(
        groups=["CLEAR", "RAIN"],
        weights=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        bias=np.array([0.0, 0.0]),
    )


def test_apply_returns_one_score_per_group():
    scores = a_probe().apply(np.array([1.0, 0.0, 0.0, 0.0]))

    assert set(scores) == {"CLEAR", "RAIN"}


def test_apply_sums_to_one():
    scores = a_probe().apply(np.array([0.3, 0.7, 0.1, 0.2]))

    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-6)


def test_apply_favours_the_group_whose_weights_match_the_embedding():
    scores = a_probe().apply(np.array([5.0, 0.0, 0.0, 0.0]))

    assert scores["CLEAR"] > scores["RAIN"]


def test_apply_rejects_an_embedding_of_the_wrong_width():
    with pytest.raises(ValueError):
        a_probe().apply(np.array([1.0, 2.0]))


def test_load_probe_returns_none_when_the_file_is_absent(tmp_path):
    assert load_probe(tmp_path / "nothing.npz") is None


def test_load_probe_round_trips_a_saved_probe(tmp_path):
    path = tmp_path / "probe.npz"
    original = a_probe()
    np.savez(
        path,
        groups=np.array(original.groups),
        weights=original.weights,
        bias=original.bias,
    )

    loaded = load_probe(path)

    assert loaded is not None
    assert loaded.groups == original.groups
    np.testing.assert_allclose(loaded.weights, original.weights)


def test_load_probe_rejects_groups_outside_the_known_set(tmp_path):
    # A probe trained on labels the backend has never heard of would return
    # scores nothing downstream can read. Refuse it at load rather than serve it.
    path = tmp_path / "probe.npz"
    np.savez(
        path,
        groups=np.array(["CLEAR", "TORNADO"]),
        weights=np.zeros((2, 4)),
        bias=np.zeros(2),
    )

    with pytest.raises(ValueError, match="TORNADO"):
        load_probe(path)


def test_group_order_is_what_a_probe_may_use():
    assert "TORNADO" not in GROUP_ORDER
