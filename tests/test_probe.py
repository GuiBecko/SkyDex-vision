import numpy as np
import pytest

from app.probe import Probe, load_probe
from app.prompts import GROUP_ORDER


def a_probe() -> Probe:
    # Two groups, hand-built so the expected output is calculable by hand.
    # Deliberately not a shippable probe: `load_probe` requires all six groups,
    # but `Probe` itself is arithmetic and does not care how many there are.
    return Probe(
        groups=["CLEAR", "RAIN"],
        weights=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        bias=np.array([0.0, 0.0]),
    )


def a_full_probe(dimensions: int = 4) -> Probe:
    """A probe covering every group in GROUP_ORDER — what a real one looks like."""
    return Probe(
        groups=list(GROUP_ORDER),
        weights=np.eye(len(GROUP_ORDER), dimensions),
        bias=np.zeros(len(GROUP_ORDER)),
    )


def save_probe(path, groups, weights, bias) -> None:
    np.savez(path, groups=np.array(groups), weights=weights, bias=bias)


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


def test_probe_rejects_more_group_names_than_weight_rows():
    # Same failure mode, and now the same policy, as group_scores' length check:
    # zip() would pair the first two names with the two rows and silently drop
    # the third, serving a short, confidently wrong phenomenon_scores.
    with pytest.raises(ValueError):
        Probe(
            groups=["CLEAR", "RAIN", "SNOW"],
            weights=np.zeros((2, 4)),
            bias=np.zeros(2),
        )


def test_probe_rejects_more_weight_rows_than_group_names():
    with pytest.raises(ValueError):
        Probe(groups=["CLEAR"], weights=np.zeros((2, 4)), bias=np.zeros(2))


def test_probe_rejects_a_bias_of_the_wrong_length():
    with pytest.raises(ValueError):
        Probe(groups=["CLEAR", "RAIN"], weights=np.zeros((2, 4)), bias=np.zeros(3))


def test_load_probe_returns_none_when_the_file_is_absent(tmp_path):
    assert load_probe(tmp_path / "nothing.npz") is None


def test_load_probe_round_trips_a_saved_probe(tmp_path):
    path = tmp_path / "probe.npz"
    original = a_full_probe()
    save_probe(path, original.groups, original.weights, original.bias)

    loaded = load_probe(path)

    assert loaded is not None
    assert loaded.groups == original.groups
    np.testing.assert_allclose(loaded.weights, original.weights)


def test_load_probe_accepts_the_groups_in_any_order(tmp_path):
    # The response is keyed by name and each row of weights travels with its
    # label, so a probe whose classes_ came back in a different order is fine.
    path = tmp_path / "probe.npz"
    shuffled = list(reversed(GROUP_ORDER))
    save_probe(path, shuffled, np.eye(len(shuffled), 4), np.zeros(len(shuffled)))

    loaded = load_probe(path)

    assert loaded is not None
    assert sorted(loaded.groups) == sorted(GROUP_ORDER)


def test_load_probe_rejects_groups_outside_the_known_set(tmp_path):
    # A probe trained on labels the backend has never heard of would return
    # scores nothing downstream can read. Refuse it at load rather than serve
    # it. TORNADO stands in for such a label precisely because it is not one of
    # the six in app/prompts.py::GROUP_ORDER.
    path = tmp_path / "probe.npz"
    save_probe(path, ["CLEAR", "TORNADO"], np.zeros((2, 4)), np.zeros(2))

    with pytest.raises(ValueError, match="TORNADO"):
        load_probe(path)


def test_load_probe_rejects_a_probe_missing_a_group(tmp_path):
    # Reachable through the documented retrain: train() saves whatever classes
    # data/train happened to contain, so a pool with no hail and no lightning
    # writes a probe with no STORM. Serving it would answer the documented
    # six-key phenomenon_scores contract with five keys.
    path = tmp_path / "probe.npz"
    incomplete = [group for group in GROUP_ORDER if group != "STORM"]
    save_probe(path, incomplete, np.eye(len(incomplete), 4), np.zeros(len(incomplete)))

    with pytest.raises(ValueError, match="STORM"):
        load_probe(path)
