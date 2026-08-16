import math

import pytest

from app.scoring import group_scores, outdoor_score, softmax


def test_softmax_sums_to_one():
    result = softmax([0.1, 0.2, 0.3], temperature=1.0)

    assert math.isclose(sum(result), 1.0, abs_tol=1e-9)


def test_softmax_ranks_the_largest_input_highest():
    result = softmax([0.1, 0.9, 0.2], temperature=1.0)

    assert result.index(max(result)) == 1


def test_softmax_is_stable_for_large_inputs():
    # Temperature 100 is CLIP's logit scale, so a similarity of 0.9 becomes 90.
    # A naive exp() on that is fine, but on a batch with a wide spread it is not:
    # this test pins the max-subtraction that keeps it from overflowing to inf.
    result = softmax([0.9, 0.1, 0.05], temperature=100.0)

    assert all(math.isfinite(value) for value in result)
    assert math.isclose(sum(result), 1.0, abs_tol=1e-9)


def test_outdoor_score_is_the_mass_on_the_sky_prompts():
    # Three sky prompts, two not-sky prompts. The sky ones dominate.
    similarities = [0.30, 0.30, 0.30, 0.05, 0.05]

    score = outdoor_score(similarities, sky_count=3)

    assert score > 0.99


def test_outdoor_score_is_low_when_the_not_sky_prompts_win():
    similarities = [0.05, 0.05, 0.05, 0.40, 0.40]

    score = outdoor_score(similarities, sky_count=3)

    assert score < 0.01


def test_group_scores_sums_to_one_across_the_groups():
    sizes = [("CLEAR", 2), ("CLOUDY", 2), ("RAIN", 2)]
    similarities = [0.1, 0.1, 0.5, 0.4, 0.2, 0.2]

    scores = group_scores(similarities, sizes)

    assert set(scores) == {"CLEAR", "CLOUDY", "RAIN"}
    assert math.isclose(sum(scores.values()), 1.0, abs_tol=1e-6)


def test_group_scores_credits_a_group_with_both_of_its_prompts():
    # CLOUDY holds the two strongest prompts, so it must win even though each
    # individual prompt is only slightly ahead.
    sizes = [("CLEAR", 2), ("CLOUDY", 2)]
    similarities = [0.20, 0.20, 0.25, 0.25]

    scores = group_scores(similarities, sizes)

    assert scores["CLOUDY"] > scores["CLEAR"]


def test_group_scores_rejects_a_length_mismatch():
    # A silent mismatch would shift every group's slice by one and produce
    # confident, wrong answers. Fail loudly instead.
    with pytest.raises(ValueError):
        group_scores([0.1, 0.2], [("CLEAR", 2), ("CLOUDY", 2)])
