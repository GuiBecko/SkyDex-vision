import math

import pytest

from app.scoring import group_scores, outdoor_score, softmax


def test_softmax_sums_to_one():
    result = softmax([0.1, 0.2, 0.3], temperature=1.0)

    assert math.isclose(sum(result), 1.0, abs_tol=1e-9)


def test_softmax_ranks_the_largest_input_highest():
    result = softmax([0.1, 0.9, 0.2], temperature=1.0)

    assert result.index(max(result)) == 1


def test_softmax_is_well_formed_at_clips_logit_scale():
    # Renamed from ..._is_stable_for_large_inputs, which it never was: at
    # temperature 100 a similarity of 0.9 becomes exp(90) = 1.2e39, and a naive
    # un-stabilised softmax returns exactly the same finite, summing-to-one
    # vector for this input. Cosine similarity is bounded in [-1, 1], so no
    # input reachable from this module can overflow. What this test actually
    # pins is that the real CLIP logit scale produces a usable distribution.
    # The max-subtraction is pinned by the test below.
    result = softmax([0.9, 0.1, 0.05], temperature=100.0)

    assert all(math.isfinite(value) for value in result)
    assert math.isclose(sum(result), 1.0, abs_tol=1e-9)


def test_softmax_survives_inputs_a_naive_exp_cannot():
    # Outside the cosine domain, and therefore outside anything this module's
    # own callers pass — but `temperature` is public and `softmax` is public.
    # exp(800) overflows float64, so an implementation without the max
    # subtraction raises OverflowError here instead of answering.
    result = softmax([8.0, 0.0], temperature=100.0)

    assert all(math.isfinite(value) for value in result)
    assert math.isclose(sum(result), 1.0, abs_tol=1e-9)
    assert result[0] > result[1]


def test_softmax_rejects_an_empty_vector():
    # Returning [] would push a division by zero into the caller's arithmetic
    # several frames away from the mistake.
    with pytest.raises(ValueError):
        softmax([])


def test_outdoor_score_is_the_mass_on_the_sky_prompts():
    # Three sky prompts, two not-sky prompts. The sky ones dominate.
    similarities = [0.30, 0.30, 0.30, 0.05, 0.05]

    score = outdoor_score(similarities, sky_count=3)

    assert score > 0.99


def test_outdoor_score_is_low_when_the_not_sky_prompts_win():
    similarities = [0.05, 0.05, 0.05, 0.40, 0.40]

    score = outdoor_score(similarities, sky_count=3)

    assert score < 0.01


def test_outdoor_score_rejects_a_sky_count_that_does_not_split_the_vector():
    # sky_count is the boundary between the sky prompts and the fraud
    # catalogue. At either end the split is not a split: 0 would score the mass
    # on no prompts at all, and len() would score it on all of them and always
    # return 1.0 — a confident answer that measures nothing.
    with pytest.raises(ValueError):
        outdoor_score([0.3, 0.3, 0.1], sky_count=0)

    with pytest.raises(ValueError):
        outdoor_score([0.3, 0.3, 0.1], sky_count=3)


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
