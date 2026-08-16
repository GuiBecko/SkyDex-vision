"""Turning CLIP cosine similarities into the two score sets the API returns.

Nothing here imports torch. That is the point: these are the only numbers the
SkyDex backend ever sees, so they must be testable in milliseconds against
hand-written inputs rather than only against a 350MB model.
"""

import math
from collections.abc import Sequence

# CLIP's trained logit scale. Cosine similarities live in a narrow band around
# 0.2-0.3, so a softmax over the raw values is nearly uniform and says nothing.
# Multiplying by 100 is what the model was trained with and what makes the
# distribution informative.
DEFAULT_TEMPERATURE = 100.0


def softmax(values: Sequence[float], temperature: float = DEFAULT_TEMPERATURE) -> list[float]:
    """A numerically stable softmax over ``values`` scaled by ``temperature``.

    The max is subtracted before exponentiating. Without it, a similarity of
    0.9 at temperature 100 becomes exp(90), which is finite but close enough to
    the ceiling that a slightly wider spread overflows to inf and poisons the
    whole vector with nan.
    """
    if not values:
        raise ValueError("softmax needs at least one value")

    scaled = [value * temperature for value in values]
    ceiling = max(scaled)
    exponentiated = [math.exp(value - ceiling) for value in scaled]
    total = sum(exponentiated)
    return [value / total for value in exponentiated]


def outdoor_score(similarities: Sequence[float], sky_count: int) -> float:
    """The probability mass sitting on the sky prompts.

    ``similarities`` must be ordered sky-prompts-first, which is what
    ``prompts.OUTDOOR_PROMPTS`` guarantees and ``test_prompts.py`` pins.
    """
    if sky_count <= 0 or sky_count >= len(similarities):
        raise ValueError(
            f"sky_count must split the vector, got {sky_count} for {len(similarities)} values"
        )

    probabilities = softmax(similarities)
    return sum(probabilities[:sky_count])


def group_scores(
    similarities: Sequence[float],
    group_sizes: Sequence[tuple[str, int]],
) -> dict[str, float]:
    """Collapse per-prompt probabilities into one probability per visual group.

    A group owns several prompts, and a photograph that matches any of them
    matches the group — so the group's score is the sum of its prompts' shares,
    not the best of them. Summing is what lets a group win on two decent prompts
    against a rival group's one strong prompt, which is the behaviour we want:
    the prompts within a group are alternative phrasings of the same claim.
    """
    expected = sum(count for _, count in group_sizes)
    if expected != len(similarities):
        raise ValueError(
            f"expected {expected} similarities for these groups, got {len(similarities)}"
        )

    probabilities = softmax(similarities)

    scores: dict[str, float] = {}
    offset = 0
    for name, count in group_sizes:
        scores[name] = sum(probabilities[offset : offset + count])
        offset += count
    return scores
