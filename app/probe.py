"""The trained half of the phenomenon head.

A probe is a multinomial logistic regression over frozen CLIP embeddings —
512 inputs, one output per visual group. It trains in seconds on CPU and
typically beats zero-shot prompting by 10-15 accuracy points, which is why it
exists; it is not a fine-tune of CLIP itself, which this machine has no GPU for.

The artefact is a ~30KB .npz. When one is present the service uses it; when it
is absent the service falls back to zero-shot and says so in its model name.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.prompts import GROUP_ORDER


@dataclass
class Probe:
    groups: list[str]
    weights: np.ndarray  # shape (len(groups), embedding_dim)
    bias: np.ndarray  # shape (len(groups),)

    def __post_init__(self) -> None:
        """Refuse a probe whose labels and arrays disagree on how many groups there are.

        Same policy, and the same reason, as ``scoring.group_scores``: a silent
        mismatch pairs each group name with whatever row happens to sit at its
        index and produces confident, wrong answers. Fail loudly instead.

        Checked at construction rather than inside ``apply`` because it is a
        property of the artefact, not of the embedding handed to it — so a
        mismatched probe cannot be built at all, and ``load_probe`` inherits
        the guard for free.
        """
        if self.weights.shape[0] != len(self.groups):
            raise ValueError(
                f"expected {self.weights.shape[0]} group names for these weights, "
                f"got {len(self.groups)}"
            )
        if self.bias.shape[0] != len(self.groups):
            raise ValueError(
                f"expected {len(self.groups)} bias terms for these groups, "
                f"got {self.bias.shape[0]}"
            )

    def apply(self, embedding: np.ndarray) -> dict[str, float]:
        """Group probabilities for one normalised CLIP embedding."""
        flat = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if flat.shape[0] != self.weights.shape[1]:
            raise ValueError(
                f"probe expects {self.weights.shape[1]}-dimensional embeddings, "
                f"got {flat.shape[0]}"
            )

        logits = self.weights @ flat + self.bias
        # Max subtraction, as in scoring.softmax — but load-bearing here rather
        # than defensive. These logits are `weights @ embedding + bias` with no
        # bound of any kind, so a wide enough spread really does overflow to inf
        # and poison the vector with nan; scoring.softmax's inputs are cosine
        # similarities and cannot.
        exponentiated = np.exp(logits - logits.max())
        probabilities = exponentiated / exponentiated.sum()
        # strict=True is belt and braces over the __post_init__ check above: this
        # is the line that would silently truncate if either guard were removed.
        return {
            group: float(value)
            for group, value in zip(self.groups, probabilities, strict=True)
        }


def load_probe(path: Path) -> Probe | None:
    """Read a probe from ``path``, or ``None`` if there is nothing there.

    An unreadable, wrongly-labelled or incomplete probe raises rather than
    being skipped. Silently falling back to zero-shot would mean a deploy that
    thought it shipped a trained model and did not, with no signal anywhere.

    "Incomplete" is not a hypothetical: `train()` saves whatever classes the
    training pool happened to contain, so a `data/train` missing every source
    class of one group — no `hail` and no `lightning`, hence no STORM — writes
    a five-group probe without complaint. Serving it would answer a documented
    six-key `phenomenon_scores` contract with five keys, which is the exact
    symptom the paragraph above says this function exists to prevent. So the
    probe's groups must be *exactly* GROUP_ORDER. Order may differ: the
    response is keyed by name, and the weights travel with their labels.
    """
    if not path.exists():
        return None

    payload = np.load(path, allow_pickle=False)
    groups = [str(group) for group in payload["groups"]]

    unknown = [group for group in groups if group not in GROUP_ORDER]
    if unknown:
        raise ValueError(f"probe was trained on unknown groups: {', '.join(unknown)}")

    if sorted(groups) != sorted(GROUP_ORDER):
        raise ValueError(
            f"probe must carry every group in {GROUP_ORDER}, got {groups}"
        )

    return Probe(groups=groups, weights=payload["weights"], bias=payload["bias"])
