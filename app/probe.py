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

    def apply(self, embedding: np.ndarray) -> dict[str, float]:
        """Group probabilities for one normalised CLIP embedding."""
        flat = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if flat.shape[0] != self.weights.shape[1]:
            raise ValueError(
                f"probe expects {self.weights.shape[1]}-dimensional embeddings, "
                f"got {flat.shape[0]}"
            )

        logits = self.weights @ flat + self.bias
        # Max subtraction for the same overflow reason as scoring.softmax.
        exponentiated = np.exp(logits - logits.max())
        probabilities = exponentiated / exponentiated.sum()
        return {group: float(value) for group, value in zip(self.groups, probabilities)}


def load_probe(path: Path) -> Probe | None:
    """Read a probe from ``path``, or ``None`` if there is nothing there.

    An unreadable or wrongly-labelled probe raises rather than being skipped.
    Silently falling back to zero-shot would mean a deploy that thought it
    shipped a trained model and did not, with no signal anywhere.
    """
    if not path.exists():
        return None

    payload = np.load(path, allow_pickle=False)
    groups = [str(group) for group in payload["groups"]]

    unknown = [group for group in groups if group not in GROUP_ORDER]
    if unknown:
        raise ValueError(f"probe was trained on unknown groups: {', '.join(unknown)}")

    return Probe(groups=groups, weights=payload["weights"], bias=payload["bias"])
