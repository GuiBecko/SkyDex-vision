"""The only file in this service that loads PyTorch.

Text embeddings for both prompt sets are computed once, at construction, and
kept on the instance. A request therefore costs exactly one image encode plus
two matrix multiplies against small precomputed matrices — roughly 250ms on one
CPU core for ViT-B/32.
"""

import io
import os
from functools import lru_cache
from pathlib import Path

import open_clip
import torch
from PIL import Image, UnidentifiedImageError

from app.probe import Probe, load_probe
from app.prompts import GROUP_SIZES, OUTDOOR_PROMPTS, PHENOMENON_PROMPTS, SKY_PROMPTS
from app.scoring import group_scores, outdoor_score

MODEL_ARCHITECTURE = "ViT-B-32"
MODEL_WEIGHTS = "laion2b_s34b_b79k"
PROBE_PATH = Path(os.environ.get("SKYDEX_PROBE_PATH", "data/probe.npz"))


class InvalidImageError(ValueError):
    """The uploaded bytes are not an image Pillow can open."""


class VisionModel:
    """CLIP, plus the two precomputed text matrices it is queried against."""

    def __init__(self) -> None:
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            MODEL_ARCHITECTURE, pretrained=MODEL_WEIGHTS
        )
        self._model.eval()
        tokenizer = open_clip.get_tokenizer(MODEL_ARCHITECTURE)

        self._outdoor_features = self._encode_text(tokenizer, OUTDOOR_PROMPTS)
        self._phenomenon_features = self._encode_text(tokenizer, PHENOMENON_PROMPTS)

        self._probe: Probe | None = load_probe(PROBE_PATH)
        suffix = "probe-v1" if self._probe else "zeroshot-v1"
        self.name = f"clip-{MODEL_ARCHITECTURE.lower()}-{suffix}"

    def _encode_text(self, tokenizer, texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            features = self._model.encode_text(tokenizer(texts))
        # Normalising here means the dot product below IS the cosine similarity,
        # so nothing downstream has to remember to divide.
        return features / features.norm(dim=-1, keepdim=True)

    def embed(self, image_bytes: bytes) -> torch.Tensor:
        """The normalised image embedding. Exposed because training reuses it."""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise InvalidImageError("The uploaded file is not a readable image") from error

        with torch.no_grad():
            features = self._model.encode_image(self._preprocess(image).unsqueeze(0))
        return features / features.norm(dim=-1, keepdim=True)

    def zero_shot_phenomenon(self, image_features: torch.Tensor) -> dict[str, float]:
        """Group scores from the untrained head, for an ``embed`` result.

        Public because the phenomenon head has two implementations and the
        golden-set comparison in tests/test_phenomenon_golden.py has to score
        one embedding through both of them. Without this, that test would have
        to reach into private state to re-derive one line of arithmetic.
        """
        similarities = (image_features @ self._phenomenon_features.T)[0].tolist()
        return group_scores(similarities, GROUP_SIZES)

    def probed_phenomenon(self, image_features: torch.Tensor) -> dict[str, float] | None:
        """Group scores from the trained head, or ``None`` when no probe is mounted.

        The ``None`` is a deployment switch, not an error: no ``data/probe.npz``
        means the service runs zero-shot, which is a supported configuration.
        Public for the same reason as ``zero_shot_phenomenon``.
        """
        if self._probe is None:
            return None
        return self._probe.apply(image_features[0].numpy())

    def analyze(self, image_bytes: bytes) -> tuple[float, dict[str, float]]:
        """``(outdoor_score, phenomenon_scores)`` for one photograph.

        The outdoor head is always zero-shot. Only the phenomenon head is
        trained, because the fraud catalogue in NOT_SKY_PROMPTS is a moving
        target that prompts express better than a fixed training set does.
        """
        image_features = self.embed(image_bytes)

        outdoor_similarities = (image_features @ self._outdoor_features.T)[0].tolist()
        outdoor = outdoor_score(outdoor_similarities, sky_count=len(SKY_PROMPTS))

        phenomenon = self.probed_phenomenon(image_features)
        if phenomenon is None:
            phenomenon = self.zero_shot_phenomenon(image_features)

        return outdoor, phenomenon


@lru_cache(maxsize=1)
def load_model() -> VisionModel:
    """Built once per process. The first call takes several seconds."""
    return VisionModel()
