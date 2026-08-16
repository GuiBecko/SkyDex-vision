"""The only file in this service that loads PyTorch.

Text embeddings for both prompt sets are computed once, at construction, and
kept on the instance. A request therefore costs exactly one image encode plus
two matrix multiplies against small precomputed matrices — roughly 250ms on one
CPU core for ViT-B/32.
"""

import io
from functools import lru_cache

import open_clip
import torch
from PIL import Image, UnidentifiedImageError

from app.prompts import GROUP_SIZES, OUTDOOR_PROMPTS, PHENOMENON_PROMPTS, SKY_PROMPTS
from app.scoring import group_scores, outdoor_score

MODEL_ARCHITECTURE = "ViT-B-32"
MODEL_WEIGHTS = "laion2b_s34b_b79k"


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

        self.name = f"clip-{MODEL_ARCHITECTURE.lower()}-zeroshot-v1"

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

    def analyze(self, image_bytes: bytes) -> tuple[float, dict[str, float]]:
        """``(outdoor_score, phenomenon_scores)`` for one photograph."""
        image_features = self.embed(image_bytes)

        outdoor_similarities = (image_features @ self._outdoor_features.T)[0].tolist()
        phenomenon_similarities = (image_features @ self._phenomenon_features.T)[0].tolist()

        return (
            outdoor_score(outdoor_similarities, sky_count=len(SKY_PROMPTS)),
            group_scores(phenomenon_similarities, GROUP_SIZES),
        )


@lru_cache(maxsize=1)
def load_model() -> VisionModel:
    """Built once per process. The first call takes several seconds."""
    return VisionModel()
