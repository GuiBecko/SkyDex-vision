"""HTTP surface of the SkyDex vision service.

This service answers two questions about a photograph and nothing else:
whether it looks like an outdoor sky, and which of six weather groups it
resembles. It returns numbers. It does not know what a capture is, what a
threshold is, or what the SkyDex backend will do with the answer — that
policy lives in Kotlin, next to the rest of the validation semantics.

Keeping the split there rather than here is what lets a threshold change ship
without touching Python, and what keeps this service testable as a pure
function of an image.
"""

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.model import InvalidImageError, VisionModel, load_model

app = FastAPI(title="skydex-vision", version="1.0.0")


class AnalyzeResponse(BaseModel):
    outdoor_score: float
    phenomenon_scores: dict[str, float]
    model: str


def get_model() -> VisionModel:
    """Indirection so tests can override the model without importing torch."""
    return load_model()


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness only. It deliberately does not touch the model: an endpoint
    that loads 350MB of weights to answer is not a health check."""
    return {"status": "ok"}


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    model: VisionModel = Depends(get_model),
) -> AnalyzeResponse:
    """Score one photograph.

    A 400 means the bytes are not an image. There is no status here for "this
    photo is fraudulent" because this service does not have that opinion.
    """
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="The uploaded file is not an image")

    payload = await file.read()
    try:
        outdoor, phenomenon = model.analyze(payload)
    except InvalidImageError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return AnalyzeResponse(
        outdoor_score=outdoor,
        phenomenon_scores=phenomenon,
        model=model.name,
    )
