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

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.model import InvalidImageError, VisionModel, load_model


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load CLIP and the probe before the first request, not on it.

    Eager, and deliberately unguarded: if `data/probe.npz` is corrupt,
    incomplete or wrongly labelled, `load_probe` raises, this raises, uvicorn
    logs "Application startup failed" and the process exits. The container then
    goes unhealthy and an operator sees it.

    Lazy loading looked equivalent and was not. `load_model` is `lru_cache`d and
    `functools.lru_cache` does not memoize exceptions, so a bad probe used to
    500 *every* request while re-reading the 350MB checkpoint each time — a
    one-worker service turned into a CPU amplifier by any retry loop — while
    `/health`, which correctly never touches the model, went on answering 200
    and the compose healthcheck stayed green. There was no signal anywhere.

    Degrading to zero-shot instead was considered and rejected: this service has
    already once shipped a probe that quietly underperformed zero-shot on real
    photographs and nobody noticed for seven tasks. A broken model artefact
    should stop a deploy, not survive one silently.

    The cost is paid once, at startup, which also removes the ~4s penalty the
    first request used to carry.
    """
    load_model()
    yield


app = FastAPI(title="skydex-vision", version="1.0.0", lifespan=lifespan)


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
    that loads 350MB of weights to answer is not a health check.

    It does not need to. The model is loaded in `lifespan` before this route
    can answer at all, so a process that responds here is a process whose
    model and probe loaded — which is what makes the compose healthcheck a
    real signal rather than a green light in front of a service that 500s."""
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
