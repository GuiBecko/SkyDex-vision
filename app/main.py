"""HTTP surface of the SkyDex vision service.

This service answers two questions about a photograph and nothing else:
whether it looks like an outdoor sky, and which of six weather groups it
resembles. It returns numbers. It does not know what a capture is, what a
threshold is, or what the SkyDex backend will do with the answer — that
policy lives in Kotlin, next to the rest of the validation semantics.
"""

from fastapi import FastAPI

app = FastAPI(title="skydex-vision", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness only. It deliberately does not touch the model: an endpoint
    that loads 350MB of weights to answer is not a health check."""
    return {"status": "ok"}
