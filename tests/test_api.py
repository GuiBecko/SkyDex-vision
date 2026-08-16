import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, get_model
from app.prompts import GROUP_ORDER


class StubModel:
    """Stands in for CLIP so the route contract can be tested in milliseconds.

    The real model is exercised by tests/test_accuracy.py, which is a different
    kind of test with a different failure meaning: this one breaks when the HTTP
    contract changes, that one breaks when the model gets worse.
    """

    name = "stub-v0"

    def __init__(self, outdoor: float = 0.9):
        self.outdoor = outdoor
        self.calls: list[bytes] = []

    def analyze(self, image_bytes: bytes) -> tuple[float, dict[str, float]]:
        self.calls.append(image_bytes)
        scores = {group: 0.1 for group in GROUP_ORDER}
        scores["RAIN"] = 0.5
        return self.outdoor, scores


@pytest.fixture
def stub():
    model = StubModel()
    app.dependency_overrides[get_model] = lambda: model
    yield model
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


def jpeg_bytes(colour: str = "blue") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), colour).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_reports_ok(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_analyze_returns_the_documented_shape(client, stub):
    response = client.post(
        "/v1/analyze",
        files={"file": ("sky.jpg", jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"outdoor_score", "phenomenon_scores", "model"}
    assert body["model"] == "stub-v0"
    assert body["outdoor_score"] == pytest.approx(0.9)
    assert set(body["phenomenon_scores"]) == set(GROUP_ORDER)


def test_analyze_passes_the_uploaded_bytes_to_the_model(client, stub):
    payload = jpeg_bytes("red")

    client.post("/v1/analyze", files={"file": ("sky.jpg", payload, "image/jpeg")})

    assert stub.calls == [payload]


def test_analyze_rejects_a_file_that_is_not_an_image(client, stub):
    response = client.post(
        "/v1/analyze",
        files={"file": ("notes.txt", b"this is not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_analyze_rejects_a_missing_file(client, stub):
    assert client.post("/v1/analyze").status_code == 422
