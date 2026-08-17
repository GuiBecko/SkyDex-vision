"""Coverage for the startup-refusal mechanism in app/main.py.

A whole-repo review found that a corrupt or unknown-group data/probe.npz made
every request return 500 while reloading 350MB of CLIP each time, because
`functools.lru_cache` does not memoize exceptions. The fix loads the model
eagerly in the FastAPI lifespan, so a bad model artefact kills the process at
startup instead of surviving to serve traffic. Nothing exercised that fix, so
a refactor could quietly restore the original bug while every other test kept
passing. These tests monkeypatch the model-construction path rather than
loading real CLIP, so they belong in the fast suite.
"""

import asyncio

import pytest

from app import main as main_module
from app import model as model_module


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """`load_model` is `lru_cache(maxsize=1)`d at module scope, so a result or
    a poisoned cache from one test would otherwise leak into the next one."""
    model_module.load_model.cache_clear()
    yield
    model_module.load_model.cache_clear()


def _run_lifespan_once() -> None:
    async def _run():
        async with main_module.lifespan(main_module.app):
            pass

    asyncio.run(_run())


def test_lifespan_raises_when_the_model_cannot_be_built(monkeypatch):
    """A construction failure -- a corrupt or unknown-group probe is the
    reachable case -- must surface as a startup exception, which is what
    kills the uvicorn process, rather than being swallowed or deferred to the
    first request."""

    def broken_load_model():
        raise ValueError("probe was trained on unknown groups: TORNADO")

    monkeypatch.setattr(main_module, "load_model", broken_load_model)

    with pytest.raises(ValueError, match="unknown groups"):
        _run_lifespan_once()


def test_successful_startup_builds_the_model_exactly_once(monkeypatch):
    """The half of the fix that was actually broken, and the half a refactor
    is most likely to undo: `lifespan` builds the model eagerly, and every
    later `get_model()` call from a request must resolve to that same cached
    build, not trigger a fresh construction. This patches `VisionModel`
    itself rather than `load_model`, so the real `lru_cache` on `load_model`
    is the thing under test -- a regression that dropped that cache, or that
    had `get_model` build a fresh `VisionModel` directly, would show up here
    as more than one construction."""

    calls = []

    class StubVisionModel:
        def __init__(self):
            calls.append(1)

    monkeypatch.setattr(model_module, "VisionModel", StubVisionModel)

    _run_lifespan_once()
    assert len(calls) == 1

    # Simulate several requests going through the same dependency the route
    # uses. None of them should trigger another construction.
    for _ in range(3):
        main_module.get_model()

    assert len(calls) == 1
