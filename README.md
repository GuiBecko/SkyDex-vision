# skydex-vision

Scores a photograph for the SkyDex backend: how much it looks like an outdoor
sky, and which of six weather groups it resembles. It returns numbers only —
every threshold and every verdict lives in the Kotlin backend.

## Run it

    docker compose up -d
    curl -F "file=@photo.jpg" http://localhost:8000/v1/analyze

## Develop

Python 3.11 (not 3.14 — PyTorch has no wheels for it):

    python3.11 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements-dev.txt \
        -c constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu
    .venv/bin/pytest                                    # fast suite
    .venv/bin/pytest tests/test_accuracy.py -s -m slow  # golden-set regression

`constraints.txt` pins the CPU builds of torch and torchvision. Without it,
`open_clip_torch`'s unpinned `torch>=1.9.0` lets pip's resolver pull the CUDA
wheel straight over the CPU one, which is how this install balloons by
several gigabytes on a machine with no NVIDIA GPU.

The fast suite is 34 passed, 1 deselected (the golden-set regression, marked
`slow`). Run the fast suite after every change; run the slow one before a
release.

## Retrain

1. Put the Kaggle sets under `data/train/` and `data/test/` (see `training/train_probe.py`)
2. `.venv/bin/python training/train_probe.py train data/train data/probe.npz`
3. Run `training/train.ipynb` for accuracy, the confusion matrix and the threshold curve
4. Re-run the golden-set regression
5. `docker compose restart skydex-vision`

`training/train_probe.py` caches CLIP embeddings to
`data/train_embeddings.npz` and validates that cache against a fingerprint of
the source tree before reusing it. A re-split or a changed `data/train` is
picked up automatically on the next run — there is no cache file to delete by
hand.

Without `data/probe.npz` the service runs zero-shot and reports
`clip-vit-b-32-zeroshot-v1` as its model name. With it, `...-probe-v1`.

## The contract

`POST /v1/analyze`, multipart field `file`:

```json
{
  "outdoor_score": 0.94,
  "phenomenon_scores": {
    "CLEAR": 0.02, "CLOUDY": 0.11, "FOG": 0.04,
    "RAIN": 0.62, "SNOW": 0.01, "STORM": 0.20
  },
  "model": "clip-vit-b-32-zeroshot-v1"
}
```

`400` means the bytes are not a readable image. There is no status for
"fraudulent" — this service does not have that opinion.

## Design

`docs/superpowers/specs/2026-08-16-ai-photo-validation-design.md`
