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
    .venv/bin/pytest                    # fast suite
    .venv/bin/pytest -s -m slow         # both golden-set regressions

`constraints.txt` pins the CPU builds of torch and torchvision. Without it,
`open_clip_torch`'s unpinned `torch>=1.9.0` lets pip's resolver pull the CUDA
wheel straight over the CPU one, which is how this install balloons by
several gigabytes on a machine with no NVIDIA GPU.

The fast suite is 34 passed, 3 deselected. The three deselected ones load the
real model, so they are marked `slow`, and there is one per head:

- `tests/test_accuracy.py` — the **outdoor** head against the golden set:
  false-positive and fraud-caught rates, plus the margin either side of the
  0.60 threshold.
- `tests/test_phenomenon_golden.py` — the **phenomenon** head against the same
  golden set, scored through the SkyDex backend's own stage-2 decision. It
  asserts that no honest sky photograph would be refused, and prints a
  per-photo table for both heads (trained probe and zero-shot) so a shrinking
  margin is visible before it becomes a failure. Run it with `-s`; the table
  is the point, and pytest only shows prints on failure without it.

Run the fast suite after every change; run both slow ones before a release and
after every probe retrain.

## Retrain

1. Put the Kaggle sets under `data/train/` and `data/test/` (see `training/train_probe.py`)
2. `.venv/bin/python training/train_probe.py train data/train data/probe.npz`
3. Run `training/train.ipynb` for accuracy, the confusion matrix and the threshold curve
   (`.venv/bin/jupyter nbconvert --to notebook --execute --inplace training/train.ipynb`)
4. Re-run **both** golden-set regressions — `.venv/bin/pytest -s -m slow`
5. `docker compose restart skydex-vision`

Step 4 is not optional and is not covered by step 3. The notebook measures the
probe on held-out Kaggle images, which are drawn from the same skewed
distribution the probe trains on; `tests/test_phenomenon_golden.py` measures it
on real photographs of the sky. A retrain has already once improved the
notebook's numbers while making the service refuse an honest photograph of a
clear blue sky — the probe had learned the training pool's 67%-FOG-and-SNOW
prior. `training/train_probe.py::make_classifier` now fits with
`class_weight="balanced"` for that reason; read its docstring before changing
the estimator.

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
