# Training the phenomenon probe

`data/probe.npz` is the trained phenomenon head this service ships with: a
linear probe over CLIP ViT-B/32 embeddings, mapping a 512-dimensional image
embedding onto the six visual groups in `app/prompts.py::GROUP_ORDER`.

It is committed to this repository — 25KB — because without it the service
falls back to the zero-shot head, which is a supported but measurably worse
configuration. `data/golden/README.md` records the case that made this
concrete: a zero-shot-adjacent probe once scored a clear blue sky as FOG
strongly enough for the backend to refuse the upload.

## Provenance

Trained on 2026-08-16 from two Kaggle datasets:

| Dataset | Creator | Used for | Licence |
|---|---|---|---|
| [Weather Image Recognition](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset) | Jehan Bhathena | every group except CLEAR and CLOUDY | [CC0 1.0 Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) |
| [Multi-class Weather Dataset](https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset) | Prateek Srivastava | CLEAR and CLOUDY, and the held-out test split | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

The second row's licence requires attribution, and this table is where that
obligation is discharged: **the CLEAR and CLOUDY coefficients in
`data/probe.npz` derive from Prateek Srivastava's Multi-class Weather Dataset,
used under CC BY 4.0.**

`SOURCE_TO_GROUP` in `train_probe.py` maps source classes onto the six groups;
classes absent from it are skipped, which is how `sandstorm` is dropped.

The committed file contains learned coefficients, not images. No photograph
from either dataset is redistributed by this repository.

## Reproducing it

See the "Retrain" section of the repository README. In short:

    .venv/bin/python training/train_probe.py split \
        data/_raw_multiclass/"Multi-class Weather Dataset" data/train data/test
    .venv/bin/python training/train_probe.py train data/train data/probe.npz
    .venv/bin/pytest -s -m slow

The last step is not optional. The notebook measures the probe on held-out
Kaggle images drawn from the same skewed distribution it trains on;
`tests/test_phenomenon_golden.py` measures it on real photographs. A retrain
has already once improved the first while breaking the second.
