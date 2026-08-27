# Third-party components

## OpenCLIP and its weights

This service loads `ViT-B-32` with the `laion2b_s34b_b79k` pretrained weights
through [`open_clip_torch`](https://github.com/mlfoundations/open_clip), which
is MIT-licensed.

The weights themselves are **not redistributed in this repository**. They are
downloaded from the Hugging Face Hub — about 350MB — the first time a model is
constructed.

**The Docker image is a different matter.** `Dockerfile` deliberately bakes the
weights in at build time, so that a container start does not wait on a download
and a container without internet access still works. Any image built from this
repository therefore *contains* the LAION-2B weights, and redistributing that
image redistributes them. Their terms are the ones published with the
`laion2b_s34b_b79k` checkpoint.

## Golden-set images

The 30 photographs in `data/golden/images/` are third-party works under their
own licences, not this repository's MIT. See `data/golden/LICENSE-IMAGES.md`.

## Trained probe

`data/probe.npz` holds coefficients learned from two Kaggle datasets. No image
from either is redistributed here. See `training/README.md`.

## Python dependencies

`requirements.txt` pins FastAPI, uvicorn, python-multipart, Pillow, NumPy and
`open_clip_torch`; `constraints.txt` pins the CPU builds of torch and
torchvision. All are permissively licensed (MIT, BSD or Apache-2.0). Run
`pip-licenses` in the virtualenv for the current per-package detail.
