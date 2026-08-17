"""Golden-set regression for the outdoor head.

One of the two tests that decide whether phase 3 of the SkyDex integration
ships, and it covers the **outdoor** head only. `tests/test_phenomenon_golden.py`
covers the phenomenon head; neither stands in for the other, and for six
commits this file was the only one of the pair, which is how a probe that
scored clear blue skies as FOG shipped with everything green.

It is slow — it loads the real model — so it is marked and excluded from the
fast suite. Run it after every edit to SKY_PROMPTS or NOT_SKY_PROMPTS. A probe
retrain cannot move these numbers by construction — the outdoor head is always
zero-shot, see `app/model.py::VisionModel.analyze` — so a retrain wants
`tests/test_phenomenon_golden.py`. Running both after either change costs
seconds: `.venv/bin/pytest -s -m slow`.

    .venv/bin/pytest tests/test_accuracy.py -v -s -m slow

The golden set itself is provisional. See data/golden/README.md: the `sky`
half is openly-licensed Wikimedia camera photography rather than phone photos
taken in this deployment's actual conditions, so this test calibrates the
threshold but does not yet prove domain fit. The prompts in app/prompts.py have
been tuned against this set — specifically, only SKY_PROMPTS was widened, which
raises outdoor_score for sky-adjacent images including frauds — so both the
false-positive rate and the fraud-caught rate reported here are regression
signals, not generalization estimates. The nearest thing to an unbiased
estimate this repo has is the held-out Kaggle measurement in
`training/train.ipynb` — and it covers the phenomenon head, not this one.
"""

import csv
from pathlib import Path

import pytest

from app.model import load_model

GOLDEN = Path(__file__).parent.parent / "data" / "golden"

# The threshold the SkyDex backend will use for stage 1, as
# `skydex.vision.outdoor-min`. That property does not exist in the backend yet;
# it is specified at docs/superpowers/plans/2026-08-16-skydex-ai-validation-integration.md,
# which also has not been executed, so grepping
# SkyDex-backend/src/main/resources/application.properties for it finds nothing
# today. Once the plan runs, keep the two in sync — a drift between them means
# this test measures something nobody runs.
OUTDOOR_THRESHOLD = 0.60

# The go-live bar from the spec. False positives are honest skies rejected as
# fraud, and they are far worse than the reverse: accepted fraud costs fake XP,
# a rejected honest user costs a user.
MAX_FALSE_POSITIVE_RATE = 0.02

# Deliberately loose. Some fraud is genuinely hard for CLIP (a high-quality
# photo of a monitor showing a sky is very nearly a photo of a sky), and
# tightening this at the cost of the rate above would be the wrong trade.
MIN_FRAUD_CAUGHT_RATE = 0.70


def load_manifest() -> list[tuple[Path, str]]:
    # The manifest carries a third column, `source` (a Wikimedia file URL or
    # `own-screenshot`), recording provenance for every image. csv.DictReader
    # reads it along with the rest of the row; this function only needs
    # `filename` and `label`, so the extra column is simply ignored here.
    with (GOLDEN / "manifest.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return [(GOLDEN / "images" / row["filename"], row["label"]) for row in rows]


@pytest.mark.slow
# pytest.ini turns warnings into errors and already carves out the timm
# FutureWarning. This one is separate and not this task's ignore list to grow:
# loading the cached CLIP checkpoint calls torch.load with the (still-default)
# weights_only=False, which torch 2.4 warns about on every load of a trusted
# local file. Scoped to this one slow test rather than pytest.ini's shared
# filterwarnings block.
@pytest.mark.filterwarnings("ignore:You are using `torch.load`:FutureWarning")
def test_outdoor_head_meets_the_go_live_bar():
    model = load_model()
    manifest = load_manifest()
    assert manifest, "the golden set is empty — see data/golden/README.md"

    false_positives, skies = 0, 0
    caught, frauds = 0, 0
    misses: list[str] = []

    min_sky_score = float('inf')
    min_sky_name = ""
    max_fraud_score = float('-inf')
    max_fraud_name = ""

    for path, label in manifest:
        assert path.exists(), f"manifest names a missing file: {path}"
        outdoor, _ = model.analyze(path.read_bytes())
        accepted = outdoor >= OUTDOOR_THRESHOLD

        if label == "sky":
            skies += 1
            if outdoor < min_sky_score:
                min_sky_score = outdoor
                min_sky_name = path.name
            if not accepted:
                false_positives += 1
                misses.append(f"REJECTED A REAL SKY  {path.name}  outdoor={outdoor:.3f}")
        else:
            frauds += 1
            if outdoor > max_fraud_score:
                max_fraud_score = outdoor
                max_fraud_name = path.name
            if not accepted:
                caught += 1
            else:
                misses.append(f"ACCEPTED A FRAUD     {path.name}  outdoor={outdoor:.3f}")

    false_positive_rate = false_positives / skies
    caught_rate = caught / frauds

    print(f"\nfalse positive rate: {false_positive_rate:.1%} (bar: {MAX_FALSE_POSITIVE_RATE:.0%})")
    print(f"fraud caught rate:   {caught_rate:.1%} (bar: {MIN_FRAUD_CAUGHT_RATE:.0%})")
    print(f"margin: max fraud {max_fraud_name} {max_fraud_score:.4f}, min sky {min_sky_name} {min_sky_score:.4f}")
    for line in misses:
        print("  " + line)

    assert false_positive_rate <= MAX_FALSE_POSITIVE_RATE
    assert caught_rate >= MIN_FRAUD_CAUGHT_RATE
