"""Golden-set regression for the phenomenon head.

`tests/test_accuracy.py` validates the outdoor head. Nothing validated the
phenomenon head against a real photograph until this file existed, and that
gap is what let a retrained probe start scoring clear blue skies as FOG
without a single test going red: the probe's accuracy was measured only on
held-out Kaggle images, drawn from the same skewed distribution it trained on.

So this test does what no accuracy number can: it runs the *backend's own
decision* over the golden set's real sky photographs and counts how many
honest captures that decision would refuse. That is the number the go-live bar
is about.

It is slow — it loads the real model — so it is marked and excluded from the
fast suite. Run it after every prompt edit and after every probe retrain, with
`-s` so the per-photo table is printed on success as well as on failure:

    .venv/bin/pytest tests/test_phenomenon_golden.py -v -s -m slow

Which head is measured is a deployment fact, not a test choice: `VisionModel`
uses `data/probe.npz` when that file is present and zero-shot when it is not,
and `test_the_shipped_phenomenon_head_blocks_no_honest_photograph` asks
whichever one is mounted. The second test scores both heads on the same
embeddings, so the comparison that decides which one to mount is a permanent,
re-runnable measurement rather than a throwaway script.

The golden set is provisional in the same way it is for the outdoor head — see
`data/golden/README.md`. Its `sky` half is openly-licensed Wikimedia camera
photography, not phone photos of this deployment's skies. It is still, by a
wide margin, the closest thing in this repo to what a user will actually send.
"""

import csv
from pathlib import Path

import pytest

from app.model import load_model
from app.prompts import GROUP_ORDER

GOLDEN = Path(__file__).parent.parent / "data" / "golden"

# --- Duplicated from the Kotlin backend's specification -----------------------------
#
# SOURCE OF TRUTH: `PhotoAuthenticityService` in the SkyDex Kotlin backend will
# own the stage-2 phenomenon decision — this matrix and both gate constants.
# That class does not exist yet. It is specified, in Kotlin, at
#
#     docs/superpowers/plans/2026-08-16-skydex-ai-validation-integration.md
#
# as `PhotoAuthenticityService.contradicts` and its `RECONCILABLE` companion
# map, with the two gates bound to `skydex.vision.expected-score-max` and
# `skydex.vision.top-score-min`. The copy below was transcribed from there row
# for row. So grepping the backend for the class name finds nothing today —
# read the plan instead. Once that plan is executed the Kotlin becomes the
# source of truth and this becomes the duplicate: change one, grep for the
# other, and make the Kotlin carry a pointer back to this file.
#
# The duplication is unavoidable here: this service returns numbers only and
# has no opinion about verdicts, but the question this test exists to answer
# ("would an honest capture be refused?") cannot be asked without the rule that
# refuses it. The alternative — measuring nothing until the backend integration
# is running end to end — is what let the FOG problem survive seven tasks.
#
# If these three values and the Kotlin ones ever disagree, this test measures a
# decision nobody makes.

# A capture is refused only when the expected group is this far down AND some
# other group is this far up. One gate alone is not enough: a photograph whose
# expected group merely lost is ambiguous, not fraudulent.
EXPECTED_SCORE_MAX = 0.10
TOP_SCORE_MIN = 0.70

# Which top-scoring group is reconcilable with the group the weather API
# reported for that moment. Asymmetric on purpose: an overcast sky can
# plausibly photograph as fog, rain or storm, but a clear sky cannot
# plausibly photograph as anything but clear, and snow is unmistakable.
RECONCILABLE_WITH_EXPECTED = {
    "CLEAR": {"CLEAR"},
    "CLOUDY": {"CLOUDY", "FOG", "RAIN", "STORM"},
    "FOG": {"FOG"},
    "RAIN": {"RAIN", "CLOUDY", "FOG", "STORM"},
    "SNOW": {"SNOW"},
    "STORM": {"STORM", "CLOUDY", "FOG", "RAIN"},
}

# --- The bar ------------------------------------------------------------------------
#
# Zero, not "under 2%". The spec's go-live bar is a false-positive rate under
# 2%, but this set holds 14 honest daytime photographs, and 1/14 is 7.1% — the
# only value under 2% that 14 samples can produce is 0. Asserting the
# percentage would dress up an assertion that can only ever mean "zero" as
# something statistically stronger than it is. Assert zero and say so.
MAX_BLOCKED_HONEST_PHOTOGRAPHS = 0


def load_sky_photographs() -> list[tuple[Path, str, bool]]:
    """``(path, expected group, is_day)`` for every `sky` row in the manifest.

    The expected group stands in for what the weather API would have reported
    at capture time; here it comes from the filename (`sky_cloudy_04.jpg` ->
    CLOUDY), which the manifest's own naming convention guarantees.

    `sky_night_01.jpg` carries no phenomenon in its name because it is a night
    photograph, and the backend skips the phenomenon check at night — CLIP
    cannot tell a clear night from an overcast one, so any verdict drawn from
    a dark frame would be a coin flip presented as a judgement. It is returned
    with ``is_day=False`` rather than dropped, so the table below shows it
    being skipped instead of silently omitting a row.
    """
    with (GOLDEN / "manifest.csv").open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["label"] == "sky"]

    photographs = []
    for row in sorted(rows, key=lambda r: r["filename"]):
        name = row["filename"]
        group = name.removeprefix("sky_").rsplit("_", 1)[0].upper()
        if group == "NIGHT":
            photographs.append((GOLDEN / "images" / name, "CLEAR", False))
            continue
        assert group in GROUP_ORDER, (
            f"{name} does not name one of {GROUP_ORDER} — either the manifest's "
            "naming convention changed or this photograph is mislabelled, and "
            "either way this test cannot tell what it should have scored"
        )
        photographs.append((GOLDEN / "images" / name, group, True))
    return photographs


def safety_margin(expected: str, scores: dict[str, float]) -> float:
    """How far this photograph is from being refused. Negative means refused.

    Refusal needs *both* numeric gates — the expected group below the floor and
    some rival above the ceiling — so escaping needs only one of them, and the
    distance to refusal is the better of the two slacks: how far the expected
    group's score sits above the floor, and how far the strongest *rival*
    group's score sits below the ceiling.

    The rival, not the top group. Those are the same thing whenever the head is
    wrong about the top group, which is the case this number is really about.
    They differ when the expected group *is* the top group, and measuring
    against the top group there measured the expected group against itself:
    the margin became ``max(s - 0.10, 0.70 - s)``, which bottoms out at +0.300
    around s = 0.40 and then *rises* as s falls further. A head degrading from
    CLEAR 0.360 to CLEAR 0.200 printed +0.340 and then +0.500, reading as a
    safer photograph. Against the strongest rival there is no such branch:
    probability mass leaving the expected group has to arrive somewhere else,
    so both slacks shrink together and no degradation can move this number away
    from zero. That is what makes it worth watching in the table — a regression
    that halves the slack without crossing zero is visible before it fails.

    The sign is unchanged by that switch, so this is still exactly the block
    condition. A margin can only go negative when some group is above 0.70, and
    a group above 0.70 while the expected group is below 0.10 is necessarily
    the top group.

    The matrix is not folded in here: a reconcilable pair is safe outright, no
    matter what the numbers say. Callers report that separately, because a
    reconcilable photograph with a margin of -0.2 is a photograph that would be
    refused the moment the weather API reported a different group for it.
    """
    expected_score = scores.get(expected, 0.0)
    rival = max((score for group, score in scores.items() if group != expected), default=0.0)
    return max(expected_score - EXPECTED_SCORE_MAX, TOP_SCORE_MIN - rival)


def is_blocked(expected: str, scores: dict[str, float], is_day: bool) -> bool:
    """The backend's stage-2 decision, reproduced. See the duplication note above.

    Degrades to "no opinion" on anything it cannot evaluate, matching the
    planned Kotlin line for line: `contradicts` does
    `VisualGroup.fromNameOrNull(top.key) ?: return false` and
    `scores[expectedGroup.name] ?: 0.0`, because "a check that did not run must
    never cost a user their capture". This copy used to index the matrix and
    the score dict directly, so a group name from a newer model would have
    raised KeyError here and returned false there. Unreachable today — both
    sides share GROUP_ORDER and `load_probe` refuses unknown names — but the
    two copies must not diverge on the first seventh group.
    """
    if not is_day or not scores:
        return False
    top = max(scores, key=scores.get)
    if top not in GROUP_ORDER or expected not in GROUP_ORDER:
        return False
    if top in RECONCILABLE_WITH_EXPECTED[expected]:
        return False
    return safety_margin(expected, scores) < 0


def report(head: str, rows: list[tuple[str, str, bool, dict[str, float]]]) -> int:
    """Print the per-photo table for one head and return the number blocked."""
    print(f"\n--- {head} ---")
    print(
        f"{'photograph':<22} {'expected':<9} {'top group':<15} "
        f"{'exp score':<9} {'margin':<9} verdict"
    )
    blocked = 0
    for name, expected, is_day, scores in rows:
        top = max(scores, key=scores.get)
        margin = safety_margin(expected, scores)
        reconcilable = top in RECONCILABLE_WITH_EXPECTED.get(expected, set())

        if not is_day:
            verdict = "night — phenomenon check skipped"
        elif is_blocked(expected, scores, is_day):
            blocked += 1
            verdict = "BLOCKED — an honest capture would be refused"
        elif reconcilable:
            verdict = f"ok (matrix admits {top} for {expected})"
        else:
            verdict = "ok (gates)"

        print(
            f"{name:<22} {expected:<9} {top + ' ' + format(scores[top], '.3f'):<15} "
            f"{scores.get(expected, 0.0):<9.3f} {margin:<+9.3f} {verdict}"
        )
    return blocked


def score_golden_set(model) -> dict[str, list[tuple[str, str, bool, dict[str, float]]]]:
    """Score every golden sky photograph through both heads, embedding once.

    Returns rows keyed by head name; the probe key is absent when no probe is
    mounted, which is exactly how the service behaves.
    """
    photographs = load_sky_photographs()
    assert photographs, "the golden set has no sky photographs — see data/golden/README.md"

    rows: dict[str, list[tuple[str, str, bool, dict[str, float]]]] = {"zero-shot": []}
    for path, expected, is_day in photographs:
        assert path.exists(), f"manifest names a missing file: {path}"
        features = model.embed(path.read_bytes())

        rows["zero-shot"].append((path.name, expected, is_day, model.zero_shot_phenomenon(features)))

        probed = model.probed_phenomenon(features)
        if probed is not None:
            rows.setdefault("probe", []).append((path.name, expected, is_day, probed))
    return rows


@pytest.fixture(scope="module")
def golden_scores():
    """Embedding 15 photographs costs seconds; both tests below reuse one pass."""
    return score_golden_set(load_model())


@pytest.mark.slow
# Same carve-out, same reason, as tests/test_accuracy.py: loading the cached
# CLIP checkpoint calls torch.load with the still-default weights_only=False,
# which torch 2.4 warns about on every load of a trusted local file. Scoped
# here rather than added to pytest.ini's shared filterwarnings block.
@pytest.mark.filterwarnings("ignore:You are using `torch.load`:FutureWarning")
def test_the_shipped_phenomenon_head_blocks_no_honest_photograph(golden_scores):
    """Whichever head is mounted must refuse none of these real skies.

    A refused honest capture is the worst failure this feature has: it punishes
    the user who did exactly what was asked. Accepted fraud costs fake XP; a
    rejected honest user costs a user.
    """
    model = load_model()
    head = "probe" if "probe" in golden_scores else "zero-shot"
    rows = golden_scores[head]

    print(f"\nshipped phenomenon head: {head} (model name: {model.name})")
    blocked = report(head, rows)
    daytime = sum(1 for _, _, is_day, _ in rows if is_day)
    print(f"\nblocked {blocked}/{daytime} honest daytime photographs (bar: {MAX_BLOCKED_HONEST_PHOTOGRAPHS})")

    assert blocked <= MAX_BLOCKED_HONEST_PHOTOGRAPHS


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:You are using `torch.load`:FutureWarning")
def test_both_phenomenon_heads_are_compared_on_real_photographs(golden_scores):
    """Both heads, same photographs, side by side — the comparison that decides
    which one to mount.

    The zero-shot head is the fallback the service runs whenever
    `data/probe.npz` is absent, so it is a shipping configuration and gets the
    same bar as the probe. The FOG/SNOW tallies are reported rather than
    asserted: they are the leading indicator that made the block explicable —
    the backend's matrix admits only FOG for FOG and only SNOW for SNOW, so a
    head that drifts toward those two groups is a head about to start refusing
    honest captures — but they are a diagnosis, not the bar itself.
    """
    if "probe" not in golden_scores:
        pytest.skip("no data/probe.npz mounted — there is no second head to compare")

    zero_shot, probe = golden_scores["zero-shot"], golden_scores["probe"]

    blocked_zero_shot = report("zero-shot", zero_shot)
    blocked_probe = report("probe", probe)

    differing = []
    fog_or_snow = {"zero-shot": 0, "probe": 0}
    for (name, _, _, zs), (_, _, _, pr) in zip(zero_shot, probe):
        zs_top, pr_top = max(zs, key=zs.get), max(pr, key=pr.get)
        if zs_top != pr_top:
            differing.append(f"  {name:<22} zero-shot {zs_top} {zs[zs_top]:.3f}   probe {pr_top} {pr[pr_top]:.3f}")
        fog_or_snow["zero-shot"] += zs_top in ("FOG", "SNOW")
        fog_or_snow["probe"] += pr_top in ("FOG", "SNOW")

    total = len(zero_shot)
    daytime = sum(1 for _, _, is_day, _ in zero_shot if is_day)
    print(f"\n--- heads compared over {total} real sky photographs ---")
    print(f"top group differs on {len(differing)}/{total}")
    for line in differing:
        print(line)
    print(f"scored FOG or SNOW:  zero-shot {fog_or_snow['zero-shot']}/{total}, probe {fog_or_snow['probe']}/{total}")
    print(f"blocked:             zero-shot {blocked_zero_shot}/{daytime}, probe {blocked_probe}/{daytime}")

    assert blocked_zero_shot <= MAX_BLOCKED_HONEST_PHOTOGRAPHS
    assert blocked_probe <= MAX_BLOCKED_HONEST_PHOTOGRAPHS


# --- The decision itself, on hand-written scores -------------------------------------
#
# Fast, unmarked, no model: `safety_margin` and `is_blocked` are arithmetic over a
# score dict, and the properties below are the ones the slow tests above rely on
# but cannot demonstrate — a golden set of 15 photographs samples the score space
# far too sparsely to show that the margin never rewards a worse head.


def _degrade(scores: dict[str, float], expected: str, amount: float) -> dict[str, float]:
    """Move ``amount`` of probability mass off ``expected``, spread over the rest."""
    others = [group for group in scores if group != expected]
    return {
        group: (scores[group] - amount if group == expected else scores[group] + amount / len(others))
        for group in scores
    }


def test_safety_margin_never_rises_as_the_head_degrades():
    # The property the docstring promises and the old top-group form broke: a
    # head losing confidence in the expected group must never print a larger
    # margin. Walked over the whole range, including the top == expected branch
    # the old form was U-shaped on (it bottomed out at +0.300 near 0.40 and rose
    # from there, so CLEAR 0.360 -> 0.200 read as *safer*).
    scores = {group: 0.08 for group in GROUP_ORDER}
    scores["CLEAR"] = 1.0 - 0.08 * 5

    previous = safety_margin("CLEAR", scores)
    for _ in range(55):
        scores = _degrade(scores, "CLEAR", 0.01)
        current = safety_margin("CLEAR", scores)
        assert current <= previous + 1e-12, f"margin rose to {current} from {previous} at {scores}"
        previous = current


def test_safety_margin_is_negative_exactly_when_the_gates_block():
    # The sign is the whole contract: `is_blocked` reads it. Both gates are
    # strict, matching the Kotlin's `expectedScore < max && top > min`.
    for expected_score in (0.02, 0.09, 0.10, 0.30):
        for top_score in (0.50, 0.70, 0.71, 0.95):
            scores = {group: 0.0 for group in GROUP_ORDER}
            scores["CLEAR"] = expected_score
            scores["SNOW"] = top_score
            gates_block = expected_score < EXPECTED_SCORE_MAX and top_score > TOP_SCORE_MIN
            assert (safety_margin("CLEAR", scores) < 0) is gates_block


def test_is_blocked_says_nothing_about_a_group_it_has_never_heard_of():
    # A group name from a newer model must cost nobody their capture. The
    # planned Kotlin returns false via `VisualGroup.fromNameOrNull(top.key) ?:
    # return false`; this copy used to raise KeyError instead.
    scores = {"CLEAR": 0.01, "TORNADO": 0.99}

    assert is_blocked("CLEAR", scores, is_day=True) is False


def test_is_blocked_skips_a_night_photograph_and_an_empty_analysis():
    confident_contradiction = {group: 0.0 for group in GROUP_ORDER}
    confident_contradiction["SNOW"] = 0.99

    assert is_blocked("CLEAR", confident_contradiction, is_day=True) is True
    assert is_blocked("CLEAR", confident_contradiction, is_day=False) is False
    assert is_blocked("CLEAR", {}, is_day=True) is False
