# Golden set — provisional

30 images, 15 `sky` / 15 `fraud`, scored by two tests, one per head:

- `tests/test_accuracy.py` uses all 30. It exercises the **outdoor** head —
  the `sky`/`fraud` split in `manifest.csv` is its ground truth.
- `tests/test_phenomenon_golden.py` uses the 15 `sky` rows. It exercises the
  **phenomenon** head, and its ground truth is the group in the filename
  (`sky_cloudy_04.jpg` → CLOUDY), standing in for what the weather API would
  have reported at capture time. `sky_night_01.jpg` is scored but not judged:
  the backend skips the phenomenon check at night, because CLIP cannot tell a
  clear night from an overcast one. That test's bar is therefore 14
  photographs, not 15.

The second test arrived late — six commits after the first — and the gap
between them was not free. For as long as only the outdoor head was measured
here, nothing in this repo ever pointed the phenomenon head at a real
photograph, and a probe shipped that scored a clear blue sky as FOG strongly
enough for the backend to refuse it. Held-out Kaggle accuracy went *up* over
that same period. If a third head is ever added, it wants its own row in this
list on the same commit.

Because the filename now carries the expected group, the naming convention is
load-bearing rather than cosmetic: `sky_<group>_<nn>.jpg`, with `<group>` one
of the six in `app/prompts.py::GROUP_ORDER`, or `night`. A `sky` row named
anything else fails that test loudly rather than being quietly skipped.

## Status: provisional, not the real thing yet

Task 5 as briefed asks for 15 real skies "from your own phone" — actual
captures from actual cameras that will point at actual weather, because
"the public datasets do not contain the fraud you are trying to block."

This machine has no camera and no phone to source those photos from, so the
`sky` half of this set is openly-licensed camera photography pulled from
Wikimedia Commons instead: real photographs, real cameras, real weather, but
not photos of *this deployment's* skies taken under *this deployment's*
conditions. That gap matters — a Commons photographer's idea of "cloudy" may
not match what SkyDex's actual users submit.

The `fraud` half is closer to the intended spirit: 4 of the 15 are genuine
screenshots captured on this machine (a weather site, a forum, a code page,
a wiki article — via headless Chrome, not synthesized), and the remaining 11
are real photographs of the fraud categories the spec calls out (indoor
rooms, faces, blank walls, monitors/phone screens displaying something,
close-up objects), also sourced from Wikimedia Commons.

**Before this set is trusted as the go-live gate**, replace the `sky` rows
with actual phone photos of actual skies, spread across the conditions listed
in the task brief (4 clear, 4 cloudy, 3 rain, 2 storm, 1 fog, 1 night). The
`fraud` rows are more defensible as-is, but a few real phone-camera shots of
an indoor room or a blank wall — the actual failure mode being guarded
against — would strengthen them too.

## In-sample tuning and regression signal

The prompts in `app/prompts.py` have been tuned against this very golden set,
but **only the sky side** — `SKY_PROMPTS` was widened; `NOT_SKY_PROMPTS` was
not. Three new sky prompts were added to clear images the original three would
have rejected: an outdoor landscape under a rainy sky (wet moorland), dense
outdoor fog (low visibility sky), and raindrops on a car window (weather seen
through glass).

Widening `SKY_PROMPTS` raises `outdoor_score` for every sky-adjacent image.
That includes frauds that are sky-adjacent — a weather forecast page dense with
blue-sky graphics and sun icons (`fraud_screenshot_01.jpg`) is the most
sky-adjacent fraud in this set. Therefore, **both the 0% false-positive rate
and the 100% fraud-caught rate reported by this test are in-sample and
optimistic**. As of this commit: frauds top out at `fraud_wall_02.jpg`, 0.0547,
the closest any fraud comes to the 0.60 gate, and still far below it. Honest
skies bottom out at `sky_clear_04.jpg`, 0.7831. The threshold therefore sits
inside a wide empty band, and nothing in this set is near it. However, this
set contains no genuinely hard fraud — there is no photograph of a large screen
displaying a full-frame sky — so a harder set would narrow the band. And the
`outdoor_score` values were not recorded systematically during tuning, so
whether those tuning edits eroded margins or not is unknown.

None of this makes the test useless. Its real job is as a **regression guard**:
it fails if a future prompt edit or model retrain breaks images that the
current prompts handle correctly. Regression detection does not require an
unbiased accuracy estimate; it requires a stable baseline and visible margins.
The prompts we have are that baseline, and the 0.0547–0.7831 margin is its
measurement. The next person to edit the prompts can re-run this test and watch
whether the margins shrink or grow.

The unbiased estimate of how well the prompts generalize comes from elsewhere:
the held-out Kaggle test set used in the next task, which the prompts have
never been tuned against. That is where we learn whether a 0% false-positive
rate on Wikimedia Commons photos (the sky half of this set) also holds for
phone photos taken in this deployment's actual conditions.

## Provenance

Every row in `manifest.csv` carries a `source` column:

- `own-screenshot (...)` — captured on this machine with headless Chrome
  against a real, live page. Not synthetic: it is a genuine screen capture
  of genuine rendered content, just not of a phone.
- A `https://commons.wikimedia.org/wiki/File:...` URL — an openly-licensed
  photograph pulled from Wikimedia Commons. Follow the link for the
  photographer, license and original resolution.

No image in this set is AI-generated, synthetic, or a stock-photo
composite. Every `sky` row is a real camera photograph of a real sky; every
`fraud` row is either a real screen capture or a real camera photograph of
the fraud scenario it is labeled for.

## Regenerating

Images were fetched at a 1024px-wide rendition directly from Wikimedia
(`Special:FilePath/<name>.jpg?width=1024`) or captured at native screenshot
resolution, then normalized with Pillow to a 1024px long edge, quality 85,
stripped to plain RGB JPEG. There is no single script that reproduces this
set end-to-end — the sourcing (picking which photograph represents "storm"
vs "rain") was manual curation against Commons search results, which is
exactly the manual step Task 5 expects a human to do with their own camera
roll instead.
