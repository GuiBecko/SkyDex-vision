"""The text side of both CLIP heads.

Prompt wording carries most of the accuracy in a zero-shot CLIP classifier.
"rain" and "a photo of a rainy overcast sky, raindrops, wet ground" are not
close in embedding space, and the gap between them is several accuracy points.
Treat this file as a tunable parameter, not as boilerplate. Which regression an
edit needs depends on which half of the file it touched, and the two do not
overlap:

- SKY_PROMPTS / NOT_SKY_PROMPTS — the outdoor head — tests/test_accuracy.py
- GROUP_PROMPTS — the zero-shot phenomenon head — tests/test_phenomenon_golden.py

`test_accuracy.py` cannot observe GROUP_PROMPTS at all, so a maintainer who
rewrites every phenomenon prompt, re-runs only that test and sees green has
measured nothing. `.venv/bin/pytest -s -m slow` runs both.
"""

# --- Head 1: is this an outdoor sky? -------------------------------------------------
#
# The NOT_SKY set is the fraud catalogue: it is what people actually point a
# camera at when they are not pointing it at the sky.

SKY_PROMPTS = [
    "a photo of the sky",
    "an outdoor photo of clouds in the sky",
    "a photograph taken outdoors looking up at the sky",
    # The three prompts above assume the sky itself fills the frame. Real
    # submissions are often a landscape shot with the sky as context (wet
    # moorland under a rain-laden sky), or a sky seen through weather that
    # scatters or obscures it (dense fog, raindrops on glass). Without these,
    # CLIP read those photos as closer to an indoor scene or a close-up
    # object than to "the sky". The misses that prompted this are recorded in
    # data/golden/README.md's "In-sample tuning" section — the repository state
    # that produced them no longer exists, so there is nothing left to re-run.
    "an outdoor photograph of a landscape under a rainy, overcast sky",
    "a photo of dense outdoor fog with a hazy, low-visibility sky",
    "raindrops on a car window with an outdoor sky and scene visible through the glass",
]

NOT_SKY_PROMPTS = [
    "a screenshot of a phone screen",
    "an indoor photo of a room",
    "a selfie of a person's face",
    "a photo of a blank wall",
    "a photo of a printed photograph",
    "a close-up photo of an object on a table",
]

OUTDOOR_PROMPTS = SKY_PROMPTS + NOT_SKY_PROMPTS

# --- Head 2: which visual group? -----------------------------------------------------
#
# Six groups, not nine phenomena. Drizzle, rain and rain showers are the same
# photograph; so are thunderstorms and hailstorms. Asking CLIP to separate them
# would be asking it to guess.

GROUP_ORDER = ["CLEAR", "CLOUDY", "FOG", "RAIN", "SNOW", "STORM"]

GROUP_PROMPTS: dict[str, list[str]] = {
    "CLEAR": [
        "a photo of a clear blue sky with no clouds",
        "a photo of bright sunshine in a cloudless sky",
    ],
    "CLOUDY": [
        "a photo of an overcast grey sky full of clouds",
        "a photo of a cloudy sky with thick white clouds",
    ],
    "FOG": [
        "a photo of thick fog with very low visibility",
        "a photo of a misty foggy landscape",
    ],
    "RAIN": [
        "a photo of a rainy overcast sky, raindrops, wet ground",
        "a photo taken during rainfall, wet street, grey sky",
    ],
    "SNOW": [
        "a photo of snow falling, snow covered ground",
        "a photo of a snowy winter landscape",
    ],
    "STORM": [
        "a photo of a dark dramatic storm sky with lightning",
        "a photo of heavy thunderstorm clouds, very dark sky",
    ],
}

PHENOMENON_PROMPTS = [p for group in GROUP_ORDER for p in GROUP_PROMPTS[group]]

GROUP_SIZES = [(group, len(GROUP_PROMPTS[group])) for group in GROUP_ORDER]
