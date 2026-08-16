from app import prompts


def test_group_order_is_the_six_visual_groups():
    assert prompts.GROUP_ORDER == ["CLEAR", "CLOUDY", "FOG", "RAIN", "SNOW", "STORM"]


def test_every_group_has_prompts():
    for group in prompts.GROUP_ORDER:
        assert prompts.GROUP_PROMPTS[group], f"{group} has no prompts"


def test_phenomenon_prompts_is_the_groups_concatenated_in_order():
    expected = [p for group in prompts.GROUP_ORDER for p in prompts.GROUP_PROMPTS[group]]

    assert prompts.PHENOMENON_PROMPTS == expected


def test_group_sizes_matches_the_concatenation():
    assert [name for name, _ in prompts.GROUP_SIZES] == prompts.GROUP_ORDER
    assert sum(count for _, count in prompts.GROUP_SIZES) == len(prompts.PHENOMENON_PROMPTS)


def test_outdoor_prompts_puts_sky_first():
    assert prompts.OUTDOOR_PROMPTS[: len(prompts.SKY_PROMPTS)] == prompts.SKY_PROMPTS
