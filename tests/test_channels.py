"""Declarative per-link channel plans: attack kinds, parsing, reproducibility, serialisation."""

import json

import pytest

from evefl.fl.channels import (
    ChannelPlan,
    LinkAttack,
    attacked_links,
    build_plan,
    parse_link_attack,
    parse_link_noise,
)
from evefl.quantum.base import ChannelModel


# --------------------------------------------------------------------------
# Attack kinds
# --------------------------------------------------------------------------

def test_static_attack_is_active_every_round_on_its_link_only():
    plan = ChannelPlan(attacks={"0": LinkAttack(alpha=0.6)})
    assert all(plan.channel_for("0", r) == ChannelModel(0.6, 0.0) for r in (1, 5, 50))
    assert plan.channel_for("1", 3) == ChannelModel(0.0, 0.0)
    assert plan.channel_for("2", 3) == ChannelModel(0.0, 0.0)


def test_step_and_window_attacks_follow_their_rounds():
    step = ChannelPlan(attacks={"1": LinkAttack(alpha=0.5, kind="step", start_round=20)})
    assert step.channel_for("1", 19).intercept_probability == 0.0
    assert step.channel_for("1", 20).intercept_probability == 0.5 and step.channel_for("1", 99).intercept_probability == 0.5
    window = ChannelPlan(attacks={"2": LinkAttack(alpha=0.4, kind="window", start_round=21, end_round=30)})
    inside = [window.channel_for("2", r).intercept_probability for r in (20, 21, 30, 31)]
    assert inside == [0.0, 0.4, 0.4, 0.0]


def test_intermittent_attack_is_reproducible_independent_of_other_links_and_has_the_right_rate():
    attack = LinkAttack(alpha=0.5, kind="intermittent", probability=0.3, seed=7)
    plan = ChannelPlan(attacks={"0": attack})
    first = [plan.channel_for("0", r).intercept_probability > 0 for r in range(1, 2001)]
    again = [plan.channel_for("0", r).intercept_probability > 0 for r in range(1, 2001)]
    assert first == again
    assert 0.26 < sum(first) / len(first) < 0.34
    other_seed = ChannelPlan(attacks={"0": LinkAttack(alpha=0.5, kind="intermittent", probability=0.3, seed=8)})
    assert first != [other_seed.channel_for("0", r).intercept_probability > 0 for r in range(1, 2001)]
    # the activation of link 0 does not depend on what other links do
    both = ChannelPlan(attacks={"0": attack, "1": LinkAttack(alpha=0.9, kind="intermittent", probability=0.5, seed=7)})
    assert first == [both.channel_for("0", r).intercept_probability > 0 for r in range(1, 2001)]
    assert [plan.channel_for("1", r).intercept_probability for r in range(1, 50)] == [0.0] * 49


def test_intermittent_probability_extremes():
    never = ChannelPlan(attacks={"0": LinkAttack(alpha=1.0, kind="intermittent", probability=0.0)})
    always = ChannelPlan(attacks={"0": LinkAttack(alpha=1.0, kind="intermittent", probability=1.0)})
    assert all(never.channel_for("0", r).intercept_probability == 0.0 for r in range(1, 100))
    assert all(always.channel_for("0", r).intercept_probability == 1.0 for r in range(1, 100))


def test_noise_defaults_and_per_link_overrides():
    plan = ChannelPlan(link_noise={"1": 0.03}, default_noise=0.01)
    assert plan.channel_for("0", 1).bit_flip_probability == 0.01
    assert plan.channel_for("1", 1).bit_flip_probability == 0.03


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"alpha": 1.5}, {"alpha": -0.1}, {"alpha": 0.5, "kind": "nope"}, {"alpha": 0.5, "start_round": 0},
    {"alpha": 0.5, "kind": "window"}, {"alpha": 0.5, "kind": "window", "start_round": 5, "end_round": 4},
    {"alpha": 0.5, "kind": "intermittent", "probability": 1.2},
])
def test_invalid_attacks_are_rejected(kwargs):
    with pytest.raises(ValueError):
        LinkAttack(**kwargs)


def test_invalid_noise_is_rejected():
    for bad in (-0.01, 0.6):
        with pytest.raises(ValueError):
            ChannelPlan(default_noise=bad)
        with pytest.raises(ValueError):
            ChannelPlan(link_noise={"0": bad})


# --------------------------------------------------------------------------
# CLI parsing
# --------------------------------------------------------------------------

def test_parse_link_attack_forms():
    assert parse_link_attack("0:0.6") == ("0", LinkAttack(alpha=0.6))
    assert parse_link_attack("1:0.6:step@20") == ("1", LinkAttack(alpha=0.6, kind="step", start_round=20))
    assert parse_link_attack("2:0.4:window@21-30") == ("2", LinkAttack(0.4, "window", 21, 30))
    cid, attack = parse_link_attack("0:0.5:intermittent@0.3", seed=9)
    assert cid == "0" and attack == LinkAttack(0.5, "intermittent", probability=0.3, seed=9)


@pytest.mark.parametrize("bad", ["", "0", ":0.5", "0:0.5:nope@1", "0:0.5:step", "0:x", "0:0.5:window@5", "a:b:c:d"])
def test_parse_link_attack_rejects_malformed_specs(bad):
    with pytest.raises(ValueError):
        parse_link_attack(bad)


def test_parse_link_noise():
    assert parse_link_noise("1:0.02") == ("1", 0.02)
    for bad in ("", "1", "1:0.02:3", ":0.1"):
        with pytest.raises(ValueError):
            parse_link_noise(bad)


def test_build_plan_combines_specs_and_rejects_duplicates():
    plan = build_plan(["0:0.6", "2:0.3:step@5"], ["1:0.02"], default_noise=0.01, seed=3)
    assert attacked_links(plan) == ["0", "2"]
    assert plan.channel_for("1", 1).bit_flip_probability == 0.02 and plan.channel_for("0", 1).bit_flip_probability == 0.01
    with pytest.raises(ValueError, match="more than one attack"):
        build_plan(["0:0.6", "0:0.3"])
    with pytest.raises(ValueError, match="more than one noise"):
        build_plan([], ["1:0.01", "1:0.02"])
    assert build_plan().attacks == {}


# --------------------------------------------------------------------------
# Serialisation: the full plan must be savable and reproducible
# --------------------------------------------------------------------------

def test_plan_round_trips_through_json():
    plan = build_plan(["0:0.6:window@3-8", "1:0.4:intermittent@0.25"], ["2:0.02"], default_noise=0.005, seed=11)
    restored = ChannelPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert restored == plan
    assert [restored.channel_for(c, r) for c in "012" for r in range(1, 20)] == \
        [plan.channel_for(c, r) for c in "012" for r in range(1, 20)]


def test_empty_plan_is_all_clean():
    plan = ChannelPlan.from_dict({})
    assert plan.channel_for("0", 1) == ChannelModel(0.0, 0.0)
