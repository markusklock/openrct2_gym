"""run_model.py wiring (server-free): the showcase entry point must rebuild the
TRAINING-equivalent env (cold, correct phase) and expose the flags a mature-checkpoint
demo needs -- a P6 policy in a Phase-1 env truncates at 40 pieces mid-build."""
import pytest

import openrct2_gym.envs.openrct2_env as oe_mod
from openrct2_gym.envs import footprint
from openrct2_gym.envs.openrct2_env import OpenRCT2Env

import run_model as R
from openrct2_gym.tests.test_reward import CompletingAPI, FakeAPI


def test_make_inference_env_is_cold_and_phase_aware(monkeypatch, tmp_path):
    from openrct2_gym.envs.openrct2_env import OpenRCT2Env
    monkeypatch.setattr(OpenRCT2Env, "_LOOP_LIBRARY_PATH", str(tmp_path / "lib.jsonl"))
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    env = R.make_inference_env(port=8080, start_phase=6, game_speed=1)
    w = env
    while not hasattr(w, "current_phase"):
        w = w.env
    assert w.current_phase == 6
    assert w.warm_start_enabled is False          # inference shows the TRUE task
    base = w
    while hasattr(base, "env"):
        base = base.env
    assert base.max_track_length == 90             # P6 budget, not Phase 1's 40
    env.close()


def test_cli_exposes_showcase_flags(monkeypatch):
    import argparse
    seen = []
    real_add = argparse.ArgumentParser.add_argument

    def spy(self, *a, **kw):
        seen.append(a)
        return real_add(self, *a, **kw)

    monkeypatch.setattr(argparse.ArgumentParser, "add_argument", spy)
    try:
        R.parse_args(["--model", "x"])
    except SystemExit:
        pass
    flags = [a[0] for a in seen if a and isinstance(a[0], str)]
    for flag in ("--start-phase", "--game-speed", "--sample", "--episodes", "--family"):
        assert flag in flags


# --------------------------------------- Task 10: ask the policy for a specific coaster
#
# The whole point of --family is that it holds. ImprovedPhasedCurriculumWrapper.reset()
# redraws base_env.target_family from self._sample_target_family() on EVERY episode
# (improved_phased_curriculum_wrapper.py reset()), so anything that fails to survive a
# reset would silently produce five randomly-seeded coasters that still *look* like a
# successful "on demand" gallery -- nobody would notice from the output alone. These
# tests exercise the pin directly against the real wrapper (start_phase=6, where all
# five families in PHASE_FAMILIES are active), not a mock.

def _cold_inference_env(monkeypatch, tmp_path, api=FakeAPI):
    monkeypatch.setattr(OpenRCT2Env, "_LOOP_LIBRARY_PATH", str(tmp_path / "lib.jsonl"))
    monkeypatch.setattr(oe_mod, "APIController", api)
    return R.make_inference_env(port=8080, start_phase=6, game_speed=1)


def test_find_curriculum_wrapper_locates_the_instance(monkeypatch, tmp_path):
    env = _cold_inference_env(monkeypatch, tmp_path)
    wrapper = R.find_curriculum_wrapper(env)
    assert type(wrapper).__name__ == "ImprovedPhasedCurriculumWrapper"
    env.close()


def test_pin_family_holds_across_many_resets(monkeypatch, tmp_path):
    """The sampler pin is the whole mechanism (Task 10 brief): prove it directly rather
    than trusting a single reset. Unpinned first, to prove the sampler is genuinely
    exercised (all-families-phase-6, seeded RNG) -- an environment where every draw
    happened to already be the same value would make the pinned half meaningless."""
    env = _cold_inference_env(monkeypatch, tmp_path)
    wrapper = R.find_curriculum_wrapper(env)
    base_env = wrapper._get_base_env()

    wrapper._family_rng.seed(7)
    seen = set()
    for _ in range(20):
        env.reset()
        seen.add(base_env.target_family)
    assert len(seen) > 1, "sampler never varied -- test wouldn't catch a broken pin"

    R.pin_family(wrapper, 3)
    for _ in range(20):
        env.reset()
        assert base_env.target_family == 3
    env.close()


def test_pin_family_rejects_out_of_range_with_clear_message():
    """family_match() indexes footprint.FAMILIES directly and would raise a bare,
    unhelpful IndexError deep in the reward calc -- pin_family must fail fast instead."""
    class _StubWrapper:
        pass

    stub = _StubWrapper()
    for bad in (-1, footprint.FAMILY_N, footprint.FAMILY_N + 5):
        with pytest.raises(SystemExit, match="out of range"):
            R.pin_family(stub, bad)
    assert not hasattr(stub, "_sample_target_family")   # rejected before any mutation


def test_family_omitted_leaves_sampling_behaviour_unchanged(monkeypatch, tmp_path):
    """--family defaults to None; main()'s pin call is gated on `is not None`, so the
    wrapper's sampler must be untouched (identity-equal to the original bound method)
    when it is omitted."""
    env = _cold_inference_env(monkeypatch, tmp_path)
    wrapper = R.find_curriculum_wrapper(env)
    original_sampler = wrapper._sample_target_family

    args = R.parse_args(["--model", "x"])
    assert args.family is None
    if args.family is not None:            # mirrors main()'s gating exactly
        R.pin_family(wrapper, args.family)

    assert wrapper._sample_target_family == original_sampler   # same bound method (not overridden)
    env.close()


def test_track_recorder_snapshots_before_the_next_reset_clears_it(monkeypatch, tmp_path):
    """DummyVecEnv resets the underlying env the instant a terminal step() returns
    (SB3's step_wait: step, then reset-if-done, inside one outer call) -- by the time a
    caller sees done=True, OpenRCT2Env.track_builder.history is already cleared for the
    next episode. track_recorder must have captured it INSIDE the terminal step() call,
    before any later reset() wipes it -- proven here by calling reset() again and
    checking the live history is gone while the recorder's snapshot survives."""
    env = _cold_inference_env(monkeypatch, tmp_path, api=CompletingAPI)
    wrapper = R.find_curriculum_wrapper(env)
    R.pin_family(wrapper, 3)
    base_env = wrapper._get_base_env()
    env.reset()
    state = R.track_recorder(wrapper, base_env)

    terminated = truncated = False
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    assert terminated or truncated, "CompletingAPI should have closed the loop by now"
    captured = list(state["actions"])
    assert captured                                             # non-empty finished track

    env.reset()                                                 # simulates DummyVecEnv's auto-reset
    assert list(base_env.track_builder.history) == []           # live history is gone
    assert state["actions"] == captured                         # snapshot unaffected
    env.close()
