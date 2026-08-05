"""run_model.py wiring (server-free): the showcase entry point must rebuild the
TRAINING-equivalent env (cold, correct phase) and expose the flags a mature-checkpoint
demo needs -- a P6 policy in a Phase-1 env truncates at 40 pieces mid-build."""
import openrct2_gym.envs.openrct2_env as oe_mod

import run_model as R
from openrct2_gym.tests.test_reward import FakeAPI


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
    assert base.max_track_length == 120           # P6 budget, not Phase 1's 40
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
    for flag in ("--start-phase", "--game-speed", "--sample", "--episodes"):
        assert flag in flags
