"""build_variety_gallery.py (server-free): the deliverable itself is one park with one
policy-built coaster per requested footprint family, produced ON DEMAND from its seed.
This exercises the family-request plumbing end-to-end (pin_family + track_recorder +
run_episode, all reused unchanged from run_model.py) with a stub "model" instead of a
real MaskablePPO checkpoint, plus the pure summary-table formatting."""
import numpy as np
import pytest

import openrct2_gym.envs.openrct2_env as oe_mod
from openrct2_gym.envs import footprint
from openrct2_gym.envs.openrct2_env import OpenRCT2Env
from stable_baselines3.common.vec_env import DummyVecEnv

import build_variety_gallery as G
from run_model import find_curriculum_wrapper, make_inference_env, track_recorder
from openrct2_gym.tests.test_reward import CompletingAPI


class _StubModel:
    """Always "predicts" action 0 -- paired with CompletingAPI (which treats every
    non-station piece as valid and closes the loop after 2 of them), this drives a
    short, deterministic, all-straight build without any real policy weights."""

    def predict(self, obs, action_masks=None, deterministic=True):
        return np.array([0]), None


def _stub_env(monkeypatch, tmp_path):
    monkeypatch.setattr(OpenRCT2Env, "_LOOP_LIBRARY_PATH", str(tmp_path / "lib.jsonl"))
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    single_env = make_inference_env(port=8080, start_phase=6, game_speed=1)
    wrapper = find_curriculum_wrapper(single_env)
    base_env = wrapper._get_base_env()
    recorder = track_recorder(wrapper, base_env)
    env = DummyVecEnv([lambda: single_env])
    return env, wrapper, recorder


# --------------------------------------------------------------- --families parsing

def test_parse_families_accepts_the_default_spread():
    assert G.parse_families("0,1,2,3,4") == [0, 1, 2, 3, 4]
    assert G.parse_families(" 0 , 2 ") == [0, 2]


def test_parse_families_rejects_out_of_range_with_clear_message():
    with pytest.raises(SystemExit, match="out of range"):
        G.parse_families(str(footprint.FAMILY_N))
    with pytest.raises(SystemExit, match="out of range"):
        G.parse_families("-1")


def test_parse_families_rejects_non_int_and_empty():
    with pytest.raises(SystemExit):
        G.parse_families("oval")
    with pytest.raises(SystemExit):
        G.parse_families("")


# --------------------------------------------------------------- build_one_family

def test_build_one_family_pins_the_requested_seed_and_reports_honestly(monkeypatch, tmp_path):
    env, wrapper, recorder = _stub_env(monkeypatch, tmp_path)
    model = _StubModel()

    row = G.build_one_family(env, model, wrapper, recorder, z=3, deterministic=True)

    assert row["z"] == 3
    assert row["requested"] == footprint.FAMILIES[3][0]
    assert row["closed"] is True                      # CompletingAPI closes after 2 pieces
    assert row["pieces"] == 2
    assert row["record"] is not None
    assert row["record"].actions == (0, 0)
    # An all-straight build has 0 turns/0 switches -> classifies as oval (index 0),
    # not the requested winding (index 3): an honest miss, not a fabricated hit.
    assert row["built"] == footprint.FAMILIES[0][0]
    assert row["hit"] is False
    env.close()


def test_build_one_family_survives_a_non_closing_episode(monkeypatch, tmp_path):
    """CompletingAPI never closes at z-independent phase-1 lengths if the piece budget
    truncates first; here we drop the completion threshold out of reach so the episode
    truncates, and confirm the row still reports honestly (closed=False, a real record
    of whatever was placed) instead of raising."""
    monkeypatch.setattr(OpenRCT2Env, "_LOOP_LIBRARY_PATH", str(tmp_path / "lib.jsonl"))

    class NeverCompletingAPI(CompletingAPI):
        complete_after = 10_000     # far past any phase's piece budget -> truncates

    monkeypatch.setattr(oe_mod, "APIController", NeverCompletingAPI)
    single_env = make_inference_env(port=8080, start_phase=6, game_speed=1)
    wrapper = find_curriculum_wrapper(single_env)
    base_env = wrapper._get_base_env()
    recorder = track_recorder(wrapper, base_env)
    env = DummyVecEnv([lambda: single_env])
    model = _StubModel()

    row = G.build_one_family(env, model, wrapper, recorder, z=1, deterministic=True)

    assert row["closed"] is False
    assert row["pieces"] > 0
    assert row["record"] is not None                   # a real, replayable partial build
    assert row["excitement"] == 0.0                     # never tested -> no ride_rating key
    env.close()


# --------------------------------------------------------------- summary table

def _row(requested, built, hit, closed, excitement, pieces):
    return {"requested": requested, "built": built, "hit": hit, "closed": closed,
            "excitement": excitement, "pieces": pieces}


def test_summary_table_keeps_a_did_not_close_row_instead_of_dropping_it():
    rows = [
        _row("oval", "oval", True, True, 6.2, 24),
        _row("serpentine", "none", False, False, 0.0, 41),   # missed AND never closed
    ]
    table = G.format_summary_table(rows)
    lines = table.splitlines()
    assert any("serpentine" in ln and "did not close" in ln for ln in lines), table
    assert any("oval" in ln and "closed" in ln for ln in lines), table
    # every requested row must appear -- a missing row is worse than an honest one
    assert sum(1 for ln in lines if "requested" not in ln and "---" not in ln) == len(rows)


def test_summary_table_reports_hit_and_miss_distinctly():
    rows = [
        _row("winding", "winding", True, True, 5.0, 40),
        _row("winding", "spiral", False, True, 5.0, 40),
    ]
    table = G.format_summary_table(rows)
    lines = [ln for ln in table.splitlines() if "winding" in ln]
    assert any("HIT" in ln for ln in lines)
    assert any("MISS" in ln for ln in lines)
