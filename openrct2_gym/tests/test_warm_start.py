"""Warm-start reverse curriculum: loop library + annealer (pure module, server-free).

The library persists verified/harvested closing action sequences as JSONL (shared across
SubprocVecEnv workers via atomic single-line appends); the annealer decides, per episode,
how much of a loop the env pre-places (prefix) and how much the agent must build (k),
annealing k upward on frontier success until episodes degenerate to cold starts.
"""
import json
import os
import random

import numpy as np
import pytest

from openrct2_gym.envs import openrct2_env as oe_mod
from openrct2_gym.envs.openrct2_env import OpenRCT2Env
from openrct2_gym.envs.warm_start import (
    ACTION_CLIMB_Z,
    ACTION_DROP_Z,
    CHAIN_ACTIONS,
    LoopRecord,
    LoopLibrary,
    WarmStartAnnealer,
    WarmStartPlan,
    generate_candidates,
    generate_hill_candidates,
    generate_p4_candidates,
    generate_p5_candidates,
)
from openrct2_gym.tests.test_env_smoke import FakeAPI
from openrct2_gym.tests.test_reward import CompletingAPI


@pytest.fixture(autouse=True)
def _isolate_env_side_files(tmp_path):
    """Isolate the loop-library file and the closing-geometry cache per test."""
    orig_lib = OpenRCT2Env._LOOP_LIBRARY_PATH if hasattr(OpenRCT2Env, "_LOOP_LIBRARY_PATH") else None
    orig_cache = OpenRCT2Env._close_cache
    orig_cache_path = OpenRCT2Env._CLOSE_CACHE_PATH
    orig_records = OpenRCT2Env._close_records
    OpenRCT2Env._LOOP_LIBRARY_PATH = str(tmp_path / "loop_library_env.jsonl")
    OpenRCT2Env._close_cache = None
    OpenRCT2Env._close_records = []
    OpenRCT2Env._CLOSE_CACHE_PATH = str(tmp_path / "close_geometry.json")
    yield
    if orig_lib is not None:
        OpenRCT2Env._LOOP_LIBRARY_PATH = orig_lib
    OpenRCT2Env._close_cache = orig_cache
    OpenRCT2Env._close_records = orig_records
    OpenRCT2Env._CLOSE_CACHE_PATH = orig_cache_path

# Live-verified sequences (probe run, Jun 2026): racetrack loops closing at [62,66,14] d0.
FLAT = [4, 4, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0]                    # len 12
FLAT_L = [3, 3, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0]                  # len 12, left-handed
HILL = [4, 4, 10, 9, 13, 12, 6, 14, 0, 4, 4, 0]                # len 12, chain climb + descent


def _lib(tmp_path, sequences=()):
    lib = LoopLibrary(str(tmp_path / "loop_library.jsonl"))
    for seq in sequences:
        lib.add(LoopRecord.from_actions(seq, source="scripted"))
    return lib


# ------------------------------------------------------------------- LoopRecord

def test_loop_record_from_actions_coerces_and_counts():
    rec = LoopRecord.from_actions([np.int64(4), np.int64(10), np.int64(9), np.int64(0)],
                                  source="harvest", max_gain=np.float64(3.0))
    assert rec.actions == (4, 10, 9, 0)
    assert all(isinstance(a, int) for a in rec.actions)         # json-serializable
    assert rec.length == 4
    assert rec.chain_count == 2                                 # actions 10 and 9
    assert rec.max_gain == pytest.approx(3.0) and isinstance(rec.max_gain, float)
    assert rec.source == "harvest"


def test_record_from_history_requires_completion_and_measures_gain():
    def entry(action, z_from, z_to, complete=False):
        return {"action": action, "position": [0, 0, z_from], "next_position": [1, 0, z_to],
                "is_complete": complete}
    incomplete = [entry(0, 14, 14), entry(4, 14, 14)]
    assert LoopLibrary.record_from_history(incomplete) is None
    completed = [entry(10, 14, 15), entry(9, 15, 17), entry(6, 17, 14), entry(0, 14, 14, True)]
    rec = LoopLibrary.record_from_history(completed)
    assert rec.actions == (10, 9, 6, 0)
    assert rec.chain_count == 2
    assert rec.max_gain == pytest.approx(3.0)                   # peak z 17 - base 14
    assert LoopLibrary.record_from_history([]) is None


# ------------------------------------------------------------------- LoopLibrary

def test_library_add_dedup_and_roundtrip(tmp_path):
    lib = _lib(tmp_path)
    assert lib.add(LoopRecord.from_actions(FLAT, source="scripted")) is True
    assert lib.add(LoopRecord.from_actions(FLAT, source="harvest")) is False   # dedup on actions
    assert len(lib) == 1
    reloaded = LoopLibrary(lib.path)                            # fresh instance reads the file
    assert len(reloaded) == 1
    (rec,) = reloaded.pool(phase=1, max_len=40)
    assert rec.actions == tuple(FLAT) and rec.source == "scripted"


def test_library_load_ignores_corrupt_lines(tmp_path):
    path = tmp_path / "loop_library.jsonl"
    good = json.dumps({"actions": FLAT, "length": 12, "chain_count": 0,
                       "max_gain": 0.0, "source": "scripted"})
    path.write_text("this is not json\n" + good + "\n" + '{"missing": "actions"}\n')
    lib = LoopLibrary(str(path))
    assert len(lib) == 1                                        # corrupt lines skipped, no raise


def test_library_pool_respects_track_budget(tmp_path):
    lib = _lib(tmp_path, [FLAT])                                # length 12
    assert lib.pool(phase=1, max_len=13) == []                  # 12 > 13 - 2 budget margin
    assert len(lib.pool(phase=1, max_len=14)) == 1


def test_library_pool_phase2_prefers_hill_loops_with_flat_fallback(tmp_path):
    lib = _lib(tmp_path, [FLAT, HILL])
    p2 = lib.pool(phase=2, max_len=40)
    assert [r.actions for r in p2] == [tuple(HILL)]             # hill-only for phase >= 2
    flat_only = _lib(tmp_path.joinpath("flat"), [FLAT])
    assert [r.actions for r in flat_only.pool(phase=2, max_len=40)] == [tuple(FLAT)]  # fallback
    assert len(lib.pool(phase=1, max_len=40)) == 2              # phase 1 uses everything


# ------------------------------------------- steep-aware pool (P4 60-degree scaffold)
# 12h of steep-credit P4 training: the policy never placed a steep piece on its own --
# steep prefixes appeared only at their ~7% pool share (replays of short Phase-2-era
# seeds, too short to qualify). The pool must PREFER qualifying-shaped steep loops in
# P4, and the seed generator must produce them at P4 length.

STEEP_SMALL = [4, 4, 10, 9, 9, 9, 9, 13, 12, 27, 28, 14, 0, 4, 4, 0]        # len 16
BIG_NOSTEEP = [4, 4, 10, 9, 9, 9, 13, 12, 6, 6, 6, 14] + [0] * 26 + [4, 4, 0]  # len 41
BIG_STEEP = [4, 4, 10, 9, 9, 9, 9, 13, 12, 27, 28, 14] + [0] * 26 + [4, 4, 0]  # len 41


def test_loop_record_derives_steep_drop_z(tmp_path):
    """steep_drop_z counts only the 60-degree family (8/27/28) and is derived from the
    action list, so legacy JSONL entries (no steep field persisted) get it on reload."""
    assert LoopRecord.from_actions(STEEP_SMALL, "scripted").steep_drop_z == 8.0
    assert LoopRecord.from_actions(BIG_NOSTEEP, "scripted").steep_drop_z == 0.0
    lib = _lib(tmp_path, [STEEP_SMALL])
    reloaded = LoopLibrary(lib.path)                       # fresh load from JSONL
    assert next(iter(reloaded._records.values())).steep_drop_z == 8.0


def test_library_pool_phase4_prefers_steep_when_required(tmp_path):
    lib = _lib(tmp_path, [BIG_NOSTEEP, BIG_STEEP, STEEP_SMALL])
    best = lib.pool(phase=4, max_len=80, min_chains=3, min_len=40, min_drop_z=8,
                    min_steep_z=8)
    assert [r.actions for r in best] == [tuple(BIG_STEEP)]  # full-criteria tier only
    # No big steep loop yet -> degrade to ANY steep loop (the short seeds), never to
    # the non-steep big loop that dilutes the steep signal.
    no_big = _lib(tmp_path.joinpath("nb"), [BIG_NOSTEEP, STEEP_SMALL])
    tier2 = no_big.pool(phase=4, max_len=80, min_chains=3, min_len=40, min_drop_z=8,
                        min_steep_z=8)
    assert [r.actions for r in tier2] == [tuple(STEEP_SMALL)]
    # No steep anywhere -> the scaffold must not turn off: chained fallback.
    no_steep = _lib(tmp_path.joinpath("ns"), [BIG_NOSTEEP])
    tier3 = no_steep.pool(phase=4, max_len=80, min_chains=3, min_len=40, min_drop_z=8,
                          min_steep_z=8)
    assert [r.actions for r in tier3] == [tuple(BIG_NOSTEEP)]


def test_generate_p4_candidates_are_qualifying_shaped():
    """Every P4 seed skeleton: net-z balanced, carries a full 27/28 steep segment and a
    >=6z chain climb, and is long enough that the closed loop lands at >=40 pieces
    (skeleton >= 37 + the ~3-piece closure tail), within the P4 track budget."""
    cands = generate_p4_candidates()
    assert len(cands) >= 16
    for c in cands:
        assert 37 <= len(c) <= 78
        assert 27 in c and 28 in c
        z, chain_peak = 0, 0
        for a in c:
            z += ACTION_CLIMB_Z.get(a, 0) - ACTION_DROP_Z.get(a, 0)
            if a in CHAIN_ACTIONS:
                chain_peak = max(chain_peak, z)
        assert z == 0                                      # returns to station height
        assert chain_peak >= 6                             # P4 height target reachable


# ---------------------------------------- excitement-tagged records (P5 self-imitation)
# P5 plateaued at E=1.15: quality was invisible to the scaffold because records carry no
# measured rating and the harvest ran BEFORE the ride test. Records now carry excitement,
# harvests run post-test, and a duplicate sequence with a strictly higher measured E
# upgrades the stored record (append + last-line-wins on load).

def test_loop_record_excitement_roundtrips_and_defaults_legacy(tmp_path):
    lib = LoopLibrary(str(tmp_path / "lib.jsonl"))
    assert lib.add(LoopRecord.from_actions(FLAT, source="harvest", excitement=3.4))
    reloaded = LoopLibrary(lib.path)
    assert reloaded._records[tuple(FLAT)].excitement == pytest.approx(3.4)
    with open(lib.path, "a") as f:                      # legacy line: no excitement key
        f.write(json.dumps({"actions": FLAT_L, "source": "harvest", "max_gain": 0.0}) + "\n")
    reloaded2 = LoopLibrary(lib.path)
    assert reloaded2._records[tuple(FLAT_L)].excitement == 0.0


# climb to +13 via chains, one continuous 12z drop (12,27,28,6,14), padded past the P5
# pool's min_len=40 -- a "qualifying-shaped, excitement-taggable" P5 exemplar skeleton
BIG_EXCITING = [4, 4, 10, 9, 9, 9, 9, 9, 9, 13] + [12, 27, 28, 6, 14] + [0] * 23 + [4, 4, 0]


def test_pool_prefers_excited_records_with_fallback(tmp_path):
    lib = _lib(tmp_path)
    lib.add(LoopRecord.from_actions(BIG_EXCITING, "harvest", excitement=5.0))
    lib.add(LoopRecord.from_actions(BIG_STEEP, "harvest"))              # untagged
    lib.add(LoopRecord.from_actions(STEEP_SMALL, "harvest", excitement=4.5))
    best = lib.pool(phase=5, max_len=80, min_chains=1, min_len=40, min_drop_z=12,
                    min_single_drop_z=12, min_excitement=4.0)
    assert [r.actions for r in best] == [tuple(BIG_EXCITING)]
    # no full-criteria record -> ANY excitement-tagged loop, never the untagged big one
    lib2 = _lib(tmp_path.joinpath("l2"), [BIG_STEEP])
    lib2.add(LoopRecord.from_actions(STEEP_SMALL, "harvest", excitement=4.5))
    tier2 = lib2.pool(phase=5, max_len=80, min_chains=1, min_len=40, min_drop_z=12,
                      min_single_drop_z=12, min_excitement=4.0)
    assert [r.actions for r in tier2] == [tuple(STEEP_SMALL)]
    # nothing excited anywhere -> chained fallback keeps the scaffold alive
    lib3 = _lib(tmp_path.joinpath("l3"), [BIG_STEEP])
    tier3 = lib3.pool(phase=5, max_len=80, min_chains=1, min_len=40, min_drop_z=12,
                      min_single_drop_z=12, min_excitement=4.0)
    assert [r.actions for r in tier3] == [tuple(BIG_STEEP)]


def test_best_excitement_respects_budget(tmp_path):
    lib = _lib(tmp_path)
    assert lib.best_excitement(80) == 0.0                               # empty -> 0
    lib.add(LoopRecord.from_actions(STEEP_SMALL, "harvest", excitement=2.0))    # len 16
    lib.add(LoopRecord.from_actions(BIG_EXCITING, "harvest", excitement=5.0))   # len 43
    assert lib.best_excitement(80) == pytest.approx(5.0)
    assert lib.best_excitement(40) == pytest.approx(2.0)                # big one over budget
    untagged = _lib(tmp_path.joinpath("u"), [FLAT])
    assert untagged.best_excitement(80) == 0.0


def test_library_cap_evicts_worst_excitement_for_tagged_newcomers(monkeypatch, tmp_path):
    """Jul-10 live finding: weeks of cross-run harvests left classes OVER the cap (load()
    admits everything), so every novel tagged P4/P5 build was refused and the ratchet
    starved. A newcomer with strictly higher excitement than the class's worst now evicts
    that worst record; untagged newcomers into a full class stay refused (flood guard)."""
    monkeypatch.setattr(LoopLibrary, "MAX_RECORDS_PER_CLASS", 2)
    lib = _lib(tmp_path)
    assert lib.add(LoopRecord.from_actions(FLAT, "harvest")) is True
    assert lib.add(LoopRecord.from_actions(FLAT_L, "harvest")) is True
    third = [4, 4, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5]     # novel variant, still chain-less class
    assert lib.add(LoopRecord.from_actions(third, "harvest")) is False        # flood guard
    tagged = LoopRecord.from_actions(third, "harvest", excitement=1.5)
    assert lib.add(tagged) is True                                  # evicts a worst (E=0.0)
    in_class = [r for r in lib._records.values() if r.source != "scripted"]
    assert len(in_class) == 2
    assert max(r.excitement for r in in_class) == pytest.approx(1.5)
    fourth = [3, 3, 0, 0, 0, 0, 0, 0, 0, 3, 3, 5]
    assert lib.add(LoopRecord.from_actions(fourth, "harvest", excitement=0.5)) is True
    # class is now {1.5, 0.5}: a 0.4 newcomer cannot displace anything
    assert lib.add(LoopRecord.from_actions(FLAT, "harvest", excitement=0.4)) is False


def test_p5_ratchet_keys_ride_the_step_done_info(monkeypatch, tmp_path):
    """The TB callback only reads STEP done-infos -- reset infos are never logged. The
    Jul-10 live run showed exc_bar 'na' for hours because the ratchet keys were emitted
    on the reset path only."""
    wrapper, _ = _wrapped(monkeypatch, tmp_path, api_cls=CompletingAPI, p_cold=0.0)
    wrapper.current_phase = 5
    wrapper._update_phase_settings()
    wrapper._loop_library.add(
        LoopRecord.from_actions(BIG_EXCITING, "harvest", excitement=5.0))
    info = _run_episode(wrapper)
    assert info['library_best_excitement'] == pytest.approx(5.0)
    assert info['p5_pool_exc_bar'] == pytest.approx(4.0)


def test_p5_substage_advance_reanneals(monkeypatch, tmp_path):
    """Each P5 length rung changes the pool (bigger budget) -- the annealer restarts
    like on any phase change (the _advance_phase2_stage precedent)."""
    wrapper, _ = _wrapped(monkeypatch, tmp_path, p_cold=0.0)
    wrapper.current_phase = 5
    wrapper._update_phase_settings()
    wrapper._track_stats = True
    wrapper.phase_episode_count = 60
    wrapper.episode_results.extend([True] * 50)
    wrapper._annealer.k_max = 11
    wrapper._annealer.frontier.append(True)
    assert wrapper.phase5_current_length < wrapper.phase5_target_length
    assert wrapper._check_phase_advancement() is True                   # rung 80 -> 90
    assert wrapper._annealer.k_max == wrapper._annealer.k_init
    assert len(wrapper._annealer.frontier) == 0


def test_generate_p5_candidates_are_exemplar_shaped():
    """P5 exemplar skeletons (Jul-11 rev 2: the map-wall claim was wrong -- probed 54
    tiles west, effectively unbounded): LONG rectangles (p 28-32, ~385-410m measured)
    that CROSS the game's ~370m length cap, net-z balanced, carrying a >=12z single
    drop, a SECOND >=2z drop run, and a steep segment."""
    cands = generate_p5_candidates()
    assert len(cands) >= 12
    from openrct2_gym.envs.warm_start import ACTION_DROP_Z as DZ, ACTION_CLIMB_Z as CZ
    for c in cands:
        assert 60 <= len(c) <= 116   # cap-crossers + bunny-hop bigs + max-length family
        assert 27 in c and 28 in c                       # steep segment
        banked = [a for a in c if a in (23, 24)]
        if banked:                                       # banked family: legally wrapped
            assert len(banked) >= 4
            assert (16 in c or 15 in c) and (20 in c or 19 in c)   # bank transitions
        assert sum(1 for a in c if a in (29, 30)) % 2 == 0       # S-bends in L/R pairs
        z, chain_peak = 0, 0
        best = run = 0.0
        runs = []
        for a in c:
            z += CZ.get(a, 0) - DZ.get(a, 0)
            if a in CHAIN_ACTIONS:
                chain_peak = max(chain_peak, z)
            d = DZ.get(a, 0)
            if d > 0:
                run += d
                best = max(best, run)
            else:
                if run >= 2:
                    runs.append(run)
                run = 0.0
        if run >= 2:
            runs.append(run)
        assert z == 0                                    # returns to station height
        assert chain_peak >= 12                          # crest feeds a >=12z drop
        assert best >= 12.0                              # the single-drop cap leg
        assert len(runs) >= 2                            # the num-drops cap leg
    # round 3 (Jul-11): a big family with a bunny-hop field must exist -- drop COUNT is
    # the next rating term (flat credit per drop up to 9) and 2-hump exemplars won't
    # teach it; verify some candidate carries >=4 drop runs at >=85 pieces
    def drop_runs(c):
        from openrct2_gym.envs.warm_start import ACTION_DROP_Z as DZ
        n, run = 0, 0.0
        for a in c:
            d = DZ.get(a, 0)
            if d > 0:
                run += d
            else:
                n, run = n + (1 if run >= 2 else 0), 0.0
        return n + (1 if run >= 2 else 0)
    assert any(len(c) >= 85 and drop_runs(c) >= 4 for c in cands)


def test_loop_record_max_single_drop_property():
    """Derived from the action list (like steep_drop_z), so legacy entries get it too:
    a consecutive drop-family run sums; anything else breaks the run."""
    rec = LoopRecord.from_actions([4, 4, 10, 9, 12, 27, 28, 14, 5, 6, 0], "scripted")
    assert rec.max_single_drop_z == pytest.approx(10.0)      # 12,27,28,14 -> 1+4+4+1
    assert LoopRecord.from_actions([0, 3, 3], "scripted").max_single_drop_z == 0.0


def test_library_add_upgrades_excitement_on_dup(tmp_path):
    lib = _lib(tmp_path)
    assert lib.add(LoopRecord.from_actions(FLAT, "harvest", excitement=0.0)) is True
    assert lib.add(LoopRecord.from_actions(FLAT, "harvest", excitement=2.0)) is True
    assert lib.add(LoopRecord.from_actions(FLAT, "harvest", excitement=1.0)) is False
    assert lib.add(LoopRecord.from_actions(FLAT, "harvest", excitement=2.0)) is False
    assert LoopLibrary(lib.path)._records[tuple(FLAT)].excitement == pytest.approx(2.0)


def test_harvest_carries_measured_excitement_and_runs_post_test(monkeypatch):
    """The harvested record carries the MEASURED excitement (CompletingAPI serves 8.0),
    and the harvest call happens after the ride-test verdict is settled."""
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    seen = {}
    orig = OpenRCT2Env._harvest_completed_loop

    def spy(self, excitement=0.0):
        seen['test_ok_at_harvest'] = self._last_test_ok
        seen['excitement'] = excitement
        return orig(self, excitement=excitement)

    monkeypatch.setattr(OpenRCT2Env, "_harvest_completed_loop", spy)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = False
    env.reset()
    for _ in range(12):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    assert terminated
    assert seen['test_ok_at_harvest'] is True
    assert seen['excitement'] == pytest.approx(8.0)
    lib = LoopLibrary(OpenRCT2Env._LOOP_LIBRARY_PATH)
    recs = list(lib._records.values())
    assert len(recs) == 1 and recs[0].excitement == pytest.approx(8.0)


def test_harvest_untested_completion_tags_zero(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reset()
    for _ in range(12):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    assert terminated
    recs = list(LoopLibrary(OpenRCT2Env._LOOP_LIBRARY_PATH)._records.values())
    assert len(recs) == 1 and recs[0].excitement == 0.0


def test_harvest_cap_follows_phase_budget(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    base = OpenRCT2Env(verbose=0)
    wrapper = ImprovedPhasedCurriculumWrapper(base, verbose=0)
    for phase, expect in ((1, 40), (2, 40), (3, 60), (4, 80)):
        wrapper.current_phase = phase
        wrapper._update_phase_settings()
        assert base.harvest_max_len == expect, f"phase {phase}"
    wrapper.current_phase = 5
    wrapper._update_phase_settings()
    assert base.harvest_max_len == wrapper.phase5_current_length


def test_library_maybe_refresh_picks_up_other_workers_appends(tmp_path):
    lib = _lib(tmp_path, [FLAT])
    other = LoopLibrary(lib.path)                               # simulates another worker
    other.add(LoopRecord.from_actions(FLAT_L, source="harvest"))
    assert len(lib) == 1
    for _ in range(3):
        lib.maybe_refresh(every_n_calls=3)
    assert len(lib) == 2                                        # reloaded on the Nth call


# --------------------------------------------------------------- WarmStartAnnealer

def test_annealer_cold_fraction_and_k_range(tmp_path):
    lib = _lib(tmp_path, [FLAT])
    ann = WarmStartAnnealer(k_init=3, p_cold=0.25, rng=random.Random(0))
    plans = [ann.sample_plan(lib, phase=1, max_track_length=40) for _ in range(2000)]
    cold = sum(p.cold for p in plans) / len(plans)
    assert 0.20 <= cold <= 0.30                                 # ~p_cold
    for p in plans:
        if p.cold:
            assert p.prefix == [] and p.k == 0
        else:
            assert 1 <= p.k <= 3
            assert p.prefix == FLAT[:12 - p.k]                  # closing piece never in prefix
            assert p.loop_len == 12


def test_annealer_frontier_biased_k_sampling(tmp_path):
    lib = _lib(tmp_path, [FLAT])
    ann = WarmStartAnnealer(k_init=6, p_cold=0.0, rng=random.Random(1))
    ks = [ann.sample_plan(lib, 1, 40).k for _ in range(2000)]
    at_frontier = sum(k == 6 for k in ks) / len(ks)
    assert at_frontier >= 0.5                                   # k=k_max w.p. 0.5 + uniform share
    assert min(ks) == 1                                         # uniform half still reaches k=1


def test_annealer_empty_pool_forces_cold(tmp_path):
    lib = _lib(tmp_path)                                        # empty library
    ann = WarmStartAnnealer(k_init=3, p_cold=0.0, rng=random.Random(0))
    assert ann.sample_plan(lib, 1, 40).cold is True


def test_annealer_promotes_on_frontier_success(tmp_path):
    ann = WarmStartAnnealer(k_init=3, promote_n=20, promote_rate=0.6, rng=random.Random(0))
    plan = WarmStartPlan(prefix=FLAT[:9], k=3, loop_len=12, cold=False)
    for _ in range(20):
        ann.record_outcome(plan, success=True)
    assert ann.k_max == 5                                       # +2 per promotion
    for _ in range(20):                                         # frontier was cleared: k=4 >= k_max-1
        ann.record_outcome(WarmStartPlan(FLAT[:8], 4, 12, False), success=True)
    assert ann.k_max == 7


def test_annealer_demotes_slowly_with_floor(tmp_path):
    ann = WarmStartAnnealer(k_init=5, promote_n=20, demote_rate=0.15, rng=random.Random(0))
    for _ in range(20):
        ann.record_outcome(WarmStartPlan(FLAT[:7], 5, 12, False), success=False)
    assert ann.k_max == 4                                       # -1 per demotion
    ann2 = WarmStartAnnealer(k_init=3, promote_n=20, rng=random.Random(0))
    for _ in range(20):
        ann2.record_outcome(WarmStartPlan(FLAT[:9], 3, 12, False), success=False)
    assert ann2.k_max == 3                                      # floor at k_init floor (3)


def test_annealer_ignores_cold_and_below_frontier_outcomes(tmp_path):
    ann = WarmStartAnnealer(k_init=5, promote_n=20, rng=random.Random(0))
    for _ in range(50):
        ann.record_outcome(WarmStartPlan([], 0, 0, True), success=True)        # cold: ignored
        ann.record_outcome(WarmStartPlan(FLAT[:10], 2, 12, False), success=True)  # k << frontier
    assert ann.k_max == 5                                       # nothing counted


def test_annealer_mid_band_success_does_not_promote(tmp_path):
    """Between demote (0.15) and promote (0.60) the frontier holds steady."""
    ann = WarmStartAnnealer(k_init=3, promote_n=20, rng=random.Random(0))
    for i in range(40):
        ann.record_outcome(WarmStartPlan(FLAT[:9], 3, 12, False), success=(i % 3 == 0))  # ~33%
    assert ann.k_max == 3


def test_annealer_full_anneal_degenerates_to_cold(tmp_path):
    lib = _lib(tmp_path, [FLAT])
    ann = WarmStartAnnealer(k_init=3, p_cold=0.0, rng=random.Random(2))
    ann.k_max = 12                                              # == loop length
    plans = [ann.sample_plan(lib, 1, 40) for _ in range(500)]
    assert any(p.cold for p in plans)                           # k==L draws collapse to cold
    for p in plans:
        if not p.cold:
            assert p.k < 12                                     # never a full-loop prefix... or suffix
            assert len(p.prefix) >= 1


def test_annealer_frontier_rate_diagnostic():
    """The promotion-relevant number (success at the frontier) must be observable -- the
    first smoke runs were blind to it."""
    ann = WarmStartAnnealer(k_init=3, rng=random.Random(0))
    assert ann.frontier_rate is None                    # empty window
    for success in (True, True, False, True):
        ann.record_outcome(WarmStartPlan(FLAT[:9], 3, 12, False), success)
    assert ann.frontier_rate == pytest.approx(0.75)


def test_annealer_p_cold_schedule_rises_with_k_max():
    ann = WarmStartAnnealer(k_init=3, p_cold=0.25, rng=random.Random(0))
    assert ann.p_cold == pytest.approx(0.25)
    ann.k_max = 8
    assert ann.p_cold == pytest.approx(0.35)
    ann.k_max = 16
    assert ann.p_cold == pytest.approx(0.50)


def test_annealer_phase_change_reinitializes():
    ann = WarmStartAnnealer(k_init=3, rng=random.Random(0))
    ann.k_max = 9
    ann.record_outcome(WarmStartPlan(FLAT[:4], 8, 12, False), success=True)
    ann.on_phase_change(2)
    assert ann.k_max == 3                                       # new skill (hill loops) restarts
    for _ in range(19):                                         # old frontier entries were cleared
        ann.record_outcome(WarmStartPlan(FLAT[:9], 3, 12, False), success=True)
    assert ann.k_max == 3                                       # 19 + 1 stale would have promoted


# ------------------------------------------------------------- candidate templates

def test_generate_candidates_matches_verified_family():
    cands = generate_candidates()
    assert FLAT[:11] in [c[:11] for c in cands] or FLAT[:-1] in [c[:len(FLAT) - 1] for c in cands] \
        or any(c == FLAT[:len(c)] for c in cands)               # the proven p=0 right template
    for c in cands:
        turns = [a for a in c if a in (3, 4)]
        assert len(turns) == 4 and len(set(turns)) == 1         # 4 same-handed 3-tile turns
        p = 0
        while p < len(c) and c[p] == 0:
            p += 1
        assert c[p:p + 2] == [c[p]] * 2                         # leading straights then a U-turn pair
        assert len(c) == 11 + 2 * p                             # b = 7 + p geometry (live-verified)


def test_generate_hill_candidates_carry_balanced_chain_climb():
    cands = generate_hill_candidates()
    assert cands
    saw_three_chain = False
    for c in cands:
        assert [a for a in c if a in (3, 4)].__len__() == 4     # same racetrack skeleton
        i = c.index(10)
        n = 0
        while c[i + 1 + n] == 9:                                # chain climb: 10, 9{n}, 13
            n += 1
        assert c[i + 1 + n] == 13 and n in (1, 2)
        j = c.index(12)
        assert c[j:j + n + 2] == [12] + [6] * n + [14]          # mirrored descent -> net z 0
        assert j > i                                            # climb before descent
        rec = LoopRecord.from_actions(c, source="scripted")
        assert rec.chain_count == n + 1                         # feeds the phase-2 pool filter
        saw_three_chain = saw_three_chain or rec.chain_count >= 3
    assert saw_three_chain                                      # stage 2.3 needs 3-chain demos


def test_library_pool_stage23_prefers_three_chain_hills(tmp_path):
    two_chain = [4, 4, 10, 9, 13, 12, 6, 14, 0, 4, 4, 0]
    three_chain = [0, 4, 4, 10, 9, 9, 13, 12, 6, 6, 14, 4, 4, 0]
    lib = _lib(tmp_path, [FLAT, two_chain, three_chain])
    best = lib.pool(phase=2, max_len=40, min_chains=3)
    assert [r.actions for r in best] == [tuple(three_chain)]
    only_two = _lib(tmp_path.joinpath("two"), [FLAT, two_chain])
    fallback = only_two.pool(phase=2, max_len=40, min_chains=3)
    assert [r.actions for r in fallback] == [tuple(two_chain)]  # degrade to any-hill


# ------------------------------------------------- env-side warm-start replay (FakeAPI)

def _fake_env(monkeypatch, api_cls=FakeAPI):
    monkeypatch.setattr(oe_mod, "APIController", api_cls)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    return env


def test_warm_start_replays_prefix_through_same_bookkeeping(monkeypatch):
    """Prefix pieces must be indistinguishable from agent placements to every consumer:
    history (obs buffer, chain gates, energy), track budget, chain counter -- and Phi must
    be seeded AFTER the prefix so the first agent step gets no shaping windfall."""
    env = _fake_env(monkeypatch)
    env.warm_start_actions = [0, 9, 0]
    obs, _ = env.reset()
    assert len(env.track_builder.history) == 3
    assert env.track_length == 3
    assert env.chain_lift_count == 1
    assert len(env.height_history) == 3
    assert list(obs["build_history_tokens"][:4]) == [1, 10, 1, 0]   # action+1 tokens, then PAD
    assert env.steps == 0                                           # not agent steps
    assert env.episode_rewards == []                                # no reward emitted
    assert env._phi_prev == pytest.approx(env._potential(env.reward_params))
    assert env._warm_prefix_len == 3 and env._warm_cold is False
    assert env.loop_completed is False


def test_warm_start_consumes_track_budget(monkeypatch):
    env = _fake_env(monkeypatch)
    env.max_track_length = 5
    env.warm_start_actions = [0, 0, 0]
    env.reset()
    _, _, _, truncated, _ = env.step(0)                             # track 4 of 5
    assert not truncated
    _, _, _, truncated, _ = env.step(0)                             # track 5 -> budget spent
    assert truncated


class FlakyPrefixAPI(FakeAPI):
    """Fails the 2nd non-station placement (a prefix piece), then recovers."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._agent_pieces = 0

    def place_track_piece(self, x, y, z, direction, track_type, has_chain=False):
        if track_type not in (1, 2, 3):
            self._agent_pieces += 1
            if self._agent_pieces == 2:
                return {"success": False, "error": "collision"}
        return super().place_track_piece(x, y, z, direction, track_type, has_chain)


def test_warm_start_prefix_failure_aborts_and_continues(monkeypatch):
    env = _fake_env(monkeypatch, FlakyPrefixAPI)
    env.warm_start_actions = [0, 0, 0]
    obs, _ = env.reset()                                            # piece 2 fails -> abort
    assert env._warm_prefix_len == 1                                # kept what placed
    assert len(env.track_builder.history) == 1
    assert env.observation_space.contains(obs)
    _, reward, terminated, truncated, _ = env.step(0)               # episode continues fine
    assert np.isfinite(reward) and not terminated


def test_warm_start_accidental_completion_reopens_circuit(monkeypatch):
    """A prefix must NEVER hand the agent a completed episode: if a prefix piece closes the
    circuit (geometry drift), the env removes it and aborts the prefix."""
    env = _fake_env(monkeypatch, CompletingAPI)                     # completes on 2nd agent piece
    env.warm_start_actions = [0, 0, 0]
    env.reset()
    assert env.loop_completed is False
    assert env._warm_prefix_len == 1                                # completing piece was removed
    assert len(env.track_builder.history) == 1
    assert not env.track_builder.history[-1].get("is_complete")


def test_warm_start_actions_are_one_shot(monkeypatch):
    env = _fake_env(monkeypatch)
    env.warm_start_actions = [0, 0]
    env.reset()
    assert env.warm_start_actions is None                           # consumed
    env.reset()                                                     # plain reset -> cold
    assert len(env.track_builder.history) == 0
    assert env._warm_prefix_len == 0 and env._warm_cold is True


def test_step_info_carries_cold_flag_and_prefix_len(monkeypatch):
    env = _fake_env(monkeypatch)
    env.reset()
    _, _, _, _, info = env.step(0)
    assert info['cold_start'] is True and info['warm_prefix_len'] == 0
    env.warm_start_actions = [0, 0, 0]
    env.reset()
    _, _, _, _, info = env.step(0)
    assert info['cold_start'] is False and info['warm_prefix_len'] == 3


def test_harvest_writes_completed_loop_and_dedups(monkeypatch):
    env = _fake_env(monkeypatch, CompletingAPI)
    env.reset()
    for _ in range(4):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    assert terminated and env.loop_completed
    lib = LoopLibrary(OpenRCT2Env._LOOP_LIBRARY_PATH)
    assert len(lib) == 1
    (rec,) = lib.pool(phase=1, max_len=40)
    assert rec.actions == (0, 0)                                    # the two placed agent pieces
    assert rec.source == "harvest"
    env.reset()                                                     # same loop again -> dedup
    for _ in range(4):
        _, _, terminated, _, _ = env.step(0)
        if terminated:
            break
    assert len(LoopLibrary(OpenRCT2Env._LOOP_LIBRARY_PATH)) == 1


def test_harvest_skips_incomplete_episodes(monkeypatch):
    env = _fake_env(monkeypatch)                                    # FakeAPI never completes
    env.max_track_length = 3
    env.reset()
    truncated = False
    while not truncated:
        _, _, _, truncated, _ = env.step(0)
    assert not os.path.exists(OpenRCT2Env._LOOP_LIBRARY_PATH)


def test_scaffolded_episode_budget_is_tight(monkeypatch):
    """A scaffolded episode exists to practice the LAST k decisions: without a tight budget a
    failed dock attempt wanders ~100 steps of noise (observed in the first smoke run: ep_len
    ~100 at k<=3, scaffold learning drowned). Track budget = prefix + k + slack; step budget
    proportional. Cold episodes keep the full phase budget."""
    env = _fake_env(monkeypatch)
    env.max_track_length = 40
    env.warm_start_actions = [0] * 9
    env.warm_start_suffix_k = 3
    env.reset()
    cap = 9 + 3 + oe_mod.OpenRCT2Env.WARM_TRACK_SLACK
    for _ in range(cap - 9 - 1):                                    # place up to cap-1
        _, _, _, truncated, _ = env.step(0)
        assert not truncated
    _, _, _, truncated, _ = env.step(0)                             # reaches the track cap
    assert truncated


def test_scaffolded_episode_step_cap_stops_wandering(monkeypatch):
    """Steps without track growth (failures, place/remove churn) must also be bounded in a
    scaffolded episode, or the wander just moves from pieces to steps."""
    env = _fake_env(monkeypatch)
    env.warm_start_actions = [0, 0]
    env.warm_start_suffix_k = 1
    env.reset()
    step_cap = oe_mod.OpenRCT2Env.WARM_STEP_FACTOR * (1 + oe_mod.OpenRCT2Env.WARM_TRACK_SLACK)
    truncated, steps = False, 0
    while not truncated and steps < step_cap + 5:
        action = 0 if steps % 2 == 0 else 31          # place/remove churn: track never grows
        _, _, _, truncated, _ = env.step(action)
        steps += 1
    # +1: _is_trunkated checks before steps increments (same convention as max_steps)
    assert truncated and steps <= step_cap + 1        # ended by the step cap, not the track cap


def test_cold_episode_keeps_full_budget(monkeypatch):
    env = _fake_env(monkeypatch)
    env.max_track_length = 40
    env.reset()                                                     # cold: no warm_start_actions
    for _ in range(39):
        _, _, _, truncated, _ = env.step(0)
        assert not truncated
    _, _, _, truncated, _ = env.step(0)                             # full 40-piece budget
    assert truncated


def test_warm_suffix_k_is_one_shot(monkeypatch):
    env = _fake_env(monkeypatch)
    env.warm_start_actions = [0, 0]
    env.warm_start_suffix_k = 2
    env.reset()
    assert env.warm_start_suffix_k is None
    env.reset()                                                     # cold reset -> caps cleared
    for _ in range(20):                                             # far beyond any stale cap
        _, _, _, truncated, _ = env.step(0)
        assert not truncated


class ClimbAPI(FakeAPI):
    """FakeAPI with real z geometry: ascending track types climb by their span, descending
    types are placed at base z and end there (mirroring the live plugin contract)."""
    _DZ = {6: 1, 4: 2, 9: 1, 8: 4, 5: 8, 7: 4}   # track TYPE -> z gain (descents: dz 0 from base)

    def place_track_piece(self, x, y, z, direction, track_type, has_chain=False):
        resp = super().place_track_piece(x, y, z, direction, track_type, has_chain)
        resp["payload"]["nextEndpoint"]["z"] = z + self._DZ.get(track_type, 0)
        return resp


def test_prefix_satisfied_milestones_are_prelatched(monkeypatch):
    """Once-per-episode climb milestones must pay AGENT work only: a hill prefix that already
    summited AND returned banks R_summit+R_roundtrip (+120 at stage 2.1) on the agent's first
    step otherwise (observed live: scaffolded completion paid 1326 = 1000+250+40+80-Phi)."""
    from openrct2_gym.envs.improved_phased_curriculum_wrapper import ImprovedPhasedCurriculumWrapper
    env = _fake_env(monkeypatch, ClimbAPI)
    env.reward_params = ImprovedPhasedCurriculumWrapper._phase_reward_params(2, phase2_stage=1)
    env.warm_start_actions = [10, 9, 13, 12, 6, 14]     # climb + full descent -> back at z=14
    env.warm_start_suffix_k = 2
    env.reset()
    assert env._chain_max_gain() >= env.reward_params.roundtrip_gain
    assert env.current_position[2] == 14
    assert env._summit_awarded is True                  # prefix climbed -> not the agent's summit
    assert env._roundtrip_awarded is True               # prefix returned -> not the agent's return


def test_prefix_summit_leaves_return_earnable(monkeypatch):
    """If the prefix ends AT the summit, the descent is the agent's work: summit pre-latched,
    round-trip still earnable when the agent brings the head home."""
    from openrct2_gym.envs.improved_phased_curriculum_wrapper import ImprovedPhasedCurriculumWrapper
    env = _fake_env(monkeypatch, ClimbAPI)
    env.reward_params = ImprovedPhasedCurriculumWrapper._phase_reward_params(2, phase2_stage=1)
    env.warm_start_actions = [10, 9, 13]                # climb only -> head at z=18
    env.warm_start_suffix_k = 4
    env.reset()
    assert env._summit_awarded is True
    assert env._roundtrip_awarded is False
    p = env.reward_params
    _, r12, *_ = env.step(12)                           # z 18 -> 17: still elevated
    assert env._roundtrip_awarded is False
    _, r6, *_ = env.step(6)                             # z 17 -> 15: within tolerance but AT the
    assert env._roundtrip_awarded is False              # gain-1 climb bar -> not yet a return
    _, r14, *_ = env.step(14)                           # z 15 -> 14: the agent's actual return
    assert env._roundtrip_awarded is True
    assert r14 > p.R_roundtrip / 2                      # the milestone actually paid the agent


def test_flat_prefix_latches_nothing(monkeypatch):
    env = _fake_env(monkeypatch, ClimbAPI)
    env.warm_start_actions = [0, 0, 0]
    env.warm_start_suffix_k = 2
    env.reset()
    assert env._summit_awarded is False and env._roundtrip_awarded is False


def test_aborted_prefix_gets_full_budget_and_flag(monkeypatch):
    """A mid-replay placement failure leaves the head far from closure: the tight budget
    would make the episode geometrically impossible, and its outcome would poison the
    annealer frontier. Aborted episodes run with the full phase budget and are flagged."""
    env = _fake_env(monkeypatch, FlakyPrefixAPI)        # fails prefix piece 2
    env.max_track_length = 40
    env.warm_start_actions = [0, 0, 0]
    env.warm_start_suffix_k = 1
    env.reset()
    assert env._warm_aborted is True
    assert env._warm_track_cap is None and env._warm_step_cap is None
    _, _, _, _, info = env.step(0)
    assert info['warm_aborted'] is True
    assert info['cold_start'] is False                  # still NOT a cold episode (partial prefix)


def test_clean_prefix_is_not_flagged_aborted(monkeypatch):
    env = _fake_env(monkeypatch)
    env.warm_start_actions = [0, 0]
    env.warm_start_suffix_k = 1
    env.reset()
    assert env._warm_aborted is False
    _, _, _, _, info = env.step(0)
    assert info['warm_aborted'] is False


def test_harvest_skips_overlong_loops(monkeypatch):
    """Long meander completions (30-40 pieces) are legal but junk scaffolds: over a long
    run they would swamp the pool and slow every prefix replay. Harvest keeps loops
    <= HARVEST_MAX_LEN only."""
    env = _fake_env(monkeypatch, CompletingAPI)
    env.api_controller.complete_after = OpenRCT2Env.HARVEST_MAX_LEN + 3
    env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(0)
    assert terminated and env.loop_completed
    assert len(env.track_builder.history) > OpenRCT2Env.HARVEST_MAX_LEN
    assert not os.path.exists(OpenRCT2Env._LOOP_LIBRARY_PATH)


def test_library_caps_flat_and_hill_classes_separately(tmp_path):
    """A single global cap is a first-come-forever lock: Phase-1 flat harvests would fill
    every slot and silently refuse Phase-2's first hill discoveries -- exactly the records
    the phase-2 pool exists for. Flat and hill loops are capped independently."""
    lib = LoopLibrary(str(tmp_path / "cap.jsonl"))
    # fixed-length unique tails (digits 0-4: no chain actions, stays below BIG_LEN)
    def digits(i):
        return [i // 125 % 5, i // 25 % 5, i // 5 % 5, i % 5]
    for i in range(LoopLibrary.MAX_RECORDS_PER_CLASS):                  # fill the small-FLAT class
        assert lib.add(LoopRecord.from_actions([4, 4] + digits(i), "harvest"))
    assert lib.add(LoopRecord.from_actions([3, 3, 0, 1, 2], "harvest")) is False   # flat full
    hill = LoopRecord.from_actions([4, 4, 10, 9, 13, 12, 6, 14, 0], "harvest")
    assert lib.add(hill) is True                                        # hill class still open
    assert len(lib) == LoopLibrary.MAX_RECORDS_PER_CLASS + 1
    for i in range(LoopLibrary.MAX_RECORDS_PER_CLASS - 1):              # now fill the small-HILL class
        assert lib.add(LoopRecord.from_actions([4, 4, 10, 9, 13, 12, 6, 14] + digits(i),
                                               "harvest")) is True
    assert lib.add(LoopRecord.from_actions([3, 3, 10, 9, 13, 12, 6, 14, 1], "harvest")) is False


def test_create_env_threads_library_path_to_env_harvest(monkeypatch, tmp_path):
    """--loop-library must redirect BOTH sides: the wrapper's read pool AND the env's
    harvest destination. (Split-brain bug: the wrapper read the custom path while the env
    harvested to the default -- the run's discoveries leaked into a file no one read.)"""
    import train as T
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    custom = str(tmp_path / "custom_lib.jsonl")
    env = T.create_curriculum_masked_env(8080, verbose=0, loop_library_path=custom)
    base = env.env.env  # ActionMasker -> Monitor -> curriculum wrapper
    assert base._loop_library.path == custom                      # read side
    assert OpenRCT2Env._LOOP_LIBRARY_PATH == custom               # harvest (write) side
    env.close()


class FirstPieceFailsAPI(FakeAPI):
    """Fails the very first non-station placement (a first-prefix-piece infra hiccup)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._agent_pieces = 0

    def place_track_piece(self, x, y, z, direction, track_type, has_chain=False):
        if track_type not in (1, 2, 3):
            self._agent_pieces += 1
            if self._agent_pieces == 1:
                return {"success": False, "error": "hiccup"}
        return super().place_track_piece(x, y, z, direction, track_type, has_chain)


def test_zero_length_aborted_prefix_counts_as_cold(monkeypatch):
    """A prefix that aborts before placing ANYTHING is bit-identical to a cold episode
    (empty track, full budget) -- classifying it as scaffolded would silently starve the
    cold gate windows under recurring infra hiccups."""
    env = _fake_env(monkeypatch, FirstPieceFailsAPI)
    env.warm_start_actions = [0, 0, 0]
    env.warm_start_suffix_k = 1
    env.reset()
    assert env._warm_prefix_len == 0 and env._warm_aborted is True
    assert env._warm_cold is True                                 # nothing scaffolded happened
    assert env._warm_track_cap is None and env._warm_step_cap is None


# ------------------------------------------------ wrapper wiring (owns the annealer)

from openrct2_gym.envs.improved_phased_curriculum_wrapper import ImprovedPhasedCurriculumWrapper


def _wrapped(monkeypatch, tmp_path, api_cls=FakeAPI, seed_loops=(FLAT,), **kw):
    monkeypatch.setattr(oe_mod, "APIController", api_cls)
    lib_path = str(tmp_path / "wrapper_lib.jsonl")
    lib = LoopLibrary(lib_path)
    for seq in seed_loops:
        lib.add(LoopRecord.from_actions(seq, source="scripted"))
    base = OpenRCT2Env(verbose=0)
    wrapper = ImprovedPhasedCurriculumWrapper(
        base, verbose=0, loop_library_path=lib_path, **kw)
    wrapper._annealer._rng = random.Random(0)                       # deterministic sampling
    return wrapper, base


def _run_episode(wrapper, action=0, max_steps=60):
    wrapper.reset()
    for _ in range(max_steps):
        _, _, terminated, truncated, info = wrapper.step(action)
        if terminated or truncated:
            return info
    raise AssertionError("episode did not end")


def test_wrapper_stages_prefix_before_env_reset(monkeypatch, tmp_path):
    wrapper, base = _wrapped(monkeypatch, tmp_path, p_cold=0.0)     # always scaffolded
    wrapper.reset()
    assert wrapper._current_plan.cold is False
    assert 1 <= wrapper._current_plan.k <= 3
    assert len(base.track_builder.history) == 12 - wrapper._current_plan.k
    assert base.warm_start_actions is None                          # consumed by env.reset


def test_wrapper_cold_when_disabled_or_pool_empty(monkeypatch, tmp_path):
    off, base_off = _wrapped(monkeypatch, tmp_path, warm_start_enabled=False, p_cold=0.0)
    off.reset()
    assert off._current_plan.cold is True and len(base_off.track_builder.history) == 0
    empty, base_e = _wrapped(monkeypatch, tmp_path.joinpath("e"), seed_loops=(), p_cold=0.0)
    empty.reset()
    assert empty._current_plan.cold is True and len(base_e.track_builder.history) == 0


def test_wrapper_evaluation_mode_forces_cold(monkeypatch, tmp_path):
    wrapper, base = _wrapped(monkeypatch, tmp_path, p_cold=0.0)
    with wrapper.evaluation_mode():
        wrapper.reset()
        assert wrapper._current_plan.cold is True                   # eval measures the true task
        assert len(base.track_builder.history) == 0
    wrapper.reset()
    assert wrapper._current_plan.cold is False                      # training resumes scaffolded


def test_wrapper_warm_starts_all_phases_with_p5_exc_ratchet(monkeypatch, tmp_path):
    """The scaffold covers every discovery cliff, now INCLUDING phase 5 (Jul-9: the
    quality plateau was a discovery problem too). P3/P4 request their gate-matching
    criteria; P5 requests exemplar-shaped loops with a self-ratcheting excitement bar
    (0.8 x the best tagged excitement fitting the budget; 0 on a legacy-only pool)."""
    wrapper, base = _wrapped(monkeypatch, tmp_path, p_cold=0.0)
    seen = []
    orig = wrapper._annealer.sample_plan

    def spy(library, phase, max_len, **kw):
        seen.append((phase, kw))
        return orig(library, phase, max_len, **kw)

    wrapper._annealer.sample_plan = spy
    wrapper.current_phase = 3
    wrapper._update_phase_settings()
    wrapper.reset()
    assert wrapper._current_plan.cold is False                      # P3 scaffolds now
    assert seen[-1] == (3, {'min_chains': 2, 'min_len': 20, 'min_drop_z': 4,
                            'min_steep_z': 0, 'min_single_drop_z': 0,
                            'min_excitement': 0.0})
    wrapper.current_phase = 4
    wrapper._update_phase_settings()
    wrapper.reset()
    # Jul-8: P4 criteria raised to the gate itself (len 40, steep 8) so scaffold
    # prefixes are qualifying-shaped instead of recycled P3 material.
    assert seen[-1] == (4, {'min_chains': 3, 'min_len': 40, 'min_drop_z': 8,
                            'min_steep_z': 8, 'min_single_drop_z': 0,
                            'min_excitement': 0.0})
    wrapper.current_phase = 5
    wrapper._update_phase_settings()
    wrapper.reset()
    assert wrapper._current_plan.cold is False                      # P5 scaffolds now
    phase, kw = seen[-1]
    assert phase == 5
    assert (kw['min_chains'], kw['min_len'], kw['min_drop_z'],
            kw['min_single_drop_z']) == (1, 40, 12, 12)
    assert kw['min_excitement'] == 0.0                              # legacy-only pool -> bar 0
    # the ratchet: a tagged exemplar raises the bar to 0.8 x best-within-budget
    wrapper._loop_library.add(
        LoopRecord.from_actions(BIG_EXCITING, "harvest", excitement=5.0))
    wrapper.reset()
    assert seen[-1][1]['min_excitement'] == pytest.approx(4.0)
    # phases past the curriculum build cold (guard only -- phase 6 has no full machinery)
    wrapper.current_phase = 6
    assert wrapper._sample_warm_start().cold is True


def test_phase_gate_counts_only_cold_episodes(monkeypatch, tmp_path):
    """THE gate invariant: 50+ scaffolded completions must not advance Phase 1; cold
    completions at >= threshold must."""
    wrapper, base = _wrapped(monkeypatch, tmp_path, api_cls=CompletingAPI, p_cold=0.0)
    for _ in range(60):                                             # all scaffolded successes
        info = _run_episode(wrapper)
        assert info['loop_completed'] is True
    assert len(wrapper.episode_results) == 0                        # cold-only window untouched
    wrapper.reset()                                                 # advancement check runs here
    assert wrapper.current_phase == 1                               # no advance on scaffolds
    assert len(wrapper.scaffold_results) > 0

    wrapper._annealer.base_p_cold = 1.0                             # now force cold episodes
    for _ in range(60):
        info = _run_episode(wrapper)
        assert info['cold_start'] is True
    wrapper.reset()                                                 # cold successes -> gate opens
    assert wrapper.current_phase == 2


def test_wrapper_records_outcomes_into_annealer(monkeypatch, tmp_path):
    wrapper, base = _wrapped(monkeypatch, tmp_path, api_cls=CompletingAPI, p_cold=0.0)
    # Completion must come from the AGENT's suffix, not a prefix piece (which would be an
    # accidental-closure abort): raise the fake's completion point past any prefix length.
    base.api_controller.complete_after = 13
    calls = []
    wrapper._annealer.record_outcome = lambda plan, success: calls.append((plan, success))
    _run_episode(wrapper)
    assert base._warm_aborted is False
    assert len(calls) == 1
    plan, success = calls[0]
    assert plan is wrapper._current_plan and success is True


def test_wrapper_skips_annealer_recording_for_aborted_prefix(monkeypatch, tmp_path):
    """An aborted prefix is an infrastructure event, not an agent outcome: it must not
    demote the frontier."""
    wrapper, base = _wrapped(monkeypatch, tmp_path, api_cls=FlakyPrefixAPI, p_cold=0.0)
    calls = []
    wrapper._annealer.record_outcome = lambda plan, success: calls.append((plan, success))
    _run_episode(wrapper)
    assert base._warm_aborted is True                   # the prefix did abort in this episode
    assert calls == []                                  # ...and the annealer never heard of it


def test_wrapper_info_exposes_warm_start_diagnostics(monkeypatch, tmp_path):
    wrapper, base = _wrapped(monkeypatch, tmp_path, api_cls=CompletingAPI, p_cold=0.0)
    info = _run_episode(wrapper)
    for key in ('warm_k', 'warm_k_max', 'cold_success_rate', 'scaffold_success_rate',
                'cold_fraction', 'loop_library_size'):
        assert key in info, key
    assert info['warm_k'] == wrapper._current_plan.k
    assert info['warm_k_max'] == wrapper._annealer.k_max
    assert info['loop_library_size'] >= 1
    if wrapper._annealer.frontier_rate is not None:
        assert info['warm_frontier_rate'] == pytest.approx(wrapper._annealer.frontier_rate)
    assert info['cold_fraction'] == pytest.approx(0.0)              # p_cold=0 -> all scaffolded
    assert info['scaffold_success_rate'] == pytest.approx(1.0)


def test_wrapper_phase_change_reinitializes_annealer(monkeypatch, tmp_path):
    wrapper, base = _wrapped(monkeypatch, tmp_path, warm_k_init=3)
    wrapper._annealer.k_max = 9
    wrapper._advance_to_phase(2)
    assert wrapper._annealer.k_max == 3
    wrapper._annealer.k_max = 7
    wrapper._advance_phase2_stage(2, qualified_rate=0.4)
    assert wrapper._annealer.k_max == 3                             # sub-stage = new gate, re-anneal


def test_wrapper_default_library_path_follows_env_class_attr(monkeypatch, tmp_path):
    """loop_library_path=None falls back to OpenRCT2Env._LOOP_LIBRARY_PATH at construction,
    so test fixtures that isolate the env's harvest path isolate the wrapper too."""
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    base = OpenRCT2Env(verbose=0)
    wrapper = ImprovedPhasedCurriculumWrapper(base, verbose=0)
    assert wrapper._loop_library.path == OpenRCT2Env._LOOP_LIBRARY_PATH


def test_get_phase_stats_includes_warm_start_state(monkeypatch, tmp_path):
    wrapper, base = _wrapped(monkeypatch, tmp_path)
    stats = wrapper.get_phase_stats()
    assert 'warm_k_max' in stats and 'loop_library_size' in stats
    assert stats['warm_k_max'] == wrapper._annealer.k_max
    assert stats['loop_library_size'] == 1


# ------------------------------------------------ big-loop scaffolds (P3/P4 redesign)

from openrct2_gym.envs.warm_start import (
    ACTION_DROP_Z, ACTION_CLIMB_Z, generate_big_candidates,
)


def test_loop_record_computes_drop_z():
    """drop_z is static per-action geometry (verified live via the offset probes), so the
    pool can prefer real-drop loops without replaying them."""
    assert LoopRecord.from_actions(FLAT, "scripted").drop_z == 0
    assert LoopRecord.from_actions(HILL, "scripted").drop_z == 4          # 12+6+14 -> 1+2+1
    steep = [4, 4, 10, 9, 9, 9, 9, 13, 12, 27, 28, 14, 4, 4, 0]
    assert LoopRecord.from_actions(steep, "scripted").drop_z == 10        # 1+4+4+1


def test_library_caps_big_loops_as_their_own_class(tmp_path):
    """Four cap classes (chain x big): P1/P2 mini-loop floods must not evict or lock out
    the big P3/P4 scaffold material."""
    lib = LoopLibrary(str(tmp_path / "cap4.jsonl"))
    def digits(i):
        return [i // 125 % 5, i // 25 % 5, i // 5 % 5, i % 5]
    for i in range(LoopLibrary.MAX_RECORDS_PER_CLASS):                    # fill small-flat
        assert lib.add(LoopRecord.from_actions([4, 4] + digits(i), "harvest"))
    assert lib.add(LoopRecord.from_actions([3, 3, 0, 1, 2], "harvest")) is False   # small-flat full
    big_flat = LoopRecord.from_actions([4, 4] + [0] * 28 + [1], "harvest")         # len 31
    assert big_flat.length >= LoopLibrary.BIG_LEN
    assert lib.add(big_flat) is True                                      # big-flat class open
    small_hill = LoopRecord.from_actions(HILL, "harvest")
    assert lib.add(small_hill) is True                                    # small-hill class open


def test_pool_tiers_degrade_gracefully(tmp_path):
    """P3/P4 prefer big real-drop hill loops but must never silently turn the scaffold off:
    tiers degrade all-criteria -> chains -> any-hill -> flats."""
    big_steep = [0, 0, 0, 4, 4, 10, 9, 9, 9, 9, 13, 12, 27, 28, 14, 4, 4] + [0] * 8   # len 25, drop 10
    lib = _lib(tmp_path, [FLAT, HILL, big_steep])
    tier1 = lib.pool(phase=3, max_len=60, min_chains=2, min_len=20, min_drop_z=8)
    assert [r.actions for r in tier1] == [tuple(big_steep)]
    no_big = _lib(tmp_path.joinpath("nb"), [FLAT, HILL])
    tier2 = no_big.pool(phase=3, max_len=60, min_chains=2, min_len=20, min_drop_z=8)
    assert [r.actions for r in tier2] == [tuple(HILL)]                    # chains-only fallback
    flats = _lib(tmp_path.joinpath("f"), [FLAT])
    tier4 = flats.pool(phase=3, max_len=60, min_chains=2, min_len=20, min_drop_z=8)
    assert [r.actions for r in tier4] == [tuple(FLAT)]                    # never empty


def test_generate_big_candidates_are_height_balanced_racetracks():
    cands = generate_big_candidates()
    assert cands
    saw_steep = saw_tall25 = False
    for c in cands:
        turns = [a for a in c if a in (3, 4)]
        assert len(turns) == 4 and len(set(turns)) == 1                   # racetrack skeleton
        climb = sum(ACTION_CLIMB_Z.get(a, 0) for a in c)
        drop = sum(ACTION_DROP_Z.get(a, 0) for a in c)
        assert climb == drop and climb >= 8                               # tall AND net z 0
        rec = LoopRecord.from_actions(c, "scripted")
        assert rec.chain_count >= 4                                       # 10 + 9{3,4}
        saw_steep = saw_steep or any(a in (27, 28) for a in c)
        saw_tall25 = saw_tall25 or (drop >= 8 and not any(a in (8, 27, 28) for a in c))
    assert saw_steep and saw_tall25                                       # both families present
    assert max(len(c) for c in cands) >= 21                               # long skeletons exist


def test_scripted_seeds_bypass_harvest_caps(tmp_path):
    """The per-class caps bound the HARVEST flood; curated scripted seeds are finite by
    construction and are the pool's backbone -- a cap-saturated library must never refuse
    them (observed live: all 36 verified big loops were refused by a full small-hill class)."""
    lib = LoopLibrary(str(tmp_path / "seed.jsonl"))
    def digits(i):
        return [i // 125 % 5, i // 25 % 5, i // 5 % 5, i % 5]
    for i in range(LoopLibrary.MAX_RECORDS_PER_CLASS):          # saturate small-hill via harvests
        assert lib.add(LoopRecord.from_actions([4, 4, 10, 9, 13, 12, 6, 14] + digits(i), "harvest"))
    assert lib.add(LoopRecord.from_actions([3, 3, 10, 9, 13, 12, 6, 14, 1], "harvest")) is False
    seed = LoopRecord.from_actions([4, 4, 10, 9, 9, 9, 9, 13, 12, 27, 28, 14, 4, 4, 0], "scripted")
    assert lib.add(seed) is True                                # curated seed admitted
    assert lib.add(LoopRecord.from_actions([3, 3, 10, 9, 13, 12, 6, 14, 2], "harvest")) is False
