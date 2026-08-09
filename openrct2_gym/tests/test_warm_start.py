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
from openrct2_gym.envs.footprint import classify_family
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
    # The ratchet keys are family-scoped (Task 7 fix): pin the draw to BIG_EXCITING's own
    # family (0) so the bar it gates against is reachable.
    assert classify_family(BIG_EXCITING) == 0
    wrapper._family_rng.choice = lambda seq: 0
    wrapper._loop_library.add(
        LoopRecord.from_actions(BIG_EXCITING, "harvest", excitement=5.0))
    info = _run_episode(wrapper)
    assert info['library_best_excitement'] == pytest.approx(5.0)
    assert info['p5_pool_exc_bar'] == pytest.approx(4.0)


def test_p5_ratchet_keys_scoped_to_the_episodes_family(monkeypatch, tmp_path):
    """Fix pass (Task 7, resolution 5): library_best_excitement / p5_pool_exc_bar are
    diagnostics for the bar that actually gated THIS episode's pool -- _sample_warm_start
    scopes the real gate to the episode's family, so an unscoped diagnostic query would
    report a different (higher) number whenever the drawn family lacks the library's
    cross-family top exemplar. Family 3 (winding) draws its OWN best (3.0), not the
    family-0 oval's 9.0."""
    oval = [4, 0, 4, 0, 4, 0, 4] + [0] * 20
    assert classify_family(oval) == 0
    winding = [4, 4, 3, 3] * 3 + [0] * 20
    assert classify_family(winding) == 3

    wrapper, _ = _wrapped(monkeypatch, tmp_path, api_cls=CompletingAPI, p_cold=0.0)
    wrapper.current_phase = 5
    wrapper._update_phase_settings()
    wrapper._family_rng.choice = lambda seq: 3          # pin this episode's draw
    wrapper._loop_library.add(LoopRecord.from_actions(oval, "harvest", excitement=9.0))
    wrapper._loop_library.add(LoopRecord.from_actions(winding, "harvest", excitement=3.0))
    info = _run_episode(wrapper)
    assert info['target_family'] == 3
    assert info['library_best_excitement'] == pytest.approx(3.0)     # NOT the oval's 9.0
    assert info['p5_pool_exc_bar'] == pytest.approx(2.4)


def test_p5_ratchet_keys_unscoped_in_earlier_phases(monkeypatch, tmp_path):
    """Phases 1-2 must stay bit-identical: PHASE_FAMILIES is empty there, so the ratchet
    keys are never even emitted (unchanged pre-fix behavior) -- verified, not assumed."""
    wrapper, _ = _wrapped(monkeypatch, tmp_path, api_cls=CompletingAPI, p_cold=0.0)
    wrapper.current_phase = 2
    wrapper._update_phase_settings()
    wrapper._loop_library.add(
        LoopRecord.from_actions(BIG_EXCITING, "harvest", excitement=5.0))
    info = _run_episode(wrapper)
    assert 'library_best_excitement' not in info
    assert 'p5_pool_exc_bar' not in info


def test_annealer_p5_promotes_with_larger_step():
    """Jul-12: cold internalization of ~90-piece exemplars at +2/promotion is a
    multi-day grind (measured k 3->17 in ~15h). P5 promotes +4; earlier phases keep
    the proven +2; demotion unchanged."""
    ann = WarmStartAnnealer(k_init=3, promote_n=20, promote_rate=0.6, rng=random.Random(0))
    ann.on_phase_change(5)
    plan = WarmStartPlan(prefix=[0] * 70, k=3, loop_len=73, cold=False)
    for _ in range(20):
        ann.record_outcome(plan, success=True)
    assert ann.k_max == 7                                   # +4 in P5
    ann.on_phase_change(2)
    for _ in range(20):
        ann.record_outcome(WarmStartPlan([0] * 9, ann.k_max - 1, 12, False), success=True)
    assert ann.k_max == 5                                   # back to +2 elsewhere


def test_p5_substage_advance_keeps_frontier(monkeypatch, tmp_path):
    """Jul-15 (reversing Jul-12): each ladder rung RESETTING the anneal discarded
    frontier progress ~4x per run segment while the pool only ever grows -- closure
    skill transfers across budgets, so rung advances now KEEP k."""
    wrapper, _ = _wrapped(monkeypatch, tmp_path, p_cold=0.0)
    wrapper.current_phase = 5
    wrapper._update_phase_settings()
    wrapper._track_stats = True
    wrapper.phase_episode_count = 60
    wrapper.episode_results.extend([True] * 50)
    wrapper._annealer.k_max = 11
    assert wrapper.phase5_current_length < wrapper.phase5_target_length
    assert wrapper._check_phase_advancement() is True                   # rung 80 -> 90
    assert wrapper._annealer.k_max == 11                                # progress kept


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


def test_generate_p6_candidates_wind_both_ways():
    """P6 exemplars must carry the variety gate legs: >=12 turn pieces with handedness
    balance >=2 (canceling jog-pairs wind out and back), plus every P5 cap leg (z-balance,
    steep segment, >=12z single drop, >=2 drop runs)."""
    from openrct2_gym.envs.warm_start import (
        generate_p6_candidates, ACTION_DROP_Z as DZ, ACTION_CLIMB_Z as CZ)
    cands = generate_p6_candidates()
    assert len(cands) >= 8
    for c in cands:
        assert 56 <= len(c) <= 118    # incl. Jul-19 long winders for the topped frontier
        left = sum(1 for a in c if a in (1, 3, 21, 23, 29))
        right = sum(1 for a in c if a in (2, 4, 22, 24, 30))
        assert min(left, right) >= 2
        turns = sum(1 for a in c if a in (1, 2, 3, 4, 21, 22, 23, 24, 29, 30))
        assert turns >= 12
        assert 27 in c and 28 in c
        z = 0
        best = run = 0.0
        runs = 0
        for a in c:
            z += CZ.get(a, 0) - DZ.get(a, 0)
            d = DZ.get(a, 0)
            if d > 0:
                run += d
                best = max(best, run)
            else:
                runs, run = runs + (1 if run >= 2 else 0), 0.0
        runs += 1 if run >= 2 else 0
        assert z == 0
        assert best >= 12.0
        assert runs >= 2


def test_generate_serpentine_candidates_are_serpentine_shaped():
    """Family 4 (serpentine: 14+ turns, 6+ switches -- footprint.FAMILIES) had 63 measured
    exemplars against oval's ~148k: thin enough that P6's winding fix (family 3, two jog
    pairs) does not carry it, and no other generator ever emits the shape. Three/four
    canceling jog-pairs (P6's own construction, extended) push turns/switches past P6's
    12-turn/4-switch ceiling and into the serpentine bands.

    The `straights < 8` guard is the documented failure mode here: if the piece budgets
    were too tight for the extra jogs, every candidate would be filtered out and the
    generator would silently return []/near-empty -- which would pass every test below
    except the explicit non-emptiness assertion, so that assertion comes first."""
    from openrct2_gym.envs.footprint import switch_count
    from openrct2_gym.envs.track_pieces import TURN_ACTIONS
    from openrct2_gym.envs.warm_start import generate_serpentine_candidates

    cands = generate_serpentine_candidates()
    assert len(cands) >= 8, f"generator returned only {len(cands)} candidates"

    for c in cands:
        assert classify_family(c) == 4, c            # every candidate, not "most"

    # Spot-check turns/switches directly against the real bands (footprint.FAMILIES[4]
    # is (14, None, 6, None)) so a future change to the band definitions fails loudly
    # here rather than only inside classify_family.
    sample = cands[0]
    turns = sum(1 for a in sample if a in TURN_ACTIONS)
    switches = switch_count(sample)
    assert turns >= 14
    assert switches >= 6

    lengths = {len(c) for c in cands}
    assert len(lengths) >= 2                          # more than one shape bin


def test_loop_record_turn_and_sbend_properties():
    rec = LoopRecord.from_actions([4, 4, 0, 3, 29, 30, 0, 4, 4], "scripted")
    assert rec.turn_count == 5                               # 4,4,3,4,4 (S-bends excluded)
    assert rec.sbend_count == 2
    assert LoopRecord.from_actions([0, 0, 6], "scripted").turn_count == 0


# --------------------------------------------- family-aware pool (Aug-9 gap fix, task 6)
# A spiral seed must be scaffolded by spiral exemplars, or the seed means nothing during
# the phases where its reward is still off. Note: the brief's original winding fixture
# ([4,4,0,3,3,0,4,4,0,3,3,0]) is 8 turns / 3 switches -- the straights between jog pairs
# drop it out of the 10-13 turn winding band entirely (classify_family returns None).
# [4,4,3,3]*3 is 12 turns / 5 switches, genuinely winding.

def test_loop_record_exposes_its_family():
    oval = LoopRecord.from_actions([4, 0, 4, 0, 4, 0, 4], "scripted")
    winding = LoopRecord.from_actions([4, 4, 3, 3] * 3, "scripted")
    assert oval.family == 0
    assert winding.family == 3


def test_pool_filters_by_requested_family(tmp_path):
    """A spiral seed must be scaffolded by spiral exemplars, or the seed means nothing
    during the phases where its reward is still off."""
    lib = _lib(tmp_path)
    oval = [4, 0, 4, 0, 4, 0, 4] + [0] * 20
    winding = [4, 4, 3, 3] * 3 + [0] * 20
    lib.add(LoopRecord.from_actions(oval, "scripted"))
    lib.add(LoopRecord.from_actions(winding, "scripted"))
    got = lib.pool(phase=1, max_len=120, min_chains=0, family=3)
    assert [r.actions for r in got] == [tuple(winding)]


def test_pool_family_filter_degrades_when_no_exemplar_exists(tmp_path):
    """Thin families (serpentine has 61 archive examples) must not empty the pool --
    the scaffold going silent is worse than an off-family exemplar."""
    lib = _lib(tmp_path)
    lib.add(LoopRecord.from_actions([4, 0, 4, 0, 4, 0, 4] + [0] * 20, "scripted"))
    got = lib.pool(phase=1, max_len=120, min_chains=0, family=4)
    assert got, "empty pool would silently disable the scaffold"


# ---------------------------------------- fix pass: family yields to structure (task 6 review)
# The reviewer's traced case: the 24 seeded steep P4 exemplars are single-handedness
# skeletons (4 turns, 0 switches -> family 0); short chainless cold harvests wind (12
# turns, 5 switches -> family 3). Every fixture below is verified against the real
# classify_family()/LoopRecord properties inline, not asserted by name -- this plan has
# shipped a mis-specified footprint fixture three times already.

def test_pool_family_narrowing_yields_to_structural_criteria(tmp_path):
    """Fix 1: narrowing to an off-family-preferred but structurally weak subset must not
    black out a structurally-qualifying exemplar sitting one filter away. A family=3 draw
    at P4 criteria (min_chains=3, min_len=40, min_drop_z=8, min_steep_z=8) must still
    surface the qualifying steep (family-0) record, not the weak (chainless) winding one."""
    steep_family0 = generate_p4_candidates()[0] + [0] * 3       # padded to the P4 length bar
    assert classify_family(steep_family0) == 0
    steep_rec = LoopRecord.from_actions(steep_family0, "scripted")
    assert (steep_rec.chain_count, steep_rec.length, steep_rec.drop_z, steep_rec.steep_drop_z) \
        == (5, 40, 10.0, 8.0)

    winding_weak = [4, 4, 3, 3] * 3 + [0] * 20                  # no chains, well under the bar
    assert classify_family(winding_weak) == 3
    weak_rec = LoopRecord.from_actions(winding_weak, "scripted")
    assert weak_rec.chain_count == 0

    lib = _lib(tmp_path)
    lib.add(steep_rec)
    lib.add(weak_rec)
    got = lib.pool(phase=4, max_len=120, min_chains=3, min_len=40, min_drop_z=8,
                   min_steep_z=8, family=3)
    assert [r.actions for r in got] == [tuple(steep_family0)]


def test_pool_narrows_to_family_when_it_has_a_qualifying_record(tmp_path):
    """Family preference is not simply disabled by Fix 1 -- when the requested family DOES
    contain a record clearing the phase's structural bar, narrowing still happens and an
    off-family competitor (even a structurally-qualifying one) is excluded."""
    steep_family0 = generate_p4_candidates()[0] + [0] * 3
    assert classify_family(steep_family0) == 0

    winding_qualifying = [4, 4, 3, 3] * 3 + [10, 9, 9, 9, 9, 13] + [0] * 20 + [12, 27, 28, 14]
    assert classify_family(winding_qualifying) == 3
    qual_rec = LoopRecord.from_actions(winding_qualifying, "scripted")
    assert (qual_rec.chain_count, qual_rec.length, qual_rec.drop_z, qual_rec.steep_drop_z) \
        == (5, 42, 10.0, 8.0)

    lib = _lib(tmp_path)
    lib.add(LoopRecord.from_actions(steep_family0, "scripted"))
    lib.add(qual_rec)
    got = lib.pool(phase=4, max_len=120, min_chains=3, min_len=40, min_drop_z=8,
                   min_steep_z=8, family=3)
    assert [r.actions for r in got] == [tuple(winding_qualifying)]


def test_best_excitement_restricts_to_family(tmp_path):
    """Fix 2: the P5/P6 excitement ratchet bar must come from the SAME set the pool draws
    from. An unreachable bar (best exemplar in a different family) would empty the best
    and excited tiers every time for any family lacking the library's top scorer."""
    oval = [4, 0, 4, 0, 4, 0, 4] + [0] * 20
    assert classify_family(oval) == 0
    winding = [4, 4, 3, 3] * 3 + [0] * 20
    assert classify_family(winding) == 3

    lib = _lib(tmp_path)
    lib.add(LoopRecord.from_actions(oval, "harvest", excitement=9.0))
    lib.add(LoopRecord.from_actions(winding, "harvest", excitement=3.0))
    assert lib.best_excitement(120) == pytest.approx(9.0)                 # unrestricted: the oval
    assert lib.best_excitement(120, family=3) == pytest.approx(3.0)       # family 3's own best
    assert lib.best_excitement(120, family=4) == 0.0                      # no rated family-4 record


def test_pool_records_family_narrowing_state_for_diagnostics(tmp_path):
    """Fix 3.1: pool() is a plain library method with no TB access, so the narrowing
    decision must be recorded as instance state the wrapper can read per call -- a
    consumer must be able to tell, per episode, whether a family was requested and
    whether the narrowing applied or fell back."""
    steep_family0 = generate_p4_candidates()[0] + [0] * 3
    winding_weak = [4, 4, 3, 3] * 3 + [0] * 20
    lib = _lib(tmp_path)
    lib.add(LoopRecord.from_actions(steep_family0, "scripted"))
    lib.add(LoopRecord.from_actions(winding_weak, "scripted"))

    lib.pool(phase=4, max_len=120, min_chains=3, min_len=40, min_drop_z=8,
             min_steep_z=8, family=3)                    # requested family can't meet the bar
    assert lib.last_family_requested == 3
    assert lib.last_family_narrowed is False

    lib.pool(phase=1, max_len=120, min_chains=0, family=3)  # trivially-met bar -> narrows
    assert lib.last_family_requested == 3
    assert lib.last_family_narrowed is True

    lib.pool(phase=1, max_len=120)                        # no family requested at all
    assert lib.last_family_requested is None
    assert lib.last_family_narrowed is False


def test_pool_p6_min_turns_and_shape_bin_diversity(tmp_path):
    """P6 pools must sustain MULTIPLE styles: the best tier caps each shape bin so a
    high-excitement rectangle monoculture cannot crowd out winding newcomers."""
    lib = _lib(tmp_path)
    for i in range(5):                                       # rectangle bin, high E
        seq = ([0] * (28 + i) + [4, 4] + [10, 9, 9, 9, 9, 9, 9, 13]
               + [12, 27, 28, 6, 6, 14] + [11, 5, 13] + [12, 6, 14]
               + [0] * (7 + i) + [4, 4])
        assert lib.add(LoopRecord.from_actions(seq, "scripted", excitement=5.8))
    winding = ([0] * 10 + [4, 0, 3] + [0] * 4 + [3, 0, 4] + [0] * 8 + [29, 30]
               + [4, 4] + [10, 9, 9, 9, 9, 9, 9, 13] + [12, 27, 28, 6, 6, 14]
               + [11, 5, 13] + [12, 6, 14] + [0] * 6 + [4, 4])
    assert lib.add(LoopRecord.from_actions(winding, "scripted", excitement=4.9))
    pool = lib.pool(phase=6, max_len=120, min_chains=1, min_len=40, min_drop_z=12,
                    min_single_drop_z=12, min_excitement=4.5, min_turns=4)
    actions = [r.actions for r in pool]
    assert tuple(winding) in actions                         # the winding style survives
    rect_members = [a for a in actions if a != tuple(winding)]
    assert len(rect_members) <= 3                            # monoculture bin capped
    pool_turny = lib.pool(phase=6, max_len=120, min_chains=1, min_len=40, min_drop_z=12,
                          min_single_drop_z=12, min_excitement=4.5, min_turns=8)
    assert [r.actions for r in pool_turny] == [tuple(winding)]


# ------------------------------------------- task 6b: P6's min_turns=8 was unsatisfiable
# for the oval band (<=5 turns by definition -- footprint.py FAMILIES), so an oval seed's
# narrowing could never fire and the phase's own structural "best" tier would surface
# whatever off-family exemplar happened to clear the turns>=8 bar instead. Fix: P6 no
# longer sets min_turns (stays at LoopLibrary.pool's default 0); shape is the family
# filter's job. Every fixture's family below is computed via classify_family(), not
# assumed from its shape or variable name -- this plan has shipped a mis-specified
# footprint fixture three times already.

def test_wrapper_p6_scaffold_narrows_to_oval_family_for_oval_seed(monkeypatch, tmp_path):
    """Regression: with an oval-seeded P6 episode, a library holding (a) an oval record
    clearing every P6 criterion EXCEPT the old turns>=8 bar and (b) an off-family record
    clearing ALL of them including turns>=8, the scaffold must offer the oval record --
    not the off-family one. Before the fix, the oval record fails the old turns check, so
    narrowing to family 0 never fires and the unnarrowed 'best' tier hands over the
    off-family record instead."""
    oval_seq = ([0] * 15 + [4, 4] + [10, 9, 9, 9, 9, 9, 13]
                + [12, 27, 28, 6, 14] + [0] * 10 + [4, 4])
    assert classify_family(oval_seq) == 0
    # Shape suggests "winding" but it classifies as out_and_back (family 2) -- verified,
    # not assumed. It clears the OLD P6 bar in full, including turns>=8 (turn_count 8).
    competitor = ([0] * 10 + [4, 0, 3] + [0] * 4 + [3, 0, 4] + [0] * 8 + [29, 30]
                  + [4, 4] + [10, 9, 9, 9, 9, 9, 9, 13] + [12, 27, 28, 6, 6, 14]
                  + [11, 5, 13] + [12, 6, 14] + [0] * 6 + [4, 4])
    assert classify_family(competitor) == 2

    wrapper, base = _wrapped(monkeypatch, tmp_path, seed_loops=(oval_seq, competitor),
                             p_cold=0.0)
    wrapper.current_phase = 6
    wrapper._update_phase_settings()
    base.target_family = 0                      # this episode's seed: oval

    plan = wrapper._sample_warm_start()

    assert wrapper._loop_library.last_family_requested == 0
    assert wrapper._loop_library.last_family_narrowed is True
    assert plan.cold is False
    assert plan.loop_len == len(oval_seq)


def test_pool_p6_still_enforces_length_and_chain_bars_without_min_turns(tmp_path):
    """Guard against the null fix (dropping min_turns by gutting the whole P6 bar): with
    min_turns no longer part of the P6 criteria, a same-family record failing min_len (or
    min_chains) must still miss the 'best' tier. Two oval records each fail exactly one
    of the OTHER bars; only the fully-qualifying one may come back."""
    good = ([0] * 15 + [4, 4] + [10, 9, 9, 9, 9, 9, 13]
            + [12, 27, 28, 6, 14] + [0] * 10 + [4, 4])
    fail_len = ([0] * 8 + [4, 4] + [10, 9, 9, 9, 9, 9, 13]
                + [12, 27, 28, 6, 14] + [0] * 3 + [4, 4])
    fail_chains = [0] * 15 + [4, 4] + [12, 27, 28, 6, 14] + [0] * 20 + [4, 4]
    for seq in (good, fail_len, fail_chains):
        assert classify_family(seq) == 0
    assert len(good) >= 40 and len(fail_len) < 40 and len(fail_chains) >= 40
    assert LoopRecord.from_actions(fail_chains, "scripted").chain_count == 0
    assert LoopRecord.from_actions(fail_len, "scripted").chain_count >= 1

    lib = _lib(tmp_path, sequences=(good, fail_len, fail_chains))
    pool = lib.pool(phase=6, max_len=120, min_chains=1, min_len=40, min_drop_z=12,
                    min_single_drop_z=12, min_excitement=0.0, min_turns=0, family=0)
    assert [r.actions for r in pool] == [tuple(good)]


def test_wrapper_initial_phase_starts_deep(monkeypatch, tmp_path):
    """Jul-19: a deep-P6 policy CANNOT re-walk Phase 1 (its committed 90+ piece builds
    truncate inside the 40-piece budget; live: cold completion 0.00, active unlearning).
    initial_phase starts the curriculum where the policy actually is: settings applied,
    P5 ladder marked complete, and the annealer in its P5+ (+4) mode."""
    wrapper, base = _wrapped(monkeypatch, tmp_path, initial_phase=6)
    assert wrapper.current_phase == 6
    assert base.max_track_length == wrapper.phase6_max_length == 120
    assert base.skip_ride_testing is False
    assert wrapper.phase5_current_length == wrapper.phase5_target_length
    assert getattr(wrapper._annealer, "k_step", 2) == 4
    # default stays a cold start at phase 1
    w1, _ = _wrapped(monkeypatch, tmp_path.joinpath("d"), p_cold=0.0)
    assert w1.current_phase == 1


def test_wrapper_p6_scaffold_requests_exemplar_shaped_pool(monkeypatch, tmp_path):
    """min_turns is NOT among P6's criteria (task 6b: an 8-turn floor is unsatisfiable
    for the oval band, so shape is the family filter's job, not a turn-count bar)."""
    wrapper, base = _wrapped(monkeypatch, tmp_path, p_cold=0.0)
    seen = []
    orig = wrapper._annealer.sample_plan

    def spy(library, phase, max_len, **kw):
        seen.append((phase, kw))
        return orig(library, phase, max_len, **kw)

    wrapper._annealer.sample_plan = spy
    wrapper.current_phase = 6
    wrapper._update_phase_settings()
    wrapper.reset()
    phase, kw = seen[-1]
    assert phase == 6
    assert (kw['min_chains'], kw['min_len'], kw['min_drop_z'],
            kw['min_single_drop_z'], kw['min_turns']) == (1, 40, 12, 12, 0)
    assert kw['min_excitement'] >= 0.0                       # ratchet still applies
    wrapper.current_phase = 7                                # past-curriculum guard moved
    assert wrapper._sample_warm_start().cold is True


def test_wrapper_requests_the_current_episodes_family_not_the_previous_ones(
        monkeypatch, tmp_path):
    """reset() must draw target_family BEFORE sampling the warm-start pool: sampling
    first (the pre-fix ordering) hands _sample_warm_start last episode's family, so
    the scaffold is always one episode behind what the seed actually asked for."""
    wrapper, base = _wrapped(monkeypatch, tmp_path, p_cold=0.0)
    wrapper.current_phase = 3                                # PHASE_FAMILIES[3] active
    wrapper._update_phase_settings()
    seen = []
    orig = wrapper._annealer.sample_plan

    def spy(library, phase, max_len, **kw):
        seen.append(kw.get('family'))
        return orig(library, phase, max_len, **kw)

    wrapper._annealer.sample_plan = spy
    draws = iter([2, 0, 1])
    wrapper._family_rng.choice = lambda seq: next(draws)
    for _ in range(3):
        wrapper.reset()
        assert seen[-1] == base.target_family


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
    assert rec.source == "harvest_cold"      # bare env = cold episode (Aug-4 tag)
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
    phase3, kw3 = seen[-1]
    kw3 = dict(kw3)                       # copy: kw3 IS the dict captured in `seen`, don't mutate it
    assert kw3.pop('family') in wrapper.PHASE_FAMILIES[3]           # Aug-9: family now armed
    assert (phase3, kw3) == (3, {'min_chains': 2, 'min_len': 20, 'min_drop_z': 4,
                                 'min_steep_z': 0, 'min_single_drop_z': 0,
                                 'min_excitement': 0.0, 'min_turns': 0})
    wrapper.current_phase = 4
    wrapper._update_phase_settings()
    wrapper.reset()
    # Jul-8: P4 criteria raised to the gate itself (len 40, steep 8) so scaffold
    # prefixes are qualifying-shaped instead of recycled P3 material.
    phase4, kw4 = seen[-1]
    kw4 = dict(kw4)                       # copy: same captured-dict-mutation hazard as kw3 above
    assert kw4.pop('family') in wrapper.PHASE_FAMILIES[4]
    assert (phase4, kw4) == (4, {'min_chains': 3, 'min_len': 40, 'min_drop_z': 8,
                                 'min_steep_z': 8, 'min_single_drop_z': 0,
                                 'min_excitement': 0.0, 'min_turns': 0})
    wrapper.current_phase = 5
    wrapper._update_phase_settings()
    wrapper.reset()
    assert wrapper._current_plan.cold is False                      # P5 scaffolds now
    phase, kw = seen[-1]
    assert phase == 5
    assert kw['family'] in wrapper.PHASE_FAMILIES[5]
    assert (kw['min_chains'], kw['min_len'], kw['min_drop_z'],
            kw['min_single_drop_z']) == (1, 40, 12, 12)
    assert kw['min_excitement'] == 0.0                              # legacy-only pool -> bar 0
    # the ratchet: a tagged exemplar raises the bar to 0.8 x best-within-budget.
    # Fix pass (Aug-9 review): the bar is now scoped to the EPISODE'S family (Fix 2),
    # so pin the draw to BIG_EXCITING's own family (0, verified below) -- otherwise the
    # random draw would land off-family 4/5 of the time and see bar 0.0 by construction.
    assert classify_family(BIG_EXCITING) == 0
    wrapper._family_rng.choice = lambda seq: 0
    wrapper._loop_library.add(
        LoopRecord.from_actions(BIG_EXCITING, "harvest", excitement=5.0))
    wrapper.reset()
    assert seen[-1][1]['family'] == 0
    assert seen[-1][1]['min_excitement'] == pytest.approx(4.0)
    # phases past the curriculum build cold (P6 now scaffolds -- see
    # test_wrapper_p6_scaffold_requests_turny_pool; the guard sits at > 6)
    wrapper.current_phase = 7
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
    wrapper._annealer.record_outcome = (
        lambda plan, success, styled=None: calls.append((plan, success)))
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
    wrapper._annealer.record_outcome = (
        lambda plan, success, styled=None: calls.append((plan, success)))
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


def test_wrapper_step_info_exposes_family_narrowing_diagnostics(monkeypatch, tmp_path):
    """Fix 3.1: a consumer must be able to tell, per episode, whether a family was
    requested and whether the pool's narrowing applied or fell back -- following the
    same 'read straight off self._loop_library' pattern as library_best_excitement."""
    family0_p3_qualifier = HILL + [0] * 10             # family 0; chain 2, len 22, drop_z 4
    assert classify_family(family0_p3_qualifier) == 0
    wrapper, base = _wrapped(monkeypatch, tmp_path, api_cls=CompletingAPI, p_cold=0.0,
                             seed_loops=(family0_p3_qualifier,))
    wrapper.current_phase = 3                           # PHASE_FAMILIES[3] active
    wrapper._update_phase_settings()

    wrapper._family_rng.choice = lambda seq: 0          # draw the family the pool can satisfy
    info = _run_episode(wrapper)
    assert info['warm_family_requested'] == 0
    assert info['warm_family_narrowed'] is True

    wrapper._family_rng.choice = lambda seq: 1          # a family with no exemplar in the pool
    info2 = _run_episode(wrapper)
    assert info2['warm_family_requested'] == 1
    assert info2['warm_family_narrowed'] is False

    wrapper.current_phase = 1                            # PHASE_FAMILIES[1] is empty
    wrapper._update_phase_settings()
    info3 = _run_episode(wrapper)
    assert info3['warm_family_requested'] is None


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


def test_annealer_min_prefix_floors_deep_draws(tmp_path):
    """P6 opening-seed mode (Jul-29): with min_prefix set, a deep frontier must NOT
    dissolve into cold draws -- it replays the record's first min_prefix pieces (a
    winding opening seed) and the agent builds everything after. The opening habit is
    the one skill the k-anneal otherwise never practices (k >= len converted to cold,
    so 'replay just the opening' episodes only came from the pool's few longest
    records)."""
    lib = _lib(tmp_path, [FLAT])                                # 12-piece record
    ann = WarmStartAnnealer(k_init=999, p_cold=0.0, rng=random.Random(2))
    ann.min_prefix = 6
    plans = [ann.sample_plan(lib, 1, 40) for _ in range(400)]
    # cold now comes ONLY from the competence-scaled die (0.50 at deep k), never from
    # k >= len dissolution -- without the floor this fixture converts ~half the warm
    # draws too (see the default-dissolution test), pushing cold far above the die.
    cold = sum(p.cold for p in plans) / len(plans)
    assert 0.40 <= cold <= 0.60
    warm = [p for p in plans if not p.cold]
    assert all(p.k <= 12 - 6 for p in warm)                     # k capped at len - min_prefix
    assert all(len(p.prefix) >= 6 for p in warm)                # opening seed always replayed
    assert any(len(p.prefix) == 6 for p in warm)                # the deep draw is reachable


def test_annealer_min_prefix_default_keeps_natural_dissolution(tmp_path):
    """Default min_prefix=0 preserves the pre-P6 contract: k annealed past the loop
    length converts the draw to a genuinely cold episode."""
    lib = _lib(tmp_path, [FLAT])
    ann = WarmStartAnnealer(k_init=999, p_cold=0.0, rng=random.Random(3))
    plans = [ann.sample_plan(lib, 1, 40) for _ in range(50)]
    assert any(p.cold for p in plans)                           # conversion still happens


def test_p6_sets_opening_seed_min_prefix(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    w = ImprovedPhasedCurriculumWrapper(OpenRCT2Env(verbose=0), verbose=0)
    w.current_phase = 6
    w._update_phase_settings()
    assert w._annealer.min_prefix == 6
    w.current_phase = 5
    w._update_phase_settings()
    assert w._annealer.min_prefix == 0


def test_wrapper_warm_k_init_resumes_frontier(monkeypatch, tmp_path):
    """Aug-3: every resume reset k to 3, so each 2M chunk re-proved climb competence
    the checkpoint demonstrably has (0.99+ success at k=91-95 minutes before the
    restart) and only reached the opening-seed regime near its end. warm_k_init
    seeds the frontier where the policy actually is; the annealer's demote path
    still corrects an overshoot honestly."""
    wrapper, _ = _wrapped(monkeypatch, tmp_path, initial_phase=6, warm_k_init=86)
    assert wrapper._annealer.k_max == 86
    # default unchanged: fresh runs still anneal from scratch
    w2, _ = _wrapped(monkeypatch, tmp_path.joinpath("d"))
    assert w2._annealer.k_max == 3


# ------------------------------------------------- prefix annealing (Aug-3, the last gap)
# 570k steps of static-6 opening seeds produced flawless SEEDED winding and zero COLD
# winding: the policy went bimodal by context (seed present -> wind; bare station ->
# rectangle) and both modes pay, so no gradient bridges them. The fix is the campaign's
# own playbook applied to the seed itself: when floor-bound draws succeed reliably,
# shrink min_prefix 6 -> 5 -> ... -> 0, closing the context gap one piece at a time.
# At 0, dissolution returns naturally -- and cold IS the practiced context.

def test_plan_marks_floor_bound_draws(tmp_path):
    lib = _lib(tmp_path, [FLAT])                                # 12-piece record
    ann = WarmStartAnnealer(k_init=999, p_cold=0.0, rng=random.Random(2))
    ann.min_prefix = 6
    plans = [ann.sample_plan(lib, 1, 40) for _ in range(200)]
    floor = [p for p in plans if p.at_floor]
    assert floor and all(p.k == 12 - 6 for p in floor)          # k at the per-record cap
    shallow = [p for p in plans if not p.cold and not p.at_floor]
    assert all(p.k < 12 - 6 for p in shallow)


def test_floor_success_anneals_prefix_down_to_zero(tmp_path):
    ann = WarmStartAnnealer(k_init=999, promote_n=10, promote_rate=0.6,
                            rng=random.Random(0))
    ann.min_prefix = 2
    floor_plan = WarmStartPlan(prefix=FLAT[:2], k=10, loop_len=12, cold=False,
                               at_floor=True)
    for _ in range(10):
        ann.record_outcome(floor_plan, success=True)
    assert ann.min_prefix == 1                                  # one piece at a time
    for _ in range(10):
        ann.record_outcome(WarmStartPlan(FLAT[:1], 11, 12, False, True), success=True)
    assert ann.min_prefix == 0                                  # fully cold context
    for _ in range(10):                                         # no underflow
        ann.record_outcome(WarmStartPlan([], 12, 12, False, True), success=True)
    assert ann.min_prefix == 0


def test_floor_failure_demotes_prefix_back_up(tmp_path):
    ann = WarmStartAnnealer(k_init=999, promote_n=10, promote_rate=0.6,
                            demote_rate=0.15, rng=random.Random(0))
    ann.min_prefix = 3
    ann.min_prefix_init = 6
    floor_plan = WarmStartPlan(prefix=FLAT[:3], k=9, loop_len=12, cold=False,
                               at_floor=True)
    for _ in range(10):
        ann.record_outcome(floor_plan, success=True)
    assert ann.min_prefix == 2
    for _ in range(10):
        ann.record_outcome(WarmStartPlan(FLAT[:2], 10, 12, False, True), success=False)
    assert ann.min_prefix == 3                                  # demote, never above init
    ann.min_prefix = 6
    for _ in range(10):
        ann.record_outcome(WarmStartPlan(FLAT[:6], 6, 12, False, True), success=False)
    assert ann.min_prefix == 6


def test_floor_and_k_frontiers_are_independent(tmp_path):
    """A floor-bound failure burst must not demote k_max (and vice versa): the two
    anneals track different skills (depth vs opening) and share no window."""
    ann = WarmStartAnnealer(k_init=10, promote_n=10, promote_rate=0.6,
                            demote_rate=0.15, rng=random.Random(0))
    ann.min_prefix = 3
    ann.min_prefix_init = 6
    k_before = ann.k_max
    for _ in range(10):
        ann.record_outcome(WarmStartPlan(FLAT[:3], 9, 12, False, True), success=False)
    assert ann.k_max == k_before                                # k frontier untouched
    assert ann.min_prefix == 4
    for _ in range(10):                                         # k-frontier (not floor)
        ann.record_outcome(WarmStartPlan(FLAT[:2], 10, 12, False, False), success=True)
    assert ann.k_max == 12 and ann.min_prefix == 4              # floor untouched


def test_step_info_emits_warm_min_prefix(monkeypatch, tmp_path):
    """The prefix anneal's position must be observable in TB (house rule: every new
    mechanism streams its own diagnostic) -- it IS the progress meter of the last gap."""
    wrapper, _ = _wrapped(monkeypatch, tmp_path, initial_phase=6, warm_k_init=86)
    info = _run_episode(wrapper)
    assert info['warm_min_prefix'] == 6
    wrapper._annealer.min_prefix = 2
    info = _run_episode(wrapper)
    assert info['warm_min_prefix'] == 2


def test_harvest_tags_cold_source(monkeypatch, tmp_path):
    """Aug-4 (gallery request): harvests must record whether the build was UNAIDED --
    'harvest_cold' vs 'harvest' -- so inspection samples can split current cold
    behavior from scaffolded work. Source is a free string; no schema migration."""
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    lib_path = str(tmp_path / "lib.jsonl")
    monkeypatch.setattr(OpenRCT2Env, "_LOOP_LIBRARY_PATH", lib_path)
    env = OpenRCT2Env(verbose=0)
    env.reset()
    env._warm_cold = True
    for _ in range(20):
        _, _, term, trunc, _ = env.step(0)
        if term or trunc:
            break
    lib = LoopLibrary(lib_path)
    assert len(lib) >= 1
    assert all(r.source == "harvest_cold" for r in lib._records.values())

    lib_path2 = str(tmp_path / "lib2.jsonl")
    monkeypatch.setattr(OpenRCT2Env, "_LOOP_LIBRARY_PATH", lib_path2)
    env2 = OpenRCT2Env(verbose=0)
    env2.reset()
    env2._warm_cold = False
    for _ in range(20):
        _, _, term, trunc, _ = env2.step(0)
        if term or trunc:
            break
    lib2 = LoopLibrary(lib_path2)
    assert all(r.source == "harvest" for r in lib2._records.values())


def test_wrapper_warm_min_prefix_resumes_descent(monkeypatch, tmp_path):
    """Aug-4: the prefix-descent state is in-memory like k was -- a restart re-armed
    the floor at 6 and re-walked ~30h of descent. warm_min_prefix seeds the floor at
    the achieved rung (0 = the natural end state that produced the cold drift);
    min_prefix_init stays 6 so the demote safety ceiling is unchanged."""
    wrapper, _ = _wrapped(monkeypatch, tmp_path, initial_phase=6, warm_k_init=86,
                          warm_min_prefix=0)
    assert wrapper._annealer.min_prefix == 0
    assert wrapper._annealer.min_prefix_init == 6
    w2, _ = _wrapped(monkeypatch, tmp_path.joinpath("d"), initial_phase=6)
    assert w2._annealer.min_prefix == 6                       # default arming unchanged


# ------------------------------------ harvest provenance (Aug-6, "which episode?")
# Markus asked which episodes two gallery coasters came from and the library could not
# say: records carry no time or instance identity, so age could only be guessed from
# file position and one coaster narrowed to 8 candidates. Records now stamp wall-clock
# ts + the OpenRCT2 port that built them. ts (not a step counter) because TB timestamps
# every scalar, so ts -> exact training step is a lookup -- and no plumbing has to reach
# into the running training loop to fetch num_timesteps.

def test_loop_record_provenance_roundtrip(tmp_path):
    lib = LoopLibrary(str(tmp_path / "lib.jsonl"))
    lib.add(LoopRecord.from_actions([4, 0, 3], "harvest_cold", ts=1785000000.5, port=8093))
    reloaded = LoopLibrary(str(tmp_path / "lib.jsonl"))
    (rec,) = reloaded._records.values()
    assert rec.ts == pytest.approx(1785000000.5)
    assert rec.port == 8093


def test_loop_record_provenance_defaults_for_legacy(tmp_path):
    """Every pre-Aug-6 line lacks both fields; they must load, not crash, and be
    identifiable as unknown (the excitement-field precedent: no migration)."""
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps({"actions": [4, 0, 3], "length": 3, "chain_count": 0,
                                "max_gain": 0.0, "drop_z": 0.0, "source": "harvest"}) + "\n")
    (rec,) = LoopLibrary(str(path))._records.values()
    assert rec.ts == 0.0 and rec.port == -1


def test_harvest_stamps_time_and_port(monkeypatch, tmp_path):
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    lib_path = str(tmp_path / "lib.jsonl")
    monkeypatch.setattr(OpenRCT2Env, "_LOOP_LIBRARY_PATH", lib_path)
    monkeypatch.setattr(oe_mod.time, "time", lambda: 1785123456.0)
    env = OpenRCT2Env(port=8087, verbose=0)
    env.reset()
    for _ in range(20):
        _, _, term, trunc, _ = env.step(0)
        if term or trunc:
            break
    recs = list(LoopLibrary(lib_path)._records.values())
    assert recs and all(r.ts == pytest.approx(1785123456.0) for r in recs)
    assert all(r.port == 8087 for r in recs)


# --------------------------- descent must be gated on STYLE, not completion (Aug-7)
# The prefix descent shrank the opening seed whenever floor-bound builds COMPLETED --
# never checking whether they still wound. So it dismantled the winding scaffold on
# evidence of the wrong thing: it walked 6 -> 0 twice (once on S-bend-farmed successes,
# once on honest ones) and both times left cold builds at the 4-turn rectangle, because
# a seeded build that completes as a rectangle still counts as "ready for a smaller
# seed". Floor-bound outcomes now require completion AND retained style.

def test_floor_descent_ignores_unstyled_completions():
    ann = WarmStartAnnealer(k_init=999, promote_n=10, promote_rate=0.6,
                            rng=random.Random(0))
    ann.min_prefix = 4
    ann.min_prefix_init = 6
    plan = WarmStartPlan(prefix=FLAT[:4], k=8, loop_len=12, cold=False, at_floor=True)
    for _ in range(10):
        ann.record_outcome(plan, success=True, styled=False)   # completes, but flat
    # Not merely "does not shrink": an all-unstyled window is a demote signal, so the
    # seed WIDENS back toward init -- the agent gets more winding demonstration, which
    # is the correct response to the skill being lost.
    assert ann.min_prefix == 5


def test_floor_descent_advances_on_styled_completions():
    ann = WarmStartAnnealer(k_init=999, promote_n=10, promote_rate=0.6,
                            rng=random.Random(0))
    ann.min_prefix = 4
    ann.min_prefix_init = 6
    plan = WarmStartPlan(prefix=FLAT[:4], k=8, loop_len=12, cold=False, at_floor=True)
    for _ in range(10):
        ann.record_outcome(plan, success=True, styled=True)
    assert ann.min_prefix == 3


def test_k_frontier_still_promotes_on_plain_completion():
    """Build DEPTH is genuinely about closing loops -- the style gate must not leak
    into the k anneal (they track different skills, as the split window already says)."""
    ann = WarmStartAnnealer(k_init=10, promote_n=10, promote_rate=0.6,
                            rng=random.Random(0))
    plan = WarmStartPlan(prefix=FLAT[:2], k=10, loop_len=12, cold=False, at_floor=False)
    for _ in range(10):
        ann.record_outcome(plan, success=True, styled=False)
    assert ann.k_max == 12


def test_record_outcome_styled_defaults_backward_compatible():
    ann = WarmStartAnnealer(k_init=999, promote_n=10, promote_rate=0.6,
                            rng=random.Random(0))
    ann.min_prefix = 2
    ann.min_prefix_init = 6
    plan = WarmStartPlan(prefix=FLAT[:2], k=10, loop_len=12, cold=False, at_floor=True)
    for _ in range(10):
        ann.record_outcome(plan, success=True)                 # no styled arg
    assert ann.min_prefix == 1


def test_floor_style_bar_is_reachable_and_self_correcting(monkeypatch, tmp_path):
    """The descent bar must be one FLOOR-BOUND builds can actually clear, or the seed
    never shrinks and the context gap never closes. Live evidence (Aug-8, 2,895
    provenance-stamped harvests): seeded builds average 7.9 heading turns (49% >= 8),
    but floor-bound ones -- where the agent builds everything behind a 6-piece seed --
    sit near the cold end, and a bar of 8 stalled the descent for ~570k steps.

    6 is safe because the demote path makes this an EQUILIBRIUM SEARCH, not a one-way
    walk: at any bar the seed widens back wherever style fails, so it settles at the
    seed size the policy can actually hold. (Under the old completion-only criterion no
    such feedback existed -- completion always succeeded, so the seed always hit 0.)"""
    W = ImprovedPhasedCurriculumWrapper
    assert W.FLOOR_STYLE_MIN_TURNS == 6

    ann = WarmStartAnnealer(k_init=999, promote_n=10, promote_rate=0.6,
                            demote_rate=0.15, rng=random.Random(0))
    ann.min_prefix, ann.min_prefix_init = 3, 6
    floor = WarmStartPlan(prefix=FLAT[:3], k=9, loop_len=12, cold=False, at_floor=True)
    for _ in range(10):                      # style held -> seed shrinks
        ann.record_outcome(floor, success=True, styled=True)
    assert ann.min_prefix == 2
    for _ in range(10):                      # style lost -> seed widens back
        ann.record_outcome(WarmStartPlan(FLAT[:2], 10, 12, False, True),
                           success=True, styled=False)
    assert ann.min_prefix == 3


# ------------------ what "styled" MEANS is phase-dependent (Aug-9 seed conditioning)
# The descent bar was a fixed turns>=6, which matched the P6 reward while that reward
# paid for turns. It no longer does: on an oval seed the family gate pays MAXIMALLY at
# <=5 turns, so a correctly-built oval could never be "styled" and the prefix descent
# stalled -- and cold builds are the only thing the success criteria are measured on.

def _styled_flags(wrapper, base, actions):
    """Drive one full episode of `actions` to a COMPLETION and return the `styled`
    values the wrapper handed the annealer."""
    seen = []
    wrapper._annealer.record_outcome = (
        lambda plan, success, styled=None: seen.append((success, styled)))
    base.api_controller.complete_after = len(actions)
    wrapper.reset()
    for a in actions:
        _, _, terminated, truncated, _ = wrapper.step(a)
        if terminated or truncated:
            break
    else:
        raise AssertionError("episode did not end")
    assert [s for s, _ in seen] == [True], "the build must actually close the loop"
    return [styled for _, styled in seen]


OVAL_BUILD = [4, 4, 4, 4, 0]        # 4 heading turns, no alternation -> family 0 (oval)


def test_p6_styled_means_built_the_seed_family(monkeypatch, tmp_path):
    """In P6 the descent bar is the family the seed asked for. target_family is 0 (oval)
    today, and an oval is 0-5 turns -- under the old turns>=6 bar this correctly-built
    episode would report styled=False and the seed would never shrink."""
    wrapper, base = _wrapped(monkeypatch, tmp_path, api_cls=CompletingAPI, p_cold=1.0)
    wrapper.current_phase = 6
    wrapper._update_phase_settings()
    # This test is about the family-hit predicate, not the Aug-9 per-episode sampler
    # (task 5) -- pin the seed so it stays deterministic now that P6 draws widely from
    # PHASE_FAMILIES[6].
    wrapper._sample_target_family = lambda: 0
    assert base.target_family == 0
    (styled,) = _styled_flags(wrapper, base, OVAL_BUILD)
    assert wrapper._history_turn_count(base) == 4 < wrapper.FLOOR_STYLE_MIN_TURNS
    assert styled is True


def test_pre_p6_styled_still_means_the_turn_count_bar(monkeypatch, tmp_path):
    """The bit-identical guard: phases whose reward still pays for turns keep the fixed
    turns>=FLOOR_STYLE_MIN_TURNS bar, so the SAME 4-turn build is not styled there."""
    wrapper, base = _wrapped(monkeypatch, tmp_path, api_cls=CompletingAPI, p_cold=1.0)
    assert wrapper.current_phase == 1
    assert wrapper._phase_reward_params(1).qualify_requires_family is False
    (styled,) = _styled_flags(wrapper, base, OVAL_BUILD)
    assert wrapper._history_turn_count(base) == 4
    assert styled is False


# ------------- ...and it is judged on the AGENT-BUILT SUFFIX (Aug-9 fix pass 2)
# `styled` only ever reaches the annealer on floor-bound WARM plans -- the exact episodes
# that replay a 6-piece winding OPENING. Classifying the whole track therefore let the
# scaffold's own jog decide the predicate: a jog is a direction switch, family 0 (oval)
# allows none, so every at-floor episode reported styled=False and the descent stalled at
# min_prefix_init. The predicate means "did the AGENT build the shape", so it skips the
# replayed prefix -- the same conflation LoopRecord.agent_turn_count exists to avoid.

class _FrontierRng(random.Random):
    """random() == 0.0: sample_plan never draws cold (with p_cold=0) and always takes
    k = k_hi, so every episode is the floor-bound draw the prefix descent feeds on."""

    def random(self):
        return 0.0


# A 12-piece scaffold record whose OPENING jogs (R, L, R -> a direction switch), like the
# seeded P6 exemplars; the chain piece puts it in the pool's `chained` fallback tier.
JOG_OPENER = [4, 0, 3, 0, 4, 10, 0, 0, 0, 0, 0, 0]
FLOOR_PREFIX_LEN = 5                        # k_max = 12 - 5 = 7 keeps p_cold at its base
SPIRAL_BUILD = [4, 4, 4, 4, 4, 4]           # 6 turns, no alternation -> family 1 (spiral)


def _p6_floor_wrapper(monkeypatch, tmp_path, min_prefix=FLOOR_PREFIX_LEN):
    """A P6 wrapper wired so every draw is a floor-bound warm plan (`at_floor=True`),
    the only configuration in which `styled` has any effect."""
    wrapper, base = _wrapped(monkeypatch, tmp_path, api_cls=CompletingAPI,
                             seed_loops=(JOG_OPENER,), p_cold=0.0)
    wrapper.current_phase = 6
    wrapper._update_phase_settings()
    wrapper._annealer._rng = _FrontierRng()
    # k_max == the per-record cap (len - min_prefix) is what makes a draw floor-bound;
    # keeping it under 8 also keeps the competence-scaled p_cold at its base of 0.
    wrapper._annealer.k_max = len(JOG_OPENER) - min_prefix
    assert wrapper._annealer.p_cold == 0.0
    wrapper._annealer.min_prefix = min_prefix
    # These tests are about the family-hit predicate at a fixed family, not the Aug-9
    # per-episode sampler (task 5) -- pin the seed so it stays deterministic now that
    # P6 draws widely from PHASE_FAMILIES[6].
    wrapper._sample_target_family = lambda: 0
    return wrapper, base


def _floor_episode(wrapper, base, actions):
    """One floor-bound scaffolded episode: the opening seed is replayed, then the agent
    builds `actions` and the last one closes the circuit."""
    base.api_controller.complete_after = wrapper._annealer.min_prefix + len(actions)
    wrapper.reset()
    plan = wrapper._current_plan
    assert plan.cold is False and plan.at_floor is True
    assert len(plan.prefix) == wrapper._annealer.min_prefix
    assert len(base.track_builder.history) == len(plan.prefix)
    for a in actions:
        _, _, terminated, truncated, _ = wrapper.step(a)
        if terminated or truncated:
            break
    else:
        raise AssertionError("episode did not end")
    assert base.loop_completed is True, "the build must actually close the loop"
    return [h.get('action') for h in base.track_builder.history]


def _floor_styled(wrapper, base, actions):
    """The `styled` value one floor-bound episode hands the annealer (spied, so the
    descent itself stays frozen and the floor-bound configuration stays valid)."""
    seen = []
    wrapper._annealer.record_outcome = (
        lambda plan, success, styled=None: seen.append((success, styled)))
    history = _floor_episode(wrapper, base, actions)
    assert [s for s, _ in seen] == [True]
    return seen[0][1], history


def test_p6_styled_judges_the_agent_suffix_not_the_replayed_opening(monkeypatch, tmp_path):
    """THE regression: the scaffold's jogging opening must not decide the predicate. The
    agent builds a clean oval behind it, so the episode is styled even though the whole
    track (jog + oval) classifies as out_and_back."""
    wrapper, base = _p6_floor_wrapper(monkeypatch, tmp_path)
    styled, history = _floor_styled(wrapper, base, OVAL_BUILD)
    assert base.target_family == 0
    assert classify_family(history) != 0                    # whole track: the jog decides
    assert classify_family(history[FLOOR_PREFIX_LEN:]) == 0  # agent's suffix: a clean oval
    assert styled is True


def test_p6_styled_still_says_no_when_the_agent_suffix_misses_the_family(monkeypatch,
                                                                        tmp_path):
    """...and the predicate must still be able to refuse: same jogging opening, but the
    agent's own suffix is a spiral, not the oval the seed asked for."""
    wrapper, base = _p6_floor_wrapper(monkeypatch, tmp_path)
    styled, history = _floor_styled(wrapper, base, SPIRAL_BUILD)
    assert base.target_family == 0
    assert classify_family(history[FLOOR_PREFIX_LEN:]) == 1  # spiral
    assert styled is False


def test_p6_prefix_descent_moves_on_the_agent_built_style(monkeypatch, tmp_path):
    """The descent itself, end to end: `styled` reaches the annealer only through
    floor-bound WARM plans (record_outcome returns early on cold ones, and phases 1-5
    pin min_prefix to 0), so this is the only configuration where the predicate has
    consequences. Styled successes shrink the opening seed; unstyled ones widen it."""
    wrapper, base = _p6_floor_wrapper(monkeypatch, tmp_path)
    wrapper._annealer.promote_n = 4
    for _ in range(4):
        _floor_episode(wrapper, base, OVAL_BUILD)
    assert wrapper._annealer.min_prefix == FLOOR_PREFIX_LEN - 1

    # re-arm the floor-bound draw, then lose the style
    wrapper._annealer.min_prefix = FLOOR_PREFIX_LEN
    wrapper._annealer.floor_frontier.clear()
    for _ in range(4):
        _floor_episode(wrapper, base, SPIRAL_BUILD)
    assert wrapper._annealer.min_prefix == FLOOR_PREFIX_LEN + 1
    assert wrapper._annealer.min_prefix <= wrapper._annealer.min_prefix_init


# ------------------------ agent-built vs replayed credit (Aug-9, "who built the turns?")
# "Warm builds wind at 7.9 turns" conflated what the AGENT built with what the scaffold
# REPLAYED: a record stores the whole track, including the pre-placed prefix, so a long
# prefix made the exemplar's own turns look like the policy's work. Same class of error
# as counting S-bends as turns. Records now store prefix_len so agent-built structure
# can be measured directly.

def test_loop_record_prefix_len_roundtrip_and_legacy(tmp_path):
    lib = LoopLibrary(str(tmp_path / "lib.jsonl"))
    lib.add(LoopRecord.from_actions([4, 0, 3, 4], "harvest", prefix_len=2))
    (rec,) = LoopLibrary(str(tmp_path / "lib.jsonl"))._records.values()
    assert rec.prefix_len == 2
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(json.dumps({"actions": [4, 0, 3], "length": 3, "chain_count": 0,
                                  "max_gain": 0.0, "drop_z": 0.0, "source": "harvest"}) + "\n")
    (old,) = LoopLibrary(str(legacy))._records.values()
    assert old.prefix_len == 0            # legacy: unknown, treated as all-agent


def test_agent_turn_count_excludes_the_replayed_prefix():
    """THE measurement that settles whether the policy composes winding or inherits it."""
    rec = LoopRecord.from_actions([4, 3, 4, 3, 0, 0, 4, 0], "harvest", prefix_len=4)
    assert rec.turn_count == 5            # whole track
    assert rec.agent_turn_count == 1      # only the piece the agent placed after the seed
    cold = LoopRecord.from_actions([4, 3, 4], "harvest_cold", prefix_len=0)
    assert cold.agent_turn_count == cold.turn_count == 3


def test_harvest_stamps_prefix_len(monkeypatch, tmp_path):
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    lib_path = str(tmp_path / "lib.jsonl")
    monkeypatch.setattr(OpenRCT2Env, "_LOOP_LIBRARY_PATH", lib_path)
    env = OpenRCT2Env(port=8081, verbose=0)
    env.reset()
    env._warm_prefix_len = 3              # pretend 3 pieces were replayed at reset
    for _ in range(20):
        _, _, term, trunc, _ = env.step(0)
        if term or trunc:
            break
    recs = list(LoopLibrary(lib_path)._records.values())
    assert recs and all(r.prefix_len == 3 for r in recs)


# ------------------------------- per-phase family sets (Aug-9 seed conditioning)

def test_phase_family_sets_widen_with_the_track_budget():
    W = ImprovedPhasedCurriculumWrapper
    assert W.PHASE_FAMILIES[1] == ()          # 40 pieces: too tight to express shape
    assert W.PHASE_FAMILIES[2] == ()
    assert W.PHASE_FAMILIES[3] == (0, 1, 2)   # oval, spiral, out-and-back
    assert W.PHASE_FAMILIES[4] == (0, 1, 2, 3)
    assert W.PHASE_FAMILIES[5] == (0, 1, 2, 3, 4)
    assert W.PHASE_FAMILIES[6] == (0, 1, 2, 3, 4)


def test_wrapper_sets_the_target_family_on_the_base_env_before_reset(monkeypatch, tmp_path):
    wrapper, base = _wrapped(monkeypatch, tmp_path, initial_phase=6)
    seen = set()
    for _ in range(40):
        wrapper.reset()
        seen.add(base.target_family)
    assert seen <= set(range(5))
    assert len(seen) >= 2, "the seed must actually vary across episodes"


def test_early_phases_pin_the_family_to_zero(monkeypatch, tmp_path):
    """Phases 1-2 have no family reward, so the seed must not wander -- it would be
    pure noise in the observation."""
    wrapper, base = _wrapped(monkeypatch, tmp_path)
    for _ in range(10):
        wrapper.reset()
        assert base.target_family == 0


def test_step_info_reports_target_family_and_family_hit(monkeypatch, tmp_path):
    wrapper, base = _wrapped(monkeypatch, tmp_path, api_cls=CompletingAPI,
                              initial_phase=6, p_cold=1.0)
    info = _run_episode(wrapper)
    assert "target_family" in info
    assert "family_hit" in info


# ---------------------- Fix 1: the per-family window survives phase advancement

def test_family_window_clears_on_phase_advancement(monkeypatch, tmp_path):
    """episode_family_results must not survive a phase change -- otherwise P6 entry
    inherits stale per-family outcomes measured against an earlier phase's qualified
    predicate (see _clear_phase_windows)."""
    wrapper, base = _wrapped(monkeypatch, tmp_path, api_cls=CompletingAPI,
                              initial_phase=5, p_cold=1.0)
    wrapper._sample_target_family = lambda: 0
    base.api_controller.complete_after = len(OVAL_BUILD)
    wrapper.reset()
    for a in OVAL_BUILD:
        _, _, terminated, truncated, _ = wrapper.step(a)
        if terminated or truncated:
            break
    assert len(wrapper.episode_family_results[0]) == 1

    wrapper._advance_to_phase(6)
    assert len(wrapper.episode_family_results[0]) == 0
    for fz in range(5):
        assert len(wrapper.episode_family_results[fz]) == 0


# ---------------------- Fix 2: the rate key is gated on active families, with an
# explicit denominator (family_n_{z}) so 0.0-no-samples reads differently from
# 0.0-all-miss.

def test_family_hit_rate_key_absent_when_no_family_active(monkeypatch, tmp_path):
    wrapper, base = _wrapped(monkeypatch, tmp_path, api_cls=CompletingAPI, p_cold=1.0)
    assert wrapper.current_phase == 1
    assert wrapper.PHASE_FAMILIES[1] == ()
    info = _run_episode(wrapper)
    assert not any(k.startswith("family_hit_rate_") for k in info)
    assert not any(k.startswith("family_n_") for k in info)


def test_family_hit_rate_emits_all_active_families_with_n_companion(monkeypatch, tmp_path):
    wrapper, base = _wrapped(monkeypatch, tmp_path, api_cls=CompletingAPI,
                              initial_phase=6, p_cold=1.0)
    info = _run_episode(wrapper)
    for fz in range(5):
        assert f"family_hit_rate_{fz}" in info
        assert f"family_n_{fz}" in info


def test_family_hit_rate_reports_zero_before_any_cold_episode(monkeypatch, tmp_path):
    wrapper, base = _wrapped(monkeypatch, tmp_path, api_cls=CompletingAPI,
                              initial_phase=6, p_cold=1.0)
    wrapper._sample_target_family = lambda: 0
    base.api_controller.complete_after = len(OVAL_BUILD)
    wrapper.reset()
    info = None
    for a in OVAL_BUILD:
        _, _, terminated, truncated, info = wrapper.step(a)
        if terminated or truncated:
            break
    assert info['family_n_0'] == 1
    for fz in (1, 2, 3, 4):
        assert info[f'family_n_{fz}'] == 0
        assert info[f'family_hit_rate_{fz}'] == 0.0


# ---------------------- Fix 3: lock down what the tracking actually computes, not
# merely that the keys exist.

def test_family_hit_rate_equals_hits_over_total(monkeypatch, tmp_path):
    wrapper, base = _wrapped(monkeypatch, tmp_path, api_cls=CompletingAPI,
                              initial_phase=6, p_cold=1.0)
    wrapper._sample_target_family = lambda: 0
    info = None
    for _ in range(2):                              # two hits: oval build == oval seed
        base.api_controller.complete_after = len(OVAL_BUILD)
        wrapper.reset()
        for a in OVAL_BUILD:
            _, _, terminated, truncated, info = wrapper.step(a)
            if terminated or truncated:
                break
    # one miss: still seeded oval, but the agent builds a spiral instead
    base.api_controller.complete_after = len(SPIRAL_BUILD)
    wrapper.reset()
    for a in SPIRAL_BUILD:
        _, _, terminated, truncated, info = wrapper.step(a)
        if terminated or truncated:
            break
    assert info['family_n_0'] == 3
    assert info['family_hit_rate_0'] == pytest.approx(2 / 3)


def test_family_hit_rate_excludes_warm_episodes_from_denominator(monkeypatch, tmp_path):
    """A scaffolded build inherits its shape from the exemplar, so a warm episode must
    not move the count -- drive a mixed cold/warm/cold sequence and check the family_n_
    denominator only advances on the cold ones."""
    wrapper, base = _wrapped(monkeypatch, tmp_path, api_cls=CompletingAPI,
                              seed_loops=(FLAT,), p_cold=1.0)
    wrapper.current_phase = 6
    wrapper._update_phase_settings()
    wrapper._sample_target_family = lambda: 0

    plans = iter([
        WarmStartPlan(prefix=[], k=0, loop_len=0, cold=True),
        WarmStartPlan(prefix=list(FLAT[:6]), k=6, loop_len=len(FLAT), cold=False),
        WarmStartPlan(prefix=[], k=0, loop_len=0, cold=True),
    ])
    wrapper._sample_warm_start = lambda: next(plans)

    def _drive(actions, complete_after):
        base.api_controller.complete_after = complete_after
        wrapper.reset()
        info = None
        for a in actions:
            _, _, terminated, truncated, info = wrapper.step(a)
            if terminated or truncated:
                break
        return info

    info = _drive(OVAL_BUILD, len(OVAL_BUILD))
    assert info['family_n_0'] == 1                      # cold #1 counted

    info = _drive(FLAT[6:], len(FLAT))
    assert info['family_n_0'] == 1                      # warm episode must NOT advance it

    info = _drive(OVAL_BUILD, len(OVAL_BUILD))
    assert info['family_n_0'] == 2                      # cold #2 counted


def test_family_hit_lands_in_the_drawn_familys_bucket_only(monkeypatch, tmp_path):
    """A hit must land in the family the episode's seed drew, not some other family's
    window -- drive two different seeds and check each window only moves once."""
    wrapper, base = _wrapped(monkeypatch, tmp_path, api_cls=CompletingAPI,
                              initial_phase=6, p_cold=1.0)
    seeds = iter([0, 1])
    wrapper._sample_target_family = lambda: next(seeds)

    base.api_controller.complete_after = len(OVAL_BUILD)
    wrapper.reset()
    info = None
    for a in OVAL_BUILD:
        _, _, terminated, truncated, info = wrapper.step(a)
        if terminated or truncated:
            break
    assert info['target_family'] == 0
    assert info['family_n_0'] == 1
    assert info['family_n_1'] == 0

    base.api_controller.complete_after = len(SPIRAL_BUILD)
    wrapper.reset()
    for a in SPIRAL_BUILD:
        _, _, terminated, truncated, info = wrapper.step(a)
        if terminated or truncated:
            break
    assert info['target_family'] == 1
    assert info['family_n_1'] == 1
    assert info['family_n_0'] == 1                      # untouched by the second episode


# ---------------------------------------------- seed_p5_exemplars.py CLI (task 8, Fix 2)

def test_seed_p5_exemplars_cli_exposes_family_and_footprint_flags(monkeypatch):
    """--family already existed (choices p5/p6); this only adds "serpentine" to it and
    introduces a SEPARATE --footprint-family int flag (default None) that filters the
    seeder's output by footprint shape. Follows run_model.py's add_argument-spy pattern
    (test_run_model.py::test_cli_exposes_showcase_flags) since this script has none of
    its own CLI-surface tests yet."""
    import argparse
    import seed_p5_exemplars as S

    seen = []
    real_add = argparse.ArgumentParser.add_argument

    def spy(self, *a, **kw):
        seen.append((a, kw))
        return real_add(self, *a, **kw)

    monkeypatch.setattr(argparse.ArgumentParser, "add_argument", spy)
    args = S.parse_args(["--port", "8080"])

    by_flag = {a[0]: kw for a, kw in seen if a and isinstance(a[0], str)}
    assert "serpentine" in by_flag["--family"]["choices"]
    assert "p5" in by_flag["--family"]["choices"] and "p6" in by_flag["--family"]["choices"]
    assert by_flag["--footprint-family"]["type"] is int
    assert by_flag["--footprint-family"]["default"] is None

    assert args.family == "p5"                          # unchanged default
    assert args.footprint_family is None
