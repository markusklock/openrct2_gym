# Seed-Conditioned Coaster Variety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the agent build a different wooden coaster on each run by conditioning it on a random seed that names a footprint family, while it keeps maximising ride score.

**Architecture:** Each episode draws a seed `z` ∈ {0..4} naming a footprint family (oval, spiral, out-and-back, winding, serpentine), defined by turn count and direction-switch count. `z` enters the observation as a `Discrete` field (SB3 one-hots it). A graded `family_match` term feeds a multiplicative completion gate, a discrete bonus, and the dense PBRS potential. The fixed 12-turn target and handedness-balance leg are switched off — the target now comes from `z`.

**Tech Stack:** Python 3.11, Gymnasium, Stable-Baselines3 + sb3_contrib (MaskablePPO), PyTorch, pytest. Training runs on the `htpc` host over SSH; tests run locally and are server-free.

## Global Constraints

- **Design source of truth:** `docs/superpowers/specs/2026-08-09-seed-conditioned-coaster-variety-design.md`.
- **TDD, always:** write the failing test, run it, watch it fail for the right reason, then implement. Never write implementation first.
- **Server-free tests:** every test in this plan runs without OpenRCT2. Use the existing `FakeAPI` / `CompletingAPI` doubles in `openrct2_gym/tests/`.
- **Full suite green before each commit:** `venv/bin/python -m pytest openrct2_gym/tests/ -q` (currently 420 passing).
- **Completion-first invariant:** a completed ride must never be out-paid by not completing. `ImprovedPhasedCurriculumWrapper._validate_completion_first` enforces this and must keep passing.
- **Every new mechanism streams its own diagnostic** to TensorBoard. A mechanism with no tag is not finished.
- **Measurement rule:** reward may score the whole track, but success claims are judged on unaided (cold) builds only.
- **Phases 1–2 must stay bit-identical** — all new reward weights default to 0 / 1.0 (inert).
- **Run commands from the repo root** `/home/markus/git/openrct2_gym` with `venv/bin/python`.

---

### Task 1: Footprint descriptor module

**Files:**
- Create: `openrct2_gym/envs/footprint.py`
- Test: `openrct2_gym/tests/test_footprint.py`

**Interfaces:**
- Consumes: `openrct2_gym.envs.track_pieces.LEFT_TURN_ACTIONS`, `RIGHT_TURN_ACTIONS` (existing tuples of action ids).
- Produces: `FAMILIES` (tuple of 5 tuples `(name, turn_lo, turn_hi, switch_lo, switch_hi)`, `hi=None` means unbounded), `FAMILY_N = 5`, `switch_count(actions) -> int`, `classify_family(actions) -> int | None`, `family_match(actions, family_index, turn_falloff=5.0, switch_falloff=3.0) -> float` in `[0,1]`.

- [ ] **Step 1: Write the failing test**

Create `openrct2_gym/tests/test_footprint.py`:

```python
"""Footprint families: the shape a coaster traces on the ground.

Turn count and direction-switch count are what the eye registers from above -- an
oval never alternates direction, a serpentine alternates constantly. Both come from
the action sequence alone, so this is pure and server-free.
"""
import pytest

from openrct2_gym.envs.footprint import (
    FAMILIES, FAMILY_N, classify_family, family_match, switch_count)

OVAL = [4, 0, 4, 0, 4, 0, 4]                        # 4 right turns, never alternates
SPIRAL = [4] * 8                                     # 8 right turns, never alternates
OUT_AND_BACK = [4, 4, 4, 0, 3, 3, 3, 0]              # 6 turns, one L/R switch
WINDING = [4, 4, 0, 3, 3, 0, 4, 4, 0, 3, 3, 0]       # 12 turns, 3 switches
SERPENTINE = [4, 3] * 8                              # 16 turns, alternating every piece


def test_switch_count_counts_direction_alternations():
    assert switch_count(OVAL) == 0
    assert switch_count(OUT_AND_BACK) == 1
    assert switch_count(WINDING) == 3
    assert switch_count(SERPENTINE) == 15


def test_switch_count_ignores_sbends_and_straights():
    """S-bends hand back the original heading, so they are not turns (Aug-6 exploit)."""
    assert switch_count([4, 29, 30, 4]) == 0
    assert switch_count([0, 0, 0]) == 0


def test_each_family_classifies_to_its_own_index():
    names = [f[0] for f in FAMILIES]
    assert names == ["oval", "spiral", "out_and_back", "winding", "serpentine"]
    assert FAMILY_N == 5
    assert classify_family(OVAL) == 0
    assert classify_family(SPIRAL) == 1
    assert classify_family(OUT_AND_BACK) == 2
    assert classify_family(WINDING) == 3
    assert classify_family(SERPENTINE) == 4


def test_classify_returns_none_when_no_family_fits():
    """11 turns with no alternation belongs to nothing: too many for oval, too few
    switches for winding. An honest None beats forcing it into a bin."""
    assert classify_family([4] * 11) is None


def test_family_match_is_full_inside_the_band():
    assert family_match(WINDING, 3) == pytest.approx(1.0)
    assert family_match(OVAL, 0) == pytest.approx(1.0)


def test_family_match_falls_off_gradually_outside_the_band():
    """Graded, not pass/fail: partial credit for getting nearer, or the target is
    never discovered (the campaign's every-leg-needs-a-ramp rule)."""
    near = family_match([4, 4, 0, 3, 3, 0, 4, 4, 0, 3], 3)      # 10 turns, 3 switches
    far = family_match([4, 0, 4], 3)                             # 2 turns, 0 switches
    assert 0.0 < far < near <= 1.0


def test_family_match_penalises_the_wrong_family():
    """The core inversion: with an oval requested, a winding build must score worse."""
    assert family_match(OVAL, 0) > family_match(WINDING, 0)
    assert family_match(WINDING, 3) > family_match(OVAL, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest openrct2_gym/tests/test_footprint.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'openrct2_gym.envs.footprint'`

- [ ] **Step 3: Write minimal implementation**

Create `openrct2_gym/envs/footprint.py`:

```python
"""Footprint families -- what a coaster looks like from above.

Two properties decide the family, and they are the two Markus named as what his eye
registers: how many heading turns the track makes, and how many times the turning
direction ALTERNATES. An oval never alternates; a serpentine alternates constantly.

Both are computed from the action sequence alone, so this module is dependency-free
(beyond the canonical piece families) and fully server-free testable.

S-bends are deliberately absent: they hand back the original heading, and counting
them as turns is precisely the exploit found on Aug-6 (61% of "turns" in cold builds
were S-bend padding). See track_pieces.
"""
from openrct2_gym.envs.track_pieces import LEFT_TURN_ACTIONS, RIGHT_TURN_ACTIONS

# (name, turn_lo, turn_hi, switch_lo, switch_hi); hi=None means unbounded.
# Every band is proven achievable at E >= 6.1 by the 194k-record archive -- see the
# design spec's family table. None of these is aspirational.
FAMILIES = (
    ("oval",         0,    5, 0,    0),
    ("spiral",       6,    9, 0,    0),
    ("out_and_back", 6,    9, 1,    2),
    ("winding",     10,   13, 3,    5),
    ("serpentine",  14, None, 6, None),
)
FAMILY_N = len(FAMILIES)


def turn_directions(actions):
    """L/R sequence of heading-changing pieces, in build order."""
    out = []
    for a in actions:
        if a in LEFT_TURN_ACTIONS:
            out.append("L")
        elif a in RIGHT_TURN_ACTIONS:
            out.append("R")
    return out


def switch_count(actions):
    """How many times the turning direction alternates -- the footprint signal."""
    dirs = turn_directions(actions)
    return sum(1 for x, y in zip(dirs, dirs[1:]) if x != y)


def _band_score(value, lo, hi, falloff):
    """1.0 inside [lo, hi], decaying linearly to 0 over `falloff` outside it."""
    if value < lo:
        return max(0.0, 1.0 - (lo - value) / falloff)
    if hi is not None and value > hi:
        return max(0.0, 1.0 - (value - hi) / falloff)
    return 1.0


def family_match(actions, family_index, turn_falloff=5.0, switch_falloff=3.0):
    """How well this build lands in the requested family, in [0, 1]. Graded so that
    getting nearer pays -- a pass/fail gate would never be discovered."""
    _, tlo, thi, slo, shi = FAMILIES[family_index]
    dirs = turn_directions(actions)
    turns = len(dirs)
    switches = sum(1 for x, y in zip(dirs, dirs[1:]) if x != y)
    return 0.5 * (_band_score(turns, tlo, thi, turn_falloff)
                  + _band_score(switches, slo, shi, switch_falloff))


def classify_family(actions):
    """Index of the family this build lands in, or None if it fits none.

    The bands do not tile the whole space (e.g. 11 turns with no alternation belongs
    to nothing). Returning None is deliberate: a build that matches no family is not
    a hit for any seed, and inventing a bin for it would hide that.
    """
    dirs = turn_directions(actions)
    turns = len(dirs)
    switches = sum(1 for x, y in zip(dirs, dirs[1:]) if x != y)
    for i, (_, tlo, thi, slo, shi) in enumerate(FAMILIES):
        if (turns >= tlo and (thi is None or turns <= thi)
                and switches >= slo and (shi is None or switches <= shi)):
            return i
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest openrct2_gym/tests/test_footprint.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Run the full suite**

Run: `venv/bin/python -m pytest openrct2_gym/tests/ -q`
Expected: PASS (427 passed — 420 existing + 7 new)

- [ ] **Step 6: Commit**

```bash
git add openrct2_gym/envs/footprint.py openrct2_gym/tests/test_footprint.py
git commit -m "feat: footprint family descriptor (turns + direction switches)"
```

---

### Task 2: Seed in the observation

**Files:**
- Modify: `openrct2_gym/envs/obs_config.py` (add constant + Discrete field)
- Modify: `openrct2_gym/envs/openrct2_env.py` (attribute + obs dict entry, near line 1791)
- Modify: `openrct2_gym/envs/feature_extractor.py:84` (`_cat_keys`)
- Test: `openrct2_gym/tests/test_footprint.py` (append)

**Interfaces:**
- Consumes: `footprint.FAMILY_N` from Task 1.
- Produces: `obs_config.TARGET_FAMILY_N`; observation key `"target_family"` (Discrete); `OpenRCT2Env.target_family` (int attribute, default 0, settable by the wrapper before `reset()`).

- [ ] **Step 1: Write the failing test**

Append to `openrct2_gym/tests/test_footprint.py`:

```python
# --------------------------------------------- the seed as an observation input
# The agent must SEE which family it was asked for, or it cannot condition on it.
# Discrete is used because SB3 one-hot-encodes it automatically, exactly like the
# existing current_direction / last_piece_type fields.

def test_observation_space_exposes_the_target_family():
    from openrct2_gym.envs.obs_config import TARGET_FAMILY_N, make_observation_space
    space = make_observation_space()
    assert TARGET_FAMILY_N == FAMILY_N
    assert "target_family" in space.spaces
    assert space["target_family"].n == FAMILY_N


def test_env_reports_the_target_family_in_its_observation(monkeypatch):
    import openrct2_gym.envs.openrct2_env as oe_mod
    from openrct2_gym.envs.openrct2_env import OpenRCT2Env
    from openrct2_gym.tests.test_env_smoke import FakeAPI

    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    env = OpenRCT2Env(verbose=0)
    assert env.target_family == 0                      # inert default
    env.target_family = 3                              # wrapper sets this before reset
    obs, _ = env.reset()
    assert obs["target_family"] == 3
    assert space_contains(env, obs)


def space_contains(env, obs):
    return env.observation_space.contains(obs)


def test_feature_extractor_consumes_the_target_family():
    """A Discrete field the extractor does not read is dead weight -- the policy
    would never see the seed."""
    from openrct2_gym.envs.feature_extractor import BuildHistoryExtractor
    from openrct2_gym.envs.obs_config import make_observation_space
    ex = BuildHistoryExtractor(make_observation_space())
    assert "target_family" in ex._cat_keys
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest openrct2_gym/tests/test_footprint.py -k "target_family" -q`
Expected: FAIL — `ImportError: cannot import name 'TARGET_FAMILY_N'`

- [ ] **Step 3: Add the constant and the space field**

In `openrct2_gym/envs/obs_config.py`, after the `LAST_PIECE_N` definition:

```python
# --- seed / requested footprint family -------------------------------------------
# The episode's requested family (see envs/footprint.py). Discrete so SB3 one-hots it
# like the other categoricals. Present from Phase 1 even while the family reward is
# off: an input that predicts nothing gets zeroed out by the network and has to be
# unlearned later, so the warm-start scaffold keys off it from the start.
from openrct2_gym.envs.footprint import FAMILY_N as TARGET_FAMILY_N  # noqa: E402
```

and inside `make_observation_space()`'s dict, after `"last_piece_type"`:

```python
        "target_family": gym.spaces.Discrete(TARGET_FAMILY_N),
```

- [ ] **Step 4: Add the attribute and the observation entry**

In `openrct2_gym/envs/openrct2_env.py`, in `__init__` next to the other episode state (near `self._warm_prefix_len = 0`):

```python
        # Requested footprint family for this episode (the "seed"). The curriculum
        # wrapper sets it before reset(); a bare env leaves it at 0, and the family
        # reward weights default to inert, so nothing changes for phases 1-2.
        self.target_family = 0
```

In the observation dict (near line 1791, beside `'current_direction'`):

```python
            'target_family': int(self.target_family),
```

- [ ] **Step 5: Let the extractor read it**

In `openrct2_gym/envs/feature_extractor.py:84`:

```python
        self._cat_keys = ["current_direction", "last_piece_type", "target_family"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `venv/bin/python -m pytest openrct2_gym/tests/test_footprint.py -q`
Expected: PASS, 10 passed

- [ ] **Step 7: Run the full suite**

Run: `venv/bin/python -m pytest openrct2_gym/tests/ -q`
Expected: PASS. If `test_feature_extractor.py` asserts an exact `features_dim` or an
exact `cat_in`, update that expectation — the categorical input grew by `FAMILY_N`
floats. That is a deliberate spec change, not a break.

- [ ] **Step 8: Commit**

```bash
git add openrct2_gym/envs/obs_config.py openrct2_gym/envs/openrct2_env.py \
        openrct2_gym/envs/feature_extractor.py openrct2_gym/tests/test_footprint.py
git commit -m "feat: target_family seed in the observation space"
```

---

### Task 3: Family-match reward — gate and bonus

**Files:**
- Modify: `openrct2_gym/envs/openrct2_env.py` (`RewardParams`, `_family_match`, `_calculate_reward`, `_qualifies`, episode_metrics)
- Modify: `openrct2_gym/envs/improved_phased_curriculum_wrapper.py` (`_validate_completion_first`, P6 params)
- Test: `openrct2_gym/tests/test_reward.py` (append)

**Interfaces:**
- Consumes: `footprint.family_match`, `footprint.classify_family`, `OpenRCT2Env.target_family`.
- Produces: `RewardParams.completion_family_floor` (default 1.0 = off), `RewardParams.R_family` (default 0.0), `RewardParams.qualify_requires_family` (default False), `RewardParams.family_turn_falloff` (5.0), `RewardParams.family_switch_falloff` (3.0); `OpenRCT2Env._family_match(params) -> float`; `episode_metrics["family_gate"]`, `["family_match"]`, `["family_hit"]`.

- [ ] **Step 1: Write the failing test**

Append to `openrct2_gym/tests/test_reward.py`:

```python
# ------------------------------- seed-conditioned family reward (Aug-9 redesign)
# The target now comes from the seed, so "more turns is always better" is WRONG for
# oval seeds. The both-directions economics test below is the guard whose absence
# allowed the S-bend exploit to survive for a week.

def _family_env(rows, family, params, excitement=6.0):
    env = _bare_env(history=_env_hist(rows))
    env.loop_completed = True
    env._phi_prev = 0.0
    env.reward_params = params
    env.target_family = family
    env._last_test_ok = True
    env.last_ride_excitement = excitement
    env._calculate_energy_margin = lambda: 10.0
    return env


def test_family_reward_fields_default_inert():
    p = RewardParams()
    assert p.completion_family_floor == 1.0
    assert p.R_family == 0.0
    assert p.qualify_requires_family is False


def test_family_gate_scales_completion_by_match():
    params = replace(RewardParams(), completion_family_floor=0.5, roundtrip_gain=0.0)
    oval_rows = [(4, 14, 14)] * 4 + [(0, 14, 14)] * 8
    hit = _family_env(oval_rows, 0, params)._calculate_reward(True, 0)
    miss = _family_env(oval_rows, 3, params)._calculate_reward(True, 0)   # asked winding
    assert hit > miss
    assert hit == pytest.approx(params.R_complete)          # perfect match, full pay


def test_family_bonus_requires_hit_and_quality():
    params = replace(RewardParams(), R_family=200.0, qualify_requires_family=True,
                     qualify_min_excitement=4.5, qualify_requires_test=True,
                     struct_height_target=0.0, struct_drop_target=0.0,
                     struct_length_target=0.0)
    oval = [(4, 14, 14)] * 4 + [(0, 14, 14)] * 8
    assert _family_env(oval, 0, params)._qualifies(params) is True
    assert _family_env(oval, 3, params)._qualifies(params) is False          # wrong family
    assert _family_env(oval, 0, params, excitement=3.0)._qualifies(params) is False


def test_oval_seed_beats_winding_build_from_step_one():
    """THE inversion test. With an oval requested, building winding must LOSE.
    Without this, 'always add more turns' creeps back in."""
    P6 = ImprovedPhasedCurriculumWrapper._phase_reward_params(6)
    oval_rows = _p6_mix(n_right=4, n_left=0)
    wind_rows = _p6_mix(n_right=6, n_left=6)
    oval_payout = _family_env(oval_rows, 0, P6)._calculate_reward(True, 0)
    wind_payout = _family_env(wind_rows, 0, P6)._calculate_reward(True, 0)
    assert oval_payout > wind_payout * 1.2


def test_winding_seed_beats_oval_build_from_step_one():
    P6 = ImprovedPhasedCurriculumWrapper._phase_reward_params(6)
    oval_rows = _p6_mix(n_right=4, n_left=0)
    wind_rows = _p6_mix(n_right=6, n_left=6)
    assert (_family_env(wind_rows, 3, P6)._calculate_reward(True, 0)
            > _family_env(oval_rows, 3, P6)._calculate_reward(True, 0) * 1.2)


def test_p6_params_switch_off_the_fixed_turn_target_and_balance_leg():
    """The target comes from the seed now; the old always-more-turns legs must be off
    or they fight it on oval seeds."""
    p6 = ImprovedPhasedCurriculumWrapper._phase_reward_params(6)
    assert p6.struct_w_turns == 0.0
    assert p6.struct_w_turn_balance == 0.0
    assert p6.completion_style_floor == 1.0
    assert p6.completion_family_floor == 0.5
    assert p6.R_family == 200.0
    assert p6.qualify_requires_family is True


def test_validate_completion_first_folds_the_family_floor():
    W = ImprovedPhasedCurriculumWrapper
    bad = replace(RewardParams(), completion_hill_floor=0.5,
                  completion_quality_floor=0.4, completion_family_floor=0.4,
                  R_roundtrip=100.0)
    with pytest.raises(AssertionError):
        W._validate_completion_first(bad, "test")      # 100 >= .5*.4*.4*1000 = 80
    W._validate_completion_first(replace(bad, completion_family_floor=1.0), "test")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest openrct2_gym/tests/test_reward.py -k "family" -q`
Expected: FAIL — `AttributeError: 'RewardParams' object has no attribute 'completion_family_floor'`

- [ ] **Step 3: Add the params**

In `openrct2_gym/envs/openrct2_env.py`, in `RewardParams` beside `completion_style_floor`:

```python
    # Seed-conditioned footprint family (Aug-9). The episode's requested family comes
    # from the observation, so the target is no longer fixed: on an oval seed, MORE
    # turns is wrong. completion_family_floor is the fraction of the (hill x length x
    # style)-gated payout a build earns when it ignores its seed entirely; the rest
    # ramps with family_match. 1.0 = no gating (phases 1-5 default).
    completion_family_floor: float = 1.0
    R_family: float = 0.0                    # discrete bonus on family hit + tested quality
    qualify_requires_family: bool = False    # qualified gate leg: built what was asked
    family_turn_falloff: float = 5.0         # turns outside the band before credit hits 0
    family_switch_falloff: float = 3.0       # switches outside the band before credit hits 0
```

- [ ] **Step 4: Add the match helper and wire the gate**

In `openrct2_gym/envs/openrct2_env.py`, import at the top beside the `track_pieces` import:

```python
from .footprint import classify_family, family_match
```

Add the helper next to `_turn_count`:

```python
    def _family_match(self, params):
        """How well this build matches the family the seed asked for, in [0, 1]."""
        hist = getattr(self.track_builder, "history", None) or []
        actions = [h.get("action") for h in hist]
        return family_match(actions, int(getattr(self, "target_family", 0)),
                            turn_falloff=params.family_turn_falloff,
                            switch_falloff=params.family_switch_falloff)

    def _family_hit(self):
        """True when the build lands exactly in the requested family."""
        hist = getattr(self.track_builder, "history", None) or []
        actions = [h.get("action") for h in hist]
        return classify_family(actions) == int(getattr(self, "target_family", 0))
```

In `_calculate_reward`, immediately after the style-gate block and before the
`self._last_gate_prequality = gate` line:

```python
            # Family gate (multiplicative, like length/quality/style): a build that
            # ignores its seed forfeits most of the completion payout. Multiplicative
            # because additive carrots have repeatedly lost to reliable alternatives
            # in this project.
            if params.completion_family_floor < 1.0:
                self._last_family_match = self._family_match(params)
                self._last_family_gate = (
                    params.completion_family_floor
                    + (1.0 - params.completion_family_floor) * self._last_family_match)
                gate *= self._last_family_gate
```

Reset both alongside the other gate resets (`self._last_style_gate = 0.0`, two places):

```python
        self._last_family_gate = 0.0
        self._last_family_match = 0.0
```

In `_qualifies`, after the `qualify_requires_test` leg:

```python
        if params.qualify_requires_family and not self._family_hit():
            return False
```

In the `episode_metrics` dict beside `'style_gate'`:

```python
                'family_gate': float(getattr(self, '_last_family_gate', 0.0)),
                'family_match': float(getattr(self, '_last_family_match', 0.0)),
                'family_hit': float(self._family_hit()),
                'target_family': float(getattr(self, 'target_family', 0)),
                'switch_count': float(__import__(
                    'openrct2_gym.envs.footprint', fromlist=['switch_count']
                ).switch_count([h.get("action") for h in
                                (getattr(self.track_builder, "history", None) or [])])),
```

Replace that last awkward inline import by adding `switch_count` to the top-level
import instead, and using `'switch_count': float(switch_count([...]))`.

- [ ] **Step 5: Fold the floor into the validator and update P6 params**

In `openrct2_gym/envs/improved_phased_curriculum_wrapper.py`, in
`_validate_completion_first`:

```python
            flat = (params.completion_hill_floor * params.completion_length_floor
                    * params.completion_quality_floor * params.completion_style_floor
                    * params.completion_family_floor * params.R_complete)
```

In the `phase >= 6` params block, change these fields:

```python
                completion_style_floor=1.0,       # superseded by the family gate
                completion_family_floor=0.5,      # ignore your seed -> forfeit half
                R_family=200.0,
                qualify_requires_family=True,
                struct_w_turns=0.0,               # target comes from the seed now
                struct_w_turn_balance=0.0,        # switches measure this better
```

and redistribute the freed struct weight (0.25 + 0.10) so the remaining legs still
sum to 1.0 — put it on length and single-drop:

```python
                struct_w_single_drop=0.35,
                struct_w_length=0.30,
```

Verify the sum: `0.35 + 0.15 (drop_runs) + 0.30 (length) + 0.10 (banked) + 0.05
(sbend) + 0.05 (steep, if present) = 1.00`. Run
`grep -n "struct_w_" openrct2_gym/envs/improved_phased_curriculum_wrapper.py` inside
the P6 block and adjust so the total is exactly 1.0.

- [ ] **Step 6: Pay the bonus in the terminal branch**

In `openrct2_gym/envs/openrct2_env.py`'s terminal `step()` branch, beside where
`R_qualify` is added:

```python
            if params.R_family > 0.0 and self._family_hit() and self._last_test_ok:
                if float(getattr(self, "last_ride_excitement", 0.0)) >= params.qualify_min_excitement:
                    reward += params.R_family
                    self._last_family_bonus = params.R_family
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `venv/bin/python -m pytest openrct2_gym/tests/test_reward.py -k "family" -q`
Expected: PASS, 7 passed

- [ ] **Step 8: Run the full suite**

Run: `venv/bin/python -m pytest openrct2_gym/tests/ -q`
Expected: PASS. Existing P6 spec tests that assert `struct_w_turns == 0.25`,
`struct_w_turn_balance == 0.10`, `completion_style_floor == 0.5`, or the qualified
predicate's turn/balance legs WILL fail — those are deliberate spec updates. Update
each to the new values and add a comment noting the seed now supplies the target.

- [ ] **Step 9: Commit**

```bash
git add openrct2_gym/envs/openrct2_env.py \
        openrct2_gym/envs/improved_phased_curriculum_wrapper.py \
        openrct2_gym/tests/test_reward.py
git commit -m "feat: family-match completion gate and bonus (seed-conditioned target)"
```

---

### Task 4: Dense family potential

**Files:**
- Modify: `openrct2_gym/envs/openrct2_env.py` (`RewardParams.w_family`, `_potential`)
- Modify: `openrct2_gym/envs/improved_phased_curriculum_wrapper.py` (P6 params)
- Test: `openrct2_gym/tests/test_reward.py` (append)

**Interfaces:**
- Consumes: `OpenRCT2Env._family_match` from Task 3.
- Produces: `RewardParams.w_family` (default 0.0); `episode_metrics["family_potential"]`.

- [ ] **Step 1: Write the failing test**

Append to `openrct2_gym/tests/test_reward.py`:

```python
def test_family_potential_defaults_off_and_leaves_early_phases_identical():
    assert RewardParams().w_family == 0.0
    for phase in (1, 2, 3, 4, 5):
        assert ImprovedPhasedCurriculumWrapper._phase_reward_params(phase).w_family == 0.0


def test_family_potential_rises_toward_the_requested_family():
    """Terminal-only shaping has been consistently too slow here (the style gate ran
    ~900k steps without reaching cold builds), so family progress pays per piece."""
    params = replace(RewardParams(), w_family=6.0)
    near = _bare_env(history=_env_hist([(4, 14, 14)] * 4))       # 4 turns: oval band
    far = _bare_env(history=_env_hist([(4, 14, 14)] * 12))       # 12 turns: way outside
    near.target_family = 0
    far.target_family = 0
    assert near._potential(params) > far._potential(params)


def test_family_potential_telescopes_as_a_state_function():
    """PBRS requires Phi to depend only on state, so place-then-remove must not pay."""
    params = replace(RewardParams(), w_family=6.0)
    env = _bare_env(history=_env_hist([(4, 14, 14)] * 3))
    env.target_family = 0
    before = env._potential(params)
    env.track_builder.history.append(
        {"action": 4, "position": [9, 0, 14], "next_position": [10, 0, 14]})
    env.track_builder.history.pop()
    assert env._potential(params) == pytest.approx(before)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest openrct2_gym/tests/test_reward.py -k "family_potential" -q`
Expected: FAIL — `AttributeError: 'RewardParams' object has no attribute 'w_family'`

- [ ] **Step 3: Add the weight and the potential term**

In `RewardParams`, beside `w_exc_feat`:

```python
    # Dense per-piece credit for progressing toward the requested family. Pure state
    # function of the build history -> telescopes, PBRS-clean. Zero in phases 1-5.
    w_family: float = 0.0
```

In `_potential`, beside the `w_exc_feat` term:

```python
        if params.w_family > 0.0:
            phi += params.w_family * self._family_match(params)
```

In `episode_metrics`:

```python
                'family_potential': float(
                    self.reward_params.w_family * self._family_match(self.reward_params)),
```

In the P6 params block add:

```python
                w_family=6.0,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest openrct2_gym/tests/test_reward.py -k "family" -q`
Expected: PASS, 10 passed

- [ ] **Step 5: Run the full suite**

Run: `venv/bin/python -m pytest openrct2_gym/tests/ -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add openrct2_gym/envs/openrct2_env.py \
        openrct2_gym/envs/improved_phased_curriculum_wrapper.py \
        openrct2_gym/tests/test_reward.py
git commit -m "feat: dense family-match potential (PBRS-clean)"
```

---

### Task 5: Curriculum — per-phase family sets and per-family tracking

**Files:**
- Modify: `openrct2_gym/envs/improved_phased_curriculum_wrapper.py`
- Test: `openrct2_gym/tests/test_warm_start.py` (append)

**Interfaces:**
- Consumes: `footprint.FAMILY_N`, `OpenRCT2Env.target_family`.
- Produces: `ImprovedPhasedCurriculumWrapper.PHASE_FAMILIES` (dict phase → tuple of family indices), `wrapper._sample_target_family() -> int`, `info["target_family"]`, `info["family_hit"]`, `info["family_hit_rate_{z}"]` for each active `z`.

- [ ] **Step 1: Write the failing test**

Append to `openrct2_gym/tests/test_warm_start.py`:

```python
# ------------------------------- per-phase family sets (Aug-9 seed conditioning)

def test_phase_family_sets_widen_with_the_track_budget():
    W = ImprovedPhasedCurriculumWrapper
    assert W.PHASE_FAMILIES[1] == ()          # 40 pieces: too tight to express shape
    assert W.PHASE_FAMILIES[2] == ()
    assert W.PHASE_FAMILIES[3] == (0, 1, 2)   # oval, spiral, out-and-back
    assert W.PHASE_FAMILIES[4] == (0, 1, 2, 3)
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


def test_step_info_reports_per_family_hit_rate(monkeypatch, tmp_path):
    wrapper, base = _wrapped(monkeypatch, tmp_path, initial_phase=6)
    info = _run_episode(wrapper)
    assert "target_family" in info
    assert "family_hit" in info
    assert any(k.startswith("family_hit_rate_") for k in info)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest openrct2_gym/tests/test_warm_start.py -k "family" -q`
Expected: FAIL — `AttributeError: type object 'ImprovedPhasedCurriculumWrapper' has no attribute 'PHASE_FAMILIES'`

- [ ] **Step 3: Implement the family sets and sampling**

In `improved_phased_curriculum_wrapper.py`, as a class attribute beside
`FLOOR_STYLE_MIN_TURNS`:

```python
    # Families the seed may request, per phase. Widens with the track budget: a
    # 40-piece build cannot express a serpentine. Phases 1-2 pin the seed to 0 so
    # the observation input is constant rather than noise while its reward is off.
    PHASE_FAMILIES = {
        1: (), 2: (), 3: (0, 1, 2), 4: (0, 1, 2, 3), 5: (0, 1, 2, 3, 4),
        6: (0, 1, 2, 3, 4),
    }
```

In `__init__`, beside the other per-worker state:

```python
        self._family_rng = random.Random()
        self.episode_family_results = {z: deque(maxlen=window_size)
                                       for z in range(FAMILY_N)}
```

Add `import random` and `from openrct2_gym.envs.footprint import FAMILY_N` at the top
if not already present.

Add the sampler:

```python
    def _sample_target_family(self):
        """The episode's seed: a family drawn from the phase's active set (0 when the
        phase has none, so the observation input stays constant rather than noisy)."""
        active = self.PHASE_FAMILIES.get(self.current_phase, ())
        return self._family_rng.choice(active) if active else 0
```

In `reset()`, before delegating to `self.env.reset()`:

```python
        self._get_base_env().target_family = self._sample_target_family()
```

- [ ] **Step 4: Track and report per-family results**

In `step()`, in the terminal block where `episode_qualified_results` is appended:

```python
            z = int(self._get_base_env().target_family)
            family_hit = bool(info.get('episode_metrics', {}).get('family_hit', 0.0))
            if cold:
                # Cold-only, like every other gate window: a scaffolded build inherits
                # its shape from the exemplar, so counting it would measure the
                # scaffold rather than the policy.
                self.episode_family_results[z].append(family_hit and qualified)
            info['target_family'] = z
            info['family_hit'] = float(family_hit)
            for fz, window in self.episode_family_results.items():
                if window:
                    info[f'family_hit_rate_{fz}'] = sum(window) / len(window)
```

The local holding the qualified result in that block is named `qualified`
(`improved_phased_curriculum_wrapper.py:971`), so the snippet above compiles as written.

- [ ] **Step 5: Point the P6 qualified predicate at the family**

In `_is_qualified`, replace the P6 turn/balance legs with the family check:

```python
        if self.current_phase >= 6:
            from openrct2_gym.envs.footprint import classify_family
            hist = getattr(base_env.track_builder, 'history', [])
            actions = [h.get('action') for h in hist]
            return (success
                    and getattr(base_env, '_last_test_ok', False)
                    and float(getattr(base_env, 'last_ride_excitement', 0.0)) >= 4.5
                    and classify_family(actions) == int(getattr(base_env, 'target_family', 0)))
```

Move the `classify_family` import to the module top rather than inside the method.

- [ ] **Step 6: Run tests to verify they pass**

Run: `venv/bin/python -m pytest openrct2_gym/tests/test_warm_start.py -k "family" -q`
Expected: PASS, 4 passed

- [ ] **Step 7: Run the full suite**

Run: `venv/bin/python -m pytest openrct2_gym/tests/ -q`
Expected: PASS. `test_p6_qualified_requires_variety_and_tested_excitement` will fail —
it encodes the old turns≥12/balance≥2 predicate. Rewrite it to assert the family
predicate instead: a build matching the requested family qualifies, one matching a
different family does not, and an untested or low-E build does not.

- [ ] **Step 8: Commit**

```bash
git add openrct2_gym/envs/improved_phased_curriculum_wrapper.py \
        openrct2_gym/tests/test_warm_start.py
git commit -m "feat: per-phase family sets, seed sampling, per-family hit tracking"
```

---

### Task 6: Warm-start filters exemplars by family

**Files:**
- Modify: `openrct2_gym/envs/warm_start.py` (`LoopRecord.family`, `LoopLibrary.pool`)
- Modify: `openrct2_gym/envs/improved_phased_curriculum_wrapper.py` (`_sample_warm_start`)
- Test: `openrct2_gym/tests/test_warm_start.py` (append)

**Interfaces:**
- Consumes: `footprint.classify_family`.
- Produces: `LoopRecord.family` (derived property, `int | None`); `LoopLibrary.pool(..., family=None)`.

- [ ] **Step 1: Write the failing test**

Append to `openrct2_gym/tests/test_warm_start.py`:

```python
def test_loop_record_exposes_its_family():
    oval = LoopRecord.from_actions([4, 0, 4, 0, 4, 0, 4], "scripted")
    winding = LoopRecord.from_actions([4, 4, 0, 3, 3, 0, 4, 4, 0, 3, 3, 0], "scripted")
    assert oval.family == 0
    assert winding.family == 3


def test_pool_filters_by_requested_family(tmp_path):
    """A spiral seed must be scaffolded by spiral exemplars, or the seed means nothing
    during the phases where its reward is still off."""
    lib = _lib(tmp_path)
    oval = [4, 0, 4, 0, 4, 0, 4] + [0] * 20
    winding = [4, 4, 0, 3, 3, 0, 4, 4, 0, 3, 3, 0] + [0] * 20
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest openrct2_gym/tests/test_warm_start.py -k "family" -q`
Expected: FAIL — `AttributeError: 'LoopRecord' object has no attribute 'family'`

- [ ] **Step 3: Add the property and the filter**

In `warm_start.py`, import `classify_family` from `footprint` at the top, and add
beside `turn_count`:

```python
    @property
    def family(self):
        """Footprint family index, or None when the shape fits no family. Derived, so
        every legacy record gets it on load with no schema migration."""
        return classify_family(self.actions)
```

In `pool()`, add the parameter `family=None` and apply it inside the best tier's
filter:

```python
                                        and (family is None or r.family == family)
```

Then, before the existing fallback tiers, add a family-specific fallback so the
scaffold never goes silent:

```python
            if family is not None:
                same_family = [r for r in fits if r.family == family]
                if same_family:
                    return same_family
```

- [ ] **Step 4: Pass the episode's family from the wrapper**

In `_sample_warm_start`, add to the P6 branch (and the P3–P5 branches once their
families are active):

```python
        family = int(getattr(self._get_base_env(), "target_family", 0)) \
            if self.PHASE_FAMILIES.get(self.current_phase) else None
```

and thread `family=family` into the `sample_plan(...)` call, then through
`WarmStartAnnealer.sample_plan` into its `library.pool(...)` call.

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/python -m pytest openrct2_gym/tests/test_warm_start.py -k "family" -q`
Expected: PASS, 7 passed

- [ ] **Step 6: Run the full suite**

Run: `venv/bin/python -m pytest openrct2_gym/tests/ -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add openrct2_gym/envs/warm_start.py \
        openrct2_gym/envs/improved_phased_curriculum_wrapper.py \
        openrct2_gym/tests/test_warm_start.py
git commit -m "feat: warm-start pool filters exemplars by requested family"
```

---

### Task 7: TensorBoard diagnostics

**Files:**
- Modify: `train.py` (metric forwarding block near line 601 and the curriculum key list near line 572)
- Test: `openrct2_gym/tests/test_train_wiring.py` (append)

**Interfaces:**
- Consumes: `episode_metrics` keys from Task 3/4 and `info` keys from Task 5.
- Produces: TB tags `rewards/family_gate`, `rewards/family_potential`, `structure/family_match`, `structure/switch_count`, `curriculum/target_family`, `curriculum/family_hit_rate_{0..4}`.

- [ ] **Step 1: Write the failing test**

Append to `openrct2_gym/tests/test_train_wiring.py`:

```python
def test_callback_logs_family_tags():
    """Every mechanism streams its own diagnostic; per-family hit rate is THE headline
    metric for this redesign."""
    from types import SimpleNamespace
    cb = T.ParallelCurriculumMaskableCallback(n_envs=1)
    cb.model = SimpleNamespace(target_kl=None, ent_coef=0.01, get_env=lambda: None)
    store = {}
    cb.model.logger = SimpleNamespace(
        name_to_value={}, record=lambda k, v, *a, **kw: store.__setitem__(k, v))
    cb.locals = {
        'dones': [True],
        'infos': [{'loop_completed': True, 'cold_start': True, 'learning_phase': 6,
                   'track_length': 70, 'current_distance': 0.0, 'collision_count': 0,
                   'target_family': 3, 'family_hit': 1.0, 'family_hit_rate_3': 0.42,
                   'episode_metrics': {
                       'track_length': 70, 'min_distance': 0.0,
                       'family_gate': 0.8, 'family_match': 0.9,
                       'family_potential': 5.4, 'switch_count': 4.0}}],
    }
    cb._on_step()
    assert store['rewards/family_gate'] == pytest.approx(0.8)
    assert store['structure/family_match'] == pytest.approx(0.9)
    assert store['rewards/family_potential'] == pytest.approx(5.4)
    assert store['structure/switch_count'] == pytest.approx(4.0)
    assert store['curriculum/family_hit_rate_3'] == pytest.approx(0.42)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest openrct2_gym/tests/test_train_wiring.py -k "family_tags" -q`
Expected: FAIL — `KeyError: 'rewards/family_gate'`

- [ ] **Step 3: Forward the tags**

In `train.py`, in the `if env_idx == 0:` metrics block beside the `style_gate` line:

```python
                        for _mk, _tag in (
                            ('family_gate', 'rewards/family_gate'),
                            ('family_potential', 'rewards/family_potential'),
                            ('family_match', 'structure/family_match'),
                            ('switch_count', 'structure/switch_count'),
                        ):
                            if _mk in info_metrics:
                                self.logger.record(_tag, info_metrics[_mk])
```

In the curriculum key loop (the tuple of keys near line 572), add:

```python
                    'target_family',
```

and after that loop, forward the per-family rates:

```python
                for _k, _v in _info.items():
                    if _k.startswith('family_hit_rate_'):
                        self.logger.record(f'curriculum/{_k}', _v)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest openrct2_gym/tests/test_train_wiring.py -k "family_tags" -q`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `venv/bin/python -m pytest openrct2_gym/tests/ -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add train.py openrct2_gym/tests/test_train_wiring.py
git commit -m "feat: TB diagnostics for family gate, match and per-family hit rate"
```

---

### Task 8: Thicken the thin families with live exemplars

**Files:**
- Modify: `seed_p5_exemplars.py` (add a `--family` flag)
- Run against a live OpenRCT2 instance on `htpc`

**Interfaces:**
- Consumes: `footprint.FAMILIES`, `LoopLibrary.add`.
- Produces: additional `source="scripted"` records in `logs/loop_library.jsonl` for families 1 (spiral, 874 existing) and 4 (serpentine, 61 existing).

This task has no unit test — it produces data by replaying candidate layouts against
the real game and keeping the ones that verifiably close and rate. Correctness is
checked by the archive query in Step 3.

- [ ] **Step 1: Add a family filter to the seeder**

In `seed_p5_exemplars.py`, add:

```python
    ap.add_argument("--family", type=int, default=None,
                    help="Only keep candidates that classify into this footprint "
                         "family (see envs/footprint.FAMILIES)")
```

and before a verified candidate is added to the library:

```python
    if args.family is not None and classify_family(placed) != args.family:
        continue
```

- [ ] **Step 2: Generate spiral and serpentine exemplars**

Spawn one spare instance and run (ports from `~/rl/spawn_instances.sh` output):

```bash
ssh htpc 'cd ~/rl/openrct2_gym && venv/bin/python seed_p5_exemplars.py \
    --port 8101 --family 1 --min-excitement 4.5'
ssh htpc 'cd ~/rl/openrct2_gym && venv/bin/python seed_p5_exemplars.py \
    --port 8101 --family 4 --min-excitement 4.5'
```

- [ ] **Step 3: Verify each family has usable scaffold material**

```bash
ssh htpc 'cd ~/rl/openrct2_gym && venv/bin/python -c "
from openrct2_gym.envs.warm_start import LoopLibrary
from collections import Counter
lib = LoopLibrary(\"logs/loop_library.jsonl\")
c = Counter(r.family for r in lib._records.values() if r.excitement >= 4.5)
print(sorted((k, v) for k, v in c.items() if k is not None))
"'
```

Expected: every family index 0–4 present with at least ~200 records. If serpentine
(index 4) cannot be lifted above ~50, drop it: set `PHASE_FAMILIES` to four families
and note it in the spec's risk section, per the design's stated fallback.

- [ ] **Step 4: Commit**

```bash
git add seed_p5_exemplars.py
git commit -m "feat: --family filter for exemplar seeding (thicken spiral/serpentine)"
```

---

### Task 9: Tag, archive, and launch the fresh run

**Files:**
- No source changes. Repo tag + archived artifacts + a new training run.

**Interfaces:**
- Consumes: everything above.
- Produces: git tag `v1-single-recipe-22M`, `archives/v1_single_recipe_22M/` on both hosts, a running seed-conditioned training job.

- [ ] **Step 1: Tag the pre-redesign state**

The observation change makes every earlier checkpoint unloadable, so the tag is the
only way back to a runnable pairing of code and weights.

```bash
git tag -a v1-single-recipe-22M -m "Single-recipe policy, 22M steps: last commit whose observation space matches the pre-seed checkpoints"
git push origin v1-single-recipe-22M
```

- [ ] **Step 2: Archive the checkpoint on htpc**

```bash
ssh htpc 'cd ~/rl/openrct2_gym && mkdir -p archives/v1_single_recipe_22M && \
  cp $(ls -t logs_parallel_curriculum_masked_20envs/*_steps.zip | head -1) \
     archives/v1_single_recipe_22M/ && \
  cp $(ls -t logs_parallel_curriculum_masked_20envs/*_vecnormalize.pkl | head -1) \
     archives/v1_single_recipe_22M/ && \
  cp logs/loop_library.jsonl archives/v1_single_recipe_22M/loop_library_snapshot.jsonl && \
  ls -la archives/v1_single_recipe_22M/'
```

- [ ] **Step 3: Stop the old run and deploy**

```bash
ssh htpc 'pkill -f "python train.py --ports"'
sleep 6
ssh htpc 'cd ~/rl/openrct2_gym && git pull origin main && git log --oneline -1'
```

- [ ] **Step 4: Launch fresh training**

No `--model-path`: the observation change means starting from scratch. Keep the
library — it is the scaffold's foundation and survives the obs change intact.

```bash
ssh htpc 'cd ~/rl/openrct2_gym && nohup venv/bin/python train.py \
  --ports 8081,8082,8083,8084,8085,8086,8087,8088,8089,8090,8091,8092,8093,8094,8095,8096,8097,8098,8099,8100 \
  --timesteps 3000000 --disable-eval --target-rollout 5120 \
  < /dev/null > logs/train_seedcond.log 2>&1 & echo launched'
```

- [ ] **Step 5: Verify it started and is learning**

```bash
ssh htpc 'grep -E "Created|Traceback" ~/rl/openrct2_gym/logs/train_seedcond.log | head -3'
```

Expected: `✅ Created 20 parallel environments using SubprocVecEnv`, no traceback.

- [ ] **Step 6: The Phase-3 early read (go/no-go)**

Roughly a day in, once `curriculum/phase` reaches 3, check the three active families:

```bash
ssh htpc 'cd ~/rl/openrct2_gym && venv/bin/python -c "
import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
acc = EventAccumulator(sorted(glob.glob(\"parallel_curriculum_masked_tensorboard/*/events.*\"), key=os.path.getmtime)[-1], size_guidance={\"scalars\": 0})
acc.Reload()
for z in range(3):
    tag = f\"curriculum/family_hit_rate_{z}\"
    if tag in acc.Tags()[\"scalars\"]:
        print(tag, [round(e.value, 3) for e in acc.Scalars(tag)[-10:]])
    else:
        print(tag, \"no samples yet\")
"'
```

Expected: non-zero and rising for families 0–2. If all three are flat at zero after
~500k steps in Phase 3, stop and reassess rather than spending the remaining days —
that is the design's declared go/no-go.

---

## Self-Review

**Spec coverage:** footprint descriptor → Task 1; seed in observation → Task 2; family
gate and bonus → Task 3; dense potential → Task 4; per-phase family sets, the
redefined qualified gate and per-family tracking → Task 5; warm-start family filtering
→ Task 6; diagnostics → Task 7; thin-family exemplar seeding → Task 8; tag, archive and
restart plus the Phase-3 early read → Task 9. The spec's success criteria are covered
by Task 7's `family_hit_rate_{z}` tag (metric), Task 9 Step 6 (early read) and the
existing `build_gallery.py` (deliverable — no change needed, it already selects and
replays library records).

**Placeholder scan:** no TBD/TODO; every code step carries the code. Class and
variable names were verified against the source while writing: the extractor is
`BuildHistoryExtractor` (`feature_extractor.py:36`) and the qualified local is
`qualified` (`improved_phased_curriculum_wrapper.py:971`). One `grep` remains, in
Task 3 Step 5, to confirm the P6 struct weights sum to 1.0 after redistribution —
that is an arithmetic check against live values, not an unwritten decision.

**Type consistency:** `family_match(actions, family_index, turn_falloff, switch_falloff)`
and `classify_family(actions)` are used with those exact signatures in Tasks 3, 5 and
6. `target_family` is an `int` attribute everywhere. `PHASE_FAMILIES` maps `int → tuple`
in both Task 5 and Task 6.

**Known deliberate test churn:** Task 3 Step 8 and Task 5 Step 7 update existing P6
spec tests that encode the old fixed-target predicate. These are spec changes, not
regressions, and each must be updated with a comment explaining that the seed now
supplies the target.
