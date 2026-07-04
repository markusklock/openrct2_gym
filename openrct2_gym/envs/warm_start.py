"""Warm-start reverse curriculum: loop library + annealer.

Discovery, not reward magnitude, is what stalled Phase 1 (7 completions in 31k episodes,
then entropy collapse): a ~12-piece exact docking sequence is never sampled by a near-
deterministic policy. The fix is backward chaining: at reset the env pre-places the first
(L - k) pieces of a KNOWN completable loop and the agent finishes the last k, so completions
(and their +R_complete) flow from episode 1; k anneals upward on frontier success until
episodes degenerate to cold starts and the scaffold disappears.

This module is pure Python (no API, no gym imports) so everything here is server-free
testable. The env replays ``WarmStartPlan.prefix`` at reset; the curriculum wrapper owns a
per-worker ``WarmStartAnnealer`` and a process-shared ``LoopLibrary`` JSONL file.
"""
import json
import os
import random
from collections import deque
from dataclasses import dataclass


CHAIN_ACTIONS = (9, 10)          # matches openrct2_env / api_track_builder chain-lift actions

# Static per-action z geometry (live-verified via the base-z offset probes): descents drop
# by their span, ascents climb by theirs. Lets the pool grade loops by real drop height and
# lets template generators balance climbs against descents without touching the game.
ACTION_DROP_Z = {6: 2, 8: 8, 12: 1, 14: 1, 27: 4, 28: 4}
ACTION_CLIMB_Z = {5: 2, 7: 8, 9: 2, 10: 1, 11: 1, 13: 1, 25: 4, 26: 4}


@dataclass(frozen=True)
class LoopRecord:
    """One verified/harvested closing action sequence (the last action closes the circuit)."""
    actions: tuple
    length: int
    chain_count: int
    max_gain: float              # peak z above the start height (0 for flat loops)
    drop_z: float                # total z dropped over descent pieces (static per-action geometry)
    source: str                  # "scripted" | "harvest"

    @staticmethod
    def from_actions(actions, source, max_gain=0.0):
        acts = tuple(int(a) for a in actions)     # coerce numpy ints -> json-serializable
        return LoopRecord(
            actions=acts,
            length=len(acts),
            chain_count=sum(1 for a in acts if a in CHAIN_ACTIONS),
            max_gain=float(max_gain),
            drop_z=float(sum(ACTION_DROP_Z.get(a, 0) for a in acts)),
            source=str(source),
        )


class LoopLibrary:
    """Dedup'd pool of completable loops, persisted as JSONL.

    Shared by 8-20 SubprocVecEnv workers WITHOUT locking: appends are single lines under
    PIPE_BUF (atomic on POSIX, same pattern as closing_probe.jsonl), and readers tolerate
    any corrupt/partial line by skipping it. All I/O is best-effort -- library problems
    must degrade to cold starts, never kill a worker.
    """

    # Growth bound: dedup alone does not cap a long run's harvest stream (every distinct
    # closing variant is a new record). Capped PER CLASS (flat/hill x small/big): a single
    # global cap would be first-come-forever -- Phase-1 mini-loop floods would fill every
    # slot and silently lock later phases' hill/big discoveries out of the pool.
    MAX_RECORDS_PER_CLASS = 250
    BIG_LEN = 25                 # loops this long are P3/P4 scaffold material (own cap class)

    def __init__(self, path="logs/loop_library.jsonl"):
        self.path = path
        self._records = {}                 # actions tuple -> LoopRecord
        self._calls_since_refresh = 0
        self.load()

    def __len__(self):
        return len(self._records)

    def load(self):
        """(Re)read the file, dedup on the action sequence; returns the pool size."""
        records = {}
        try:
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        rec = LoopRecord.from_actions(
                            d["actions"], d.get("source", "harvest"), d.get("max_gain", 0.0))
                    except (ValueError, KeyError, TypeError):
                        continue           # corrupt/partial line -> skip
                    if rec.length > 0:
                        records[rec.actions] = rec
        except OSError:
            pass                           # no file yet -> empty pool
        self._records = records
        return len(records)

    def maybe_refresh(self, every_n_calls=200):
        """Cheap periodic reload so a worker picks up loops other workers harvested."""
        self._calls_since_refresh += 1
        if self._calls_since_refresh >= every_n_calls:
            self._calls_since_refresh = 0
            self.load()

    @classmethod
    def _class_key(cls, record):
        return (record.chain_count >= 1, record.length >= cls.BIG_LEN)

    def add(self, record):
        """Dedup + per-class growth cap, then append one JSONL line. Returns True if stored.

        The cap bounds the HARVEST flood only: curated scripted seeds are finite by
        construction and are the pool's backbone -- a cap-saturated library must never
        refuse them (nor do they consume the harvest budget)."""
        if record is None or record.length == 0 or record.actions in self._records:
            return False
        if record.source != "scripted":
            key = self._class_key(record)
            n_class = sum(1 for r in self._records.values()
                          if r.source != "scripted" and self._class_key(r) == key)
            if n_class >= self.MAX_RECORDS_PER_CLASS:
                return False
        self._records[record.actions] = record
        try:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            line = json.dumps({
                "actions": list(record.actions), "length": record.length,
                "chain_count": record.chain_count, "max_gain": record.max_gain,
                "drop_z": record.drop_z, "source": record.source,
            }) + "\n"
            with open(self.path, "a") as f:
                f.write(line)
        except OSError:
            pass                           # in-memory record kept; persistence is best-effort
        return True

    def pool(self, phase, max_len, min_chains=1, min_len=0, min_drop_z=0):
        """Loops usable this episode: must fit the track budget with margin for the suffix
        search. Phase >= 2 prefers loops matching ALL the phase's structure criteria
        (chains, length, drop height), degrading tier by tier (all-criteria -> enough
        chains -> any hill -> everything) so the scaffold never silently turns off."""
        fits = [r for r in self._records.values() if r.length <= max_len - 2]
        if phase >= 2:
            best = [r for r in fits if (r.chain_count >= min_chains
                                        and r.length >= min_len
                                        and r.drop_z >= min_drop_z)]
            if best:
                return best
            chained = [r for r in fits if r.chain_count >= min_chains]
            if chained:
                return chained
            hills = [r for r in fits if r.chain_count >= 1]
            if hills:
                return hills
        return fits

    @staticmethod
    def record_from_history(history, source="harvest"):
        """A LoopRecord from a COMPLETED track_builder.history, else None.

        max_gain is measured against the first entry's start height (== station height,
        since builds start at the station), keeping this module env-independent.
        """
        if not history or not history[-1].get("is_complete"):
            return None
        try:
            actions = [int(h["action"]) for h in history]
            base_z = history[0]["position"][2]
            max_gain = max(float(h["next_position"][2] - base_z) for h in history)
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        return LoopRecord.from_actions(actions, source, max_gain=max(max_gain, 0.0))


@dataclass
class WarmStartPlan:
    """One episode's scaffold: replay ``prefix`` at reset, agent builds the last ``k``."""
    prefix: list
    k: int                       # suffix length the agent must build (0 for cold)
    loop_len: int                # 0 for cold
    cold: bool


class WarmStartAnnealer:
    """Per-worker backward-chaining schedule over the loop library.

    k (the suffix the agent must build) is sampled frontier-biased: k = k_max with p=0.5,
    else Uniform{1..k_max} -- the frontier carries the learning signal while the uniform
    half keeps dense easy successes anchoring the critic. The frontier (outcomes at
    k >= k_max - 1) promotes k_max by +2 at >= promote_rate over promote_n samples and
    demotes by -1 at <= demote_rate (slow down, fast up would oscillate). Once
    k_max - 1 exceeds every pool loop's length the frontier starves and k_max freezes:
    the anneal self-limits with episodes mostly cold -- the intended end state.
    """

    def __init__(self, k_init=3, p_cold=0.25, frontier_window=30, promote_n=20,
                 promote_rate=0.60, demote_rate=0.15, k_floor=3, rng=None):
        self.k_init = int(k_init)
        self.k_max = int(k_init)
        self.k_floor = int(k_floor)
        self.base_p_cold = float(p_cold)
        self.frontier = deque(maxlen=frontier_window)
        self.promote_n = int(promote_n)
        self.promote_rate = float(promote_rate)
        self.demote_rate = float(demote_rate)
        self._rng = rng if rng is not None else random.Random()

    @property
    def frontier_rate(self):
        """Success rate over the current frontier window (None while empty) -- the exact
        number promotion is judged on; exposed so training runs are not blind to it."""
        if not self.frontier:
            return None
        return sum(self.frontier) / len(self.frontier)

    @property
    def p_cold(self):
        """Cold-episode probability, rising with competence so training mass shifts onto
        the true task as the scaffold anneals (floor keeps the cold-only gates fed)."""
        if self.k_max >= 16:
            return max(self.base_p_cold, 0.50)
        if self.k_max >= 8:
            return max(self.base_p_cold, 0.35)
        return self.base_p_cold

    @staticmethod
    def _cold_plan():
        return WarmStartPlan(prefix=[], k=0, loop_len=0, cold=True)

    def sample_plan(self, library, phase, max_track_length, min_chains=1, min_len=0, min_drop_z=0):
        """The episode's warm-start plan. Cold when the die says so, the pool is empty,
        or the sampled k has annealed past the loop length (natural end of the scaffold)."""
        pool = (library.pool(phase, max_track_length, min_chains=min_chains,
                             min_len=min_len, min_drop_z=min_drop_z)
                if library is not None else [])
        if not pool or self._rng.random() < self.p_cold:
            return self._cold_plan()
        rec = pool[self._rng.randrange(len(pool))]
        k_hi = min(self.k_max, rec.length)
        k = k_hi if self._rng.random() < 0.5 else self._rng.randint(1, k_hi)
        if k >= rec.length:
            return self._cold_plan()
        return WarmStartPlan(prefix=list(rec.actions[:rec.length - k]),
                             k=k, loop_len=rec.length, cold=False)

    def record_outcome(self, plan, success):
        """Feed an episode outcome to the frontier; promote/demote k_max when it fills."""
        if plan is None or plan.cold or plan.k < self.k_max - 1:
            return
        self.frontier.append(bool(success))
        if len(self.frontier) < self.promote_n:
            return
        rate = sum(self.frontier) / len(self.frontier)
        if rate >= self.promote_rate:
            self.k_max += 2
            self.frontier.clear()
        elif rate <= self.demote_rate:
            self.k_max = max(self.k_floor, self.k_max - 1)
            self.frontier.clear()

    def on_phase_change(self, new_phase):
        """New phase == new target skill (e.g. hill loops in P2): restart the anneal."""
        self.k_max = self.k_init
        self.frontier.clear()


# --------------------------------------------------------------- candidate templates
# Live-verified racetrack family (probe run on port 8080, Jun 2026): from the post-station
# head [55,66,14] dir 0, `[0]*p + [t,t] + [0]*(7+p) + [t,t]` lands the head back on the
# station row heading the entry direction east of the staging tile; straights then dock at
# [62,66,14]. t=4 (right 3-tile turn, detour via y=69) or t=3 (left, via y=63).

def generate_candidates():
    """Racetrack skeletons (without the closing tail -- the library script replays each,
    walks straights toward the dock, and records the FULL placed sequence on closure)."""
    out = []
    for t in (4, 3):
        for p in range(0, 4):
            out.append([0] * p + [t, t] + [0] * (7 + p) + [t, t])
    return out


def generate_big_candidates():
    """Big-loop skeletons for the P3/P4 scale-up scaffold, in two height-balanced families:

      * tall-25:  climb [10,9,9,9,13]   (+8 z) / descend [12,6,6,6,14]  (-8 z) -- 10 east tiles
      * steep-60: climb [10,9,9,9,9,13] (+10 z) / descend [12,27,28,14] (-10 z) -- 10 east tiles
        (the steep family exercises the 60-degree pieces, unusable before the base-z fix)

    East leg = 7 + p (live-verified racetrack geometry) so the tallest blocks need p >= 3;
    summit straights (mid) stretch the loop toward the P3/P4 length targets; the closure
    scan supplies the west tail. Every candidate is climb/drop balanced (net z 0) so the
    second U-turn is back at station height.
    """
    out = []
    families = (
        ([10, 9, 9, 9, 13], [12, 6, 6, 6, 14]),
        ([10, 9, 9, 9, 9, 13], [12, 27, 28, 14]),
    )
    for t in (4, 3):
        for climb, descent in families:
            block = len(climb) + len(descent)
            for p in range(3, 7):
                east = 7 + p
                for mid in (0, 1, 2, 4):
                    rest = east - block - mid
                    if rest < 0:
                        continue
                    out.append([0] * p + [t, t]
                               + climb + [0] * mid + descent + [0] * rest
                               + [t, t])
    return out


def generate_hill_candidates():
    """Racetrack skeletons with a balanced chain climb on the east leg for the Phase-2 pool.

    Two families:
      * 2-chain: [10, 9, 13] climb (chain gain 3) + [12, 6, 14] descent -- 6 east tiles.
      * 3-chain: [10, 9, 9, 13] climb (chain gain 5) + [12, 6, 6, 14] descent -- 8 east
        tiles; the ONLY family whose completions can satisfy stage 2.3's 3-chain gate.
    Net z is 0 in both, so the second U-turn is back at station height.
    """
    out = []
    for t in (4, 3):
        for p in range(0, 2):
            east = 7 + p                               # east-leg tiles needed (live-verified)
            for mid in range(0, 2):
                rest = east - 6 - mid                  # 2-chain hill block consumes 6 tiles
                if rest < 0:
                    continue
                out.append([0] * p + [t, t]
                           + [10, 9, 13] + [0] * mid + [12, 6, 14] + [0] * rest
                           + [t, t])
        for p in range(1, 3):                          # 3-chain block needs east >= 8 -> p >= 1
            east = 7 + p
            for mid in range(0, 2):
                rest = east - 8 - mid                  # 3-chain hill block consumes 8 tiles
                if rest < 0:
                    continue
                out.append([0] * p + [t, t]
                           + [10, 9, 9, 13] + [0] * mid + [12, 6, 6, 14] + [0] * rest
                           + [t, t])
    return out
