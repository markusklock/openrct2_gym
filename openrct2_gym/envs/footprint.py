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
