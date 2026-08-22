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


def family_match(actions, family_index, turn_falloff, switch_falloff):
    """How well this build lands in the requested family, in [0, 1]. Graded so that
    getting nearer pays -- a pass/fail gate would never be discovered. The falloffs
    are deliberately required to prevent a stale default from silently degrading the
    grading."""
    _, tlo, thi, slo, shi = FAMILIES[family_index]
    dirs = turn_directions(actions)
    turns = len(dirs)
    switches = sum(1 for x, y in zip(dirs, dirs[1:]) if x != y)
    return 0.5 * (_band_score(turns, tlo, thi, turn_falloff)
                  + _band_score(switches, slo, shi, switch_falloff))


# Behaviour-descriptor bin edges, taken from the FAMILIES bands above so the diversity
# reward and every family metric share one source of truth. Upper edge of each band:
# turns  5 | 9 | 13 | open      (oval | spiral+out_and_back | winding | serpentine)
# switches 0 | 2 | 5  | open     (none | out_and_back | winding | serpentine)
TURN_EDGES = (5, 9, 13)
SWITCH_EDGES = (0, 2, 5)


def _bin(value, edges):
    for i, e in enumerate(edges):
        if value <= e:
            return i
    return len(edges)


def descriptor_cell(turns, switches):
    """(turn_bin, switch_bin) — the behaviour cell a build occupies.

    Coarser than an exact family match and deliberately so: the measured gap is that
    6,191 of 6,193 unaided builds had ZERO direction switches, so gaining a SINGLE
    switch must move the build to a new cell and pay, even though no family is matched
    yet. Exact family match is the destination; this is the gradient toward it.
    """
    return (_bin(turns, TURN_EDGES), _bin(switches, SWITCH_EDGES))


def classify_counts(turns, switches):
    """Index of the family a (turn count, direction-switch count) pair lands in, or None.

    Split out of classify_family so consumers holding only the counts -- the Phase-6
    variety exploration floor reads them from per-episode telemetry, where the action
    list is long gone -- classify against the SAME bands. A second copy of the bands is
    exactly how this branch shipped a mis-specified footprint four times.
    """
    for i, (_, tlo, thi, slo, shi) in enumerate(FAMILIES):
        if (turns >= tlo and (thi is None or turns <= thi)
                and switches >= slo and (shi is None or switches <= shi)):
            return i
    return None


def classify_family(actions):
    """Index of the family this build lands in, or None if it fits none.

    The bands do not tile the whole space (e.g. 11 turns with no alternation belongs
    to nothing). Returning None is deliberate: a build that matches no family is not
    a hit for any seed, and inventing a bin for it would hide that.
    """
    dirs = turn_directions(actions)
    turns = len(dirs)
    switches = sum(1 for x, y in zip(dirs, dirs[1:]) if x != y)
    return classify_counts(turns, switches)
