"""Gallery selection/layout logic (server-free). The live builder in build_gallery.py
is thin plumbing over the verified APIController/APITrackBuilder; what must be pinned
is WHICH coasters get shown and WHERE, so the inspection sample is honest:
top-by-excitement (the best work), newest harvests (current behavior, drift included),
and qualified builds (the P6 gate passed), deduped, at non-overlapping station slots."""
import pytest

from openrct2_gym.envs.warm_start import LoopRecord
from build_gallery import gallery_slots, select_gallery, turn_balance_of


def _rec(actions, excitement=0.0, source="harvest"):
    return LoopRecord.from_actions(actions, source=source, excitement=excitement)


WIND = [10, 9, 6, 4, 0, 3, 3, 0, 4, 4, 3, 0, 4, 3] + [0] * 30 + [4] * 4  # balanced turns
RECT = [10, 9, 6] + [0] * 20 + [4] * 4                                    # single-handed


def test_turn_balance_of_matches_env_definition():
    assert turn_balance_of(WIND) == min(
        sum(1 for a in WIND if a in (1, 3, 21, 23, 29)),
        sum(1 for a in WIND if a in (2, 4, 22, 24, 30)))
    assert turn_balance_of(RECT) == 0


def test_select_gallery_mixes_best_newest_and_qualified():
    older = [_rec(RECT + [0] * i, excitement=6.0 + i * 0.01) for i in range(3)]
    newest = [_rec(WIND + [0] * i, excitement=3.0) for i in range(3)]
    qualified = _rec([10, 9, 6] + [4, 0, 3] * 8 + [0] * 10, excitement=4.8)
    records = older + [qualified] + newest        # library order == recency order
    picks = select_gallery(records, top=2, newest=2, qualified=1)
    labels = [lab for lab, _ in picks]
    chosen = [r for _, r in picks]
    assert labels.count("best") == 2 and labels.count("newest") == 2
    assert labels.count("qualified") == 1
    assert older[2] in chosen and older[1] in chosen          # highest-E pair
    assert newest[-1] in chosen and newest[-2] in chosen      # most recent pair
    assert qualified in chosen
    assert len({id(r) for r in chosen}) == len(chosen)        # no duplicates


def test_select_gallery_qualified_needs_all_three_legs():
    no_balance = _rec(RECT + [4] * 9, excitement=5.0)          # 13 turns, one-handed
    low_e = _rec([10, 9, 6] + [4, 0, 3] * 8, excitement=3.0)   # wound but E < 4.5
    good = _rec([10, 9, 6] + [4, 0, 3] * 8, excitement=4.6)
    picks = select_gallery([no_balance, low_e, good], top=0, newest=0, qualified=3)
    assert [r for _, r in picks] == [good]


def test_select_gallery_dedups_across_categories():
    star = _rec([10, 9, 6] + [4, 0, 3] * 8 + [0] * 5, excitement=6.5)
    picks = select_gallery([star], top=2, newest=2, qualified=2)
    assert len(picks) == 1                                     # one record, one slot


def test_gallery_slots_spaced_and_on_map():
    slots = gallery_slots(6, base=(61, 66, 14), dy=20)
    ys = [y for _, y, _ in slots]
    assert len(slots) == 6 and len(set(ys)) == 6
    assert all(abs(a - b) >= 20 for a, b in zip(ys, ys[1:]))
    assert all(2 <= y <= 250 for y in ys)                      # stays on the flat map
    assert all(x == 61 and z == 14 for x, _, z in slots)
