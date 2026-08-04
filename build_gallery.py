#!/usr/bin/env python3
"""Build an inspection GALLERY park: replay a curated sample of library coasters at
spread-out stations in ONE park, test each so ratings display, and leave them standing.

Selection is the point (see select_gallery): top-by-excitement shows the agent's best
work, newest harvests show CURRENT behavior (cold drift included), qualified shows
builds that passed the full P6 gate. Run against a dedicated instance -- NOT a training
port (this script never calls resetEpisode/deleteAllRides, so rides accumulate):

    python build_gallery.py --port 8101 --top 3 --newest 3 --qualified 2

Then save the park (headless: `save_park gallery` on the instance's stdin console;
GUI: File > Save) and open it in any OpenRCT2 to fly around.
"""
import argparse
import time

from openrct2_gym.envs.api_controller import APIController
from openrct2_gym.envs.api_track_builder import APITrackBuilder
from openrct2_gym.envs.warm_start import LoopLibrary

STATION_LENGTH = 6
LEFT_TURNS = (1, 3, 21, 23, 29)
RIGHT_TURNS = (2, 4, 22, 24, 30)
QUALIFY_MIN_E, QUALIFY_MIN_TURNS, QUALIFY_MIN_BALANCE = 4.5, 12, 2


def turn_balance_of(actions):
    """min(left-family, right-family) turn pieces -- the env's P6 balance leg."""
    left = sum(1 for a in actions if a in LEFT_TURNS)
    right = sum(1 for a in actions if a in RIGHT_TURNS)
    return min(left, right)


def _is_qualified(rec):
    return (rec.excitement >= QUALIFY_MIN_E
            and rec.turn_count >= QUALIFY_MIN_TURNS
            and turn_balance_of(rec.actions) >= QUALIFY_MIN_BALANCE)


def select_gallery(records, top=3, newest=3, qualified=2):
    """Curate [(label, record)] without duplicates. `records` in library (recency)
    order. Priority on collision: best > newest > qualified (a coaster already shown
    for one reason should not eat another category's slot)."""
    picks, seen = [], set()

    def take(label, rec):
        if id(rec) not in seen:
            seen.add(id(rec))
            picks.append((label, rec))

    for rec in sorted(records, key=lambda r: -r.excitement)[:top]:
        take("best", rec)
    for rec in [r for r in reversed(records) if id(r) not in seen][:newest]:
        take("newest", rec)
    n_q = 0
    for rec in reversed(records):                 # newest qualified first
        if n_q >= qualified:
            break
        if id(rec) not in seen and _is_qualified(rec):
            take("qualified", rec)
            n_q += 1
    return picks


def gallery_slots(n, base=(61, 66, 14), dy=20):
    """Station starts spread along y. Builds extend WEST of x=61 and up to ~10 tiles
    laterally, so 20-tile y spacing keeps them disjoint on the flat whole-map scenario."""
    x0, y0, z0 = base
    slots = []
    for i in range(n):
        # alternate outward from the canonical row: 66, 86, 46, 106, 26, 126...
        off = (i + 1) // 2 * dy * (1 if i % 2 == 1 else -1) if i else 0
        y = y0 - off
        if not 2 <= y <= 250:
            raise SystemExit(f"slot {i} off map (y={y}); lower --dy or the sample size")
        slots.append((x0, y, z0))
    return slots


def _build_station(api, start):
    x, y, z = start
    d = 0
    for i in range(STATION_LENGTH):
        ttype = 2 if i == 0 else (1 if i == STATION_LENGTH - 1 else 3)
        resp = api.place_track_piece(x, y, z, d, ttype)
        if not resp.get("success"):
            return None
        ep = resp["payload"]["nextEndpoint"]
        x, y, z, d = ep["x"], ep["y"], ep["z"], ep["direction"]
    return [x, y, z], d


def build_one(api, rec, start):
    """Create a fresh ride (no demolition!), station at `start`, replay, test."""
    ride_id = api.create_ride()
    if ride_id is None:
        return None
    head = _build_station(api, start)
    if head is None:
        return None
    pos, direction = head
    builder = APITrackBuilder(api)
    for action in rec.actions:
        ok, pos, direction = builder.take_action(action, pos, direction)
        if not ok:
            return None
        if builder.history[-1].get("is_complete", False):
            break
    api.place_entrance_exit()
    api.start_ride_test()
    return ride_id


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--library", default="logs/loop_library.jsonl")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--newest", type=int, default=3)
    ap.add_argument("--qualified", type=int, default=2)
    ap.add_argument("--dy", type=int, default=20)
    ap.add_argument("--rate-wait", type=float, default=20.0,
                    help="seconds to wait at the end for tests to rate")
    args = ap.parse_args()

    records = list(LoopLibrary(args.library)._records.values())
    picks = select_gallery(records, args.top, args.newest, args.qualified)
    if not picks:
        raise SystemExit("nothing to show (empty library?)")
    slots = gallery_slots(len(picks), dy=args.dy)

    api = APIController("localhost", args.port, verbose=0)
    api.set_game_speed(8)
    manifest = []
    for (label, rec), slot in zip(picks, slots):
        ride_id = build_one(api, rec, slot)
        status = "OK" if ride_id is not None else "FAILED"
        manifest.append((label, rec, slot, ride_id, status))
        print(f"[{label:9s}] y={slot[1]:3d} len={rec.length:3d} "
              f"E={rec.excitement:4.2f} turns={rec.turn_count:2d} "
              f"balance={turn_balance_of(rec.actions)} -> {status}")

    print(f"\nwaiting {args.rate_wait:.0f}s for test trains to rate...")
    time.sleep(args.rate_wait)
    api.set_game_speed(1)
    print("\nManifest (fly to these y-rows in-game; station row x=61..56):")
    for label, rec, slot, ride_id, status in manifest:
        if status == "OK":
            stats = api.send_request({"endpoint": "getRideStats",
                                      "params": {"rideId": ride_id}})
            p = stats.get("payload") or {}
            e = p.get("excitement", -1)
            rated = f"live E {e:.2f}" if isinstance(e, (int, float)) and e > 0 else "unrated"
            print(f"  y={slot[1]:3d}  {label:9s}  {rec.length:3d} pieces  "
                  f"harvest-E {rec.excitement:4.2f}  {rated}")
    print("\nNow save the park and open it in any OpenRCT2 to inspect.")


if __name__ == "__main__":
    main()
