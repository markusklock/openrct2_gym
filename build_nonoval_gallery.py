#!/usr/bin/env python3
"""Build a park holding every NON-OVAL coaster the policy built UNAIDED.

These are the rare ones: across ~24,000 unaided builds since the forced-exploration
restart, a handful came out as a named non-oval family instead of the default oval.
They are the only direct evidence that the policy CAN initiate a different shape, so
they are worth being able to look at.

Selection is deliberately strict -- a build only qualifies if it is:
  * source == "harvest_cold"  (not a scaffolded or primed episode), AND
  * prefix_len == 0           (no replayed opening whatsoever), AND
  * classify_family(...) != 0 (a NAMED non-oval family, not merely unclassifiable)

Run against a dedicated instance, never a training port. Rides are replayed at
spread-out stations reusing build_gallery's proven plumbing, tested so ratings
display, then the park is saved via the plugin's execLegacy console bridge (headless
instances have no writable stdin, so this is the only save route).

    python build_nonoval_gallery.py --port 8101 --save-name nonoval_gallery
"""
import argparse
import time

from build_gallery import build_one, gallery_slots, turn_balance_of
from openrct2_gym.envs import footprint
from openrct2_gym.envs.api_controller import APIController
from openrct2_gym.envs.warm_start import LoopLibrary


def select_unaided_nonoval(records, since_ts=0.0):
    out = []
    for rec in records:
        if getattr(rec, "source", "") != "harvest_cold":
            continue
        if getattr(rec, "prefix_len", 0) != 0:
            continue
        if getattr(rec, "ts", 0.0) < since_ts:
            continue
        fam = footprint.classify_family(rec.actions)
        if fam is None or fam == 0:
            continue
        out.append((fam, rec))
    out.sort(key=lambda p: getattr(p[1], "ts", 0.0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--library", default="logs/loop_library.jsonl")
    ap.add_argument("--since-ts", type=float, default=0.0,
                    help="only records at/after this unix ts (0 = whole library)")
    ap.add_argument("--min-excitement", type=float, default=0.0,
                    help="only show builds rated at least this. The agent produces\n                         good AND poor examples of the same shape (measured: spirals\n                         span E 1.36-6.19), so an unfiltered gallery misrepresents\n                         what it can do in both directions.")
    ap.add_argument("--limit", type=int, default=0,
                    help="show only the best N by excitement (0 = all). The slot fan\n                         alternates outward and runs off the map past ~6 rides, so a\n                         large sample needs this rather than a threshold hunt.")
    ap.add_argument("--dy", type=int, default=20)
    ap.add_argument("--base-y", type=int, default=66,
                    help="centre row for the slot fan. Slots alternate outward from "
                         "here, so with many picks the outermost can reach the map "
                         "edge and its build fails; raise this to shift the fan.")
    ap.add_argument("--rate-wait", type=float, default=30.0)
    ap.add_argument("--save-name", default="nonoval_gallery")
    ap.add_argument("--keep-existing", action="store_true",
                    help="do NOT clear rides already on the instance")
    args = ap.parse_args()

    records = list(LoopLibrary(args.library)._records.values())
    picks = select_unaided_nonoval(records, args.since_ts)
    picks = [q for q in picks
             if getattr(q[1], 'excitement', 0.0) >= args.min_excitement]
    if args.limit:
        picks = sorted(picks, key=lambda q: -getattr(q[1], 'excitement', 0.0))
        picks = picks[:args.limit]
    if not picks:
        raise SystemExit("no unaided non-oval builds found")
    print("unaided non-oval builds found: %d" % len(picks))

    slots = gallery_slots(len(picks), base=(61, args.base_y, 14), dy=args.dy)
    api = APIController("localhost", args.port, verbose=0)

    if not args.keep_existing:
        api.send_request({"endpoint": "deleteAllRides", "params": {}})
        print("cleared existing rides")

    api.set_game_speed(8)
    manifest = []
    for (fam, rec), slot in zip(picks, slots):
        ride_id = build_one(api, rec, slot)
        status = "OK" if ride_id is not None else "FAILED"
        manifest.append((fam, rec, slot, ride_id, status))
        print("[%-13s] y=%3d len=%3d harvestE=%4.2f turns=%2d balance=%s -> %s"
              % (footprint.FAMILIES[fam][0], slot[1], rec.length, rec.excitement,
                 rec.turn_count, turn_balance_of(rec.actions), status))

    print("\nwaiting %.0fs for test trains to rate..." % args.rate_wait)
    time.sleep(args.rate_wait)
    api.set_game_speed(1)

    print("\nManifest (fly to these y-rows in-game; stations at x=61 heading west):")
    for fam, rec, slot, ride_id, status in manifest:
        if status != "OK":
            print("   y=%3d  FAILED TO BUILD" % slot[1])
            continue
        stats = api.send_request({"endpoint": "getRideStats",
                                  "params": {"rideId": ride_id}})
        p = stats.get("payload") or {}
        e = p.get("excitement", -1)
        live = "live E %.2f" % e if isinstance(e, (int, float)) and e > 0 else "unrated"
        print("   y=%3d  %-13s %3d pieces  harvestE=%.2f  %s  (port %s)"
              % (slot[1], footprint.FAMILIES[fam][0], rec.length, rec.excitement,
                 live, getattr(rec, "port", "?")))

    # Headless instances have no writable stdin, so the console bridge is the only
    # route to save_park. Fails loudly rather than leaving a silently unsaved park.
    resp = api.send_request({"endpoint": "execLegacy",
                             "params": {"command": "save_park %s" % args.save_name}})
    if not resp.get("success"):
        raise SystemExit("SAVE FAILED: %s" % resp.get("error"))
    print("\nsaved park: %s.park" % args.save_name)


if __name__ == "__main__":
    main()
