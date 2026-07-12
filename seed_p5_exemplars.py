#!/usr/bin/env python3
"""Seed the warm-start library with LIVE-TESTED Phase-5 quality exemplars.

Replays each generate_p5_candidates() skeleton on a free instance, walks the closure
tail, verifies the closure reproduces (the build_loop_library double-replay contract),
then RIDE-TESTS the loop and records it excitement-TAGGED — so it enters the P5 pool's
best/excited tiers immediately and snaps the self-imitation ratchet bar upward.

Written for the Jul-11 plateau: live E pinned at ~2.4 while the policy's own builds
never cross the remaining rating-cap legs together. Run with training STOPPED:
    python seed_p5_exemplars.py --port 8080 [--min-excitement 3.0]
"""
import argparse

from openrct2_gym.envs.api_controller import APIController
from openrct2_gym.envs.warm_start import LoopLibrary, LoopRecord, generate_p5_candidates
from build_loop_library import replay
from probe_measurements import poll_stats


def main():
    parser = argparse.ArgumentParser(description="Seed live-tested P5 quality exemplars")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--library", type=str, default="logs/loop_library.jsonl")
    parser.add_argument("--tail-max", type=int, default=16)
    parser.add_argument("--max-wait", type=float, default=45.0)
    parser.add_argument("--min-excitement", type=float, default=3.0,
                        help="Only record exemplars rating at least this (must beat the "
                             "plateau to move the ratchet)")
    parser.add_argument("--only-longer-than", type=int, default=0,
                        help="Skip skeletons at or under this length (re-runs test only "
                             "the new bigger families)")
    parser.add_argument("--min-single-drop", type=float, default=0.0,
                        help="Skip skeletons whose static max single drop is below this "
                             "(isolates a new hill family on incremental rounds)")
    args = parser.parse_args()

    api = APIController("localhost", args.port, verbose=0)
    if not api.connect():
        raise SystemExit(f"❌ Cannot connect to OpenRCT2 API on port {args.port}")
    api.set_game_speed(8)

    library = LoopLibrary(args.library)
    print(f"📚 {len(library)} loops before seeding; ratchet best so far: "
          f"{library.best_excitement(120):.2f}")

    tested = added = 0
    best_seen = 0.0
    for skeleton in generate_p5_candidates():
        if len(skeleton) <= args.only_longer_than:
            continue
        if LoopRecord.from_actions(skeleton, "probe").max_single_drop_z < args.min_single_drop:
            continue
        placed, closed, gain = replay(api, skeleton, tail_max=args.tail_max)
        if not closed:
            continue
        placed2, closed2, gain2 = replay(api, placed)          # reproducibility contract
        if not (closed2 and placed2 == placed):
            continue
        api.place_entrance_exit()
        api.start_ride_test()
        stats = poll_stats(api, max_wait=args.max_wait)
        if stats is None:
            print(f"  ⚠ {len(placed)}-piece exemplar never rated, skipped")
            continue
        tested += 1
        exc = float(stats.get("excitement", 0.0))
        best_seen = max(best_seen, exc)
        if exc < args.min_excitement:
            print(f"  ✗ {len(placed)} pieces rated E {exc:.2f} < {args.min_excitement}, "
                  f"not recorded")
            continue
        record = LoopRecord.from_actions(placed, source="scripted",
                                         max_gain=max(gain, gain2), excitement=exc)
        if library.add(record):
            added += 1
            print(f"  ✓ E {exc:.2f} / I {stats.get('intensity', 0):.2f} "
                  f"len={record.length} single_drop={record.max_single_drop_z:.0f}z "
                  f"steep={record.steep_drop_z:.0f}z")

    api.disconnect()
    print(f"\n📚 {tested} exemplars tested (best E {best_seen:.2f}), {added} recorded; "
          f"ratchet best now {library.best_excitement(120):.2f} "
          f"(pool bar {0.8 * library.best_excitement(120):.2f})")
    if added == 0:
        raise SystemExit("❌ NO-GO: no exemplar beat the bar -- the plateau needs a "
                         "different lever (check per-cap verdicts with probe_measurements.py)")


if __name__ == "__main__":
    main()
