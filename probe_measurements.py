#!/usr/bin/env python3
"""Calibrate the P5 measured-caps constants against LIVE game measurements.

Replays library loops of assorted lengths on a free instance, ride-tests each, and
prints the measured stats next to the five wooden-RC rating caps (the constants in
OpenRCT2Env.WOODEN_CAP_*). Two things to read off the output:

  * metres-per-piece (rideLength / pieces): pins where the ~370m length cap sits in
    piece counts -- adjust struct_length_target / WOODEN_CAP_LENGTH_M if it surprises
    (the P5 design assumes the cap lands near ~50 pieces).
  * the mini-loop row should FAIL 4-5 caps and rate E~1.15 -- reproducing the observed
    Jul-9 plateau number validates the whole cap model.

Run with training STOPPED and the v0.3 plugin (getRideMeasurements) deployed:
    python probe_measurements.py --port 8080
"""
import argparse
import time

from openrct2_gym.envs.api_controller import APIController
from openrct2_gym.envs.openrct2_env import OpenRCT2Env
from openrct2_gym.envs.warm_start import LoopLibrary
from build_loop_library import replay


def poll_stats(api, max_wait=60.0):
    """Wait for POSITIVE ratings (the -0.01 sentinel means unrated)."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        resp = api.get_ride_stats()
        if resp.get("success"):
            p = resp.get("payload", {})
            if (p.get("excitement", 0) > 0 or p.get("intensity", 0) > 0
                    or p.get("nausea", 0) > 0):
                return p
        time.sleep(0.5)
    return None


def pick_candidates(library, targets=(24, 32, 40, 44)):
    """One record per target length (nearest by |length - target|), dedup'd."""
    records = list(library._records.values())
    picked, seen = [], set()
    for t in targets:
        best = min(records, key=lambda r: abs(r.length - t), default=None)
        if best is not None and best.actions not in seen:
            seen.add(best.actions)
            picked.append(best)
    return picked


def main():
    parser = argparse.ArgumentParser(description="Live calibration of the P5 rating-cap ramps")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--library", type=str, default="logs/loop_library.jsonl")
    parser.add_argument("--max-wait", type=float, default=60.0)
    args = parser.parse_args()

    api = APIController("localhost", args.port, verbose=0)
    if not api.connect():
        raise SystemExit(f"❌ Cannot connect to OpenRCT2 API on port {args.port}")
    api.set_game_speed(8)

    caps = (
        ("drop>=12z", lambda m: m.get("highestDropHeight", 0) >= OpenRCT2Env.WOODEN_CAP_DROP_Z),
        ("drops>=2", lambda m: m.get("numDrops", 0) >= OpenRCT2Env.WOODEN_CAP_NUM_DROPS),
        ("speed>=23", lambda m: m.get("maxSpeed", 0) >= OpenRCT2Env.WOODEN_CAP_SPEED_MPH),
        ("negG<=.1", lambda m: m.get("maxNegativeVerticalGs", 1.0) <= OpenRCT2Env.WOODEN_CAP_NEG_G_PASS),
        ("len>=370", lambda m: m.get("rideLength", 0) >= OpenRCT2Env.WOODEN_CAP_LENGTH_M),
    )

    library = LoopLibrary(args.library)
    print(f"📚 {len(library)} loops in {args.library}")
    for rec in pick_candidates(library):
        placed, closed, _ = replay(api, list(rec.actions))
        if not closed:
            print(f"  ✗ {rec.length}-piece loop did not close on replay, skipped")
            continue
        api.place_entrance_exit()
        api.start_ride_test()
        stats = poll_stats(api, max_wait=args.max_wait)
        if stats is None:
            print(f"  ⚠ {len(placed)}-piece loop never rated within {args.max_wait}s")
            continue
        resp = api.get_ride_measurements()
        if not resp.get("success"):
            raise SystemExit("❌ getRideMeasurements unavailable -- deploy the v0.3 plugin "
                             f"and restart the instance ({resp.get('error')})")
        m = resp["payload"]
        verdicts = "  ".join(f"{name}:{'PASS' if ok(m) else 'fail'}" for name, ok in caps)
        print(f"  {len(placed):3d} pieces | rideLength {m.get('rideLength', 0):6.1f} m "
              f"({m.get('rideLength', 0) / max(len(placed), 1):4.1f} m/pc) | "
              f"maxSpeed {m.get('maxSpeed', 0):4.1f} mph | drops {m.get('numDrops', 0)} "
              f"(highest {m.get('highestDropHeight', 0):.0f}z) | "
              f"negG {m.get('maxNegativeVerticalGs', 0):+.2f} | air {m.get('totalAirTime', 0):.2f}s")
        print(f"      E {stats.get('excitement', 0):.2f} / I {stats.get('intensity', 0):.2f} "
              f"/ N {stats.get('nausea', 0):.2f}  |  {verdicts}")

    api.disconnect()


if __name__ == "__main__":
    main()
