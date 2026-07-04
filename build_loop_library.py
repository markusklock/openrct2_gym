#!/usr/bin/env python3
"""Seed the warm-start loop library with LIVE-VERIFIED closable loops.

Replays each racetrack candidate (warm_start.generate_candidates / _hill_candidates)
against a live OpenRCT2 instance, walks straights toward the dock until the API reports
isCircuitComplete, and records the FULL placed sequence. A record enters the library only
after a clean SECOND replay from a fresh reset closes again (probe data shows 3 distinct
pre-close head states across real closures -- never trust paper-derived sequences).

Run once per map with a free instance (no training on the port):
    python build_loop_library.py --port 8080            # flat loops (Phase 1)
    python build_loop_library.py --port 8080 --hill     # chain-hill loops (Phase 2 pool)
"""
import argparse

from openrct2_gym.envs.api_controller import APIController
from openrct2_gym.envs.api_track_builder import APITrackBuilder
from openrct2_gym.envs.warm_start import (
    LoopLibrary, LoopRecord, generate_candidates, generate_hill_candidates,
)

STATION_START = (61, 66, 14)   # must match OpenRCT2Env.reset()
STATION_LENGTH = 6
STATION_Z = 14


def _reset(api):
    payload = api.reset_episode(
        station_length=STATION_LENGTH, start=STATION_START, start_direction=0)
    if payload is None:
        raise SystemExit("❌ resetEpisode failed or unsupported by the plugin on this port "
                         "(is another process training on it?)")
    e = payload["finalEndpoint"]
    return [e["x"], e["y"], e["z"]], e["direction"]


def replay(api, actions, tail_max=0):
    """Replay `actions` from a fresh episode; optionally extend with straights until the
    circuit closes. Returns (placed_actions, closed, max_gain)."""
    builder = APITrackBuilder(api)
    pos, direction = _reset(api)
    placed, max_gain = [], 0.0

    def place(action):
        nonlocal pos, direction, max_gain
        ok, new_pos, new_dir = builder.take_action(action, pos, direction)
        if not ok:
            return None
        placed.append(int(action))
        pos, direction = new_pos, new_dir
        max_gain = max(max_gain, float(new_pos[2] - STATION_Z))
        return bool(builder.history[-1].get("is_complete", False))

    for action in actions:
        closed = place(action)
        if closed is None:
            return placed, False, max_gain
        if closed:
            return placed, True, max_gain
    for _ in range(tail_max):
        closed = place(0)
        if closed is None:
            return placed, False, max_gain
        if closed:
            return placed, True, max_gain
    return placed, False, max_gain


def main():
    parser = argparse.ArgumentParser(description="Seed the warm-start loop library (live)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--library", type=str, default="logs/loop_library.jsonl")
    parser.add_argument("--hill", action="store_true",
                        help="Seed chain-hill loop variants for the Phase-2 pool")
    parser.add_argument("--tail-max", type=int, default=16,
                        help="Max closure-walk straights after a candidate skeleton")
    args = parser.parse_args()

    api = APIController("localhost", args.port, verbose=0)
    if not api.connect():
        raise SystemExit(f"❌ Cannot connect to OpenRCT2 API on port {args.port}")

    library = LoopLibrary(args.library)
    print(f"📚 Library {args.library}: {len(library)} loops before seeding")

    candidates = generate_hill_candidates() if args.hill else generate_candidates()
    verified = added = 0
    for skeleton in candidates:
        placed, closed, gain = replay(api, skeleton, tail_max=args.tail_max)
        if not closed:
            print(f"  ✗ no closure from skeleton {skeleton} (placed {len(placed)})")
            continue
        placed2, closed2, gain2 = replay(api, placed)   # verification replay, no tail
        if not (closed2 and placed2 == placed):
            print(f"  ⚠ closure not reproducible, skipped: {placed}")
            continue
        verified += 1
        record = LoopRecord.from_actions(placed, source="scripted", max_gain=max(gain, gain2))
        if library.add(record):
            added += 1
            print(f"  ✓ verified len={record.length} chains={record.chain_count} "
                  f"gain={record.max_gain:.0f}: {list(record.actions)}")
        else:
            print(f"  = already in library (len {record.length})")

    api.disconnect()
    print(f"\n📚 Done: {verified} verified this run, {added} new, "
          f"{len(library)} total in {args.library}")
    if verified == 0:
        raise SystemExit("❌ NO-GO: zero verified closures -- debug with probe_corridor.py "
                         "before training")


if __name__ == "__main__":
    main()
