#!/usr/bin/env python3
"""Build a VARIETY gallery park: ask a trained policy for one coaster per footprint
family, on demand (Task 10 fix 2 -- the deliverable the approved spec's success
criterion #1 is judged on: "a gallery park with five visibly different coasters, each
E >= 4.5, each produced on demand from its seed").

For each requested family this:
  1. pins the curriculum's family sampler (run_model.pin_family, same mechanism as
     `run_model.py --family Z`),
  2. runs ONE cold policy episode and captures the exact action sequence placed
     (run_model.track_recorder -- needed because DummyVecEnv auto-resets the
     underlying env the instant a terminal step() returns, clearing the raw history),
  3. records whether it closed, what footprint.classify_family says it actually built,
     and its measured excitement if the ride tested.

All of the captured builds are then replayed side-by-side into ONE park, reusing
build_gallery.py's proven station/replay plumbing (gallery_slots, build_one, which
itself calls _build_station) -- imported, not duplicated. A build that misses its
family or never closes still gets its own slot and an honest row in the summary table:
a missing row would be worse than one that says "did not close".

Run against a dedicated instance -- NOT a training port. The per-family POLICY episodes
reset/recreate a ride at the canonical station each time (same as any run_model.py
inference run); the GALLERY replicas this script places at their own slots are never
deleted, so re-running accumulates more rides (see build_gallery.py):

    python build_variety_gallery.py --port 8101 --model checkpoints_showcase/<ckpt>.zip

Then save the park (headless: `save_park gallery` on the instance's stdin console; GUI:
File > Save) and open it in any OpenRCT2 to inspect -- the printed table is the evidence
for success criterion #1, read it next to the park.
"""
import argparse
import os

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from build_gallery import build_one, gallery_slots
from openrct2_gym.envs import footprint
from openrct2_gym.envs.api_controller import APIController
from openrct2_gym.envs.warm_start import LoopRecord
from run_model import (
    find_curriculum_wrapper, make_inference_env, pin_family, run_episode, track_recorder,
)
from train import _vecnormalize_path


def parse_families(spec):
    """Parse "--families 0,1,2,3,4" into a list of ints, validating each against
    footprint.FAMILY_N up front -- a bad value should fail before any env/model is
    loaded with a clear message, the same rule run_model.pin_family enforces for
    --family (family_match() indexes footprint.FAMILIES directly and would otherwise
    raise a bare IndexError deep inside the reward calc)."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            z = int(tok)
        except ValueError:
            raise SystemExit(f"--families entry {tok!r} is not an int")
        if not 0 <= z < footprint.FAMILY_N:
            raise SystemExit(
                f"--families entry {z} out of range; must be an int in "
                f"[0, {footprint.FAMILY_N - 1}]")
        out.append(z)
    if not out:
        raise SystemExit("--families produced no families to build")
    return out


def build_one_family(env, model, wrapper, recorder, z, deterministic):
    """Pin the sampler to family `z`, run one cold policy episode, and return a result
    row describing what actually happened -- honest even on a miss or a non-closure
    (see the module docstring: a missing row is worse than one that says "did not
    close"). `record` is None only when the episode placed literally zero pieces."""
    pin_family(wrapper, z)
    _steps, _total_reward, info = run_episode(env, model, deterministic=deterministic)
    actions = list(recorder["actions"])
    closed = bool(info.get("loop_completed"))
    built = footprint.classify_family(actions) if actions else None
    rating = info.get("ride_rating") or {}          # only set on a terminated (closed) episode
    excitement = float(rating.get("excitement", 0.0))
    record = LoopRecord.from_actions(actions, source="policy_variety",
                                     excitement=excitement) if actions else None
    return {
        "z": z,
        "requested": footprint.FAMILIES[z][0],
        "built": footprint.FAMILIES[built][0] if built is not None else "none",
        "hit": built == z,
        "closed": closed,
        "excitement": excitement,
        "pieces": len(actions),
        "record": record,
    }


def format_summary_table(rows):
    """Pure formatting over the result rows -- testable server-free. Every REQUESTED
    family gets exactly one row, in request order, whether or not it closed (see the
    module docstring)."""
    header = f"{'requested':<12}{'built':<12}{'match':<6}{'status':<14}{'E':>6}  {'pieces':>6}"
    lines = [header, "-" * len(header)]
    for r in rows:
        status = "closed" if r["closed"] else "did not close"
        match = "HIT" if r["hit"] else "MISS"
        lines.append(
            f"{r['requested']:<12}{r['built']:<12}{match:<6}{status:<14}"
            f"{r['excitement']:>6.2f}  {r['pieces']:>6d}")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True,
                        help="Path to the saved MaskablePPO model (with or without .zip)")
    parser.add_argument("--vecnormalize", default=None,
                        help="Path to VecNormalize stats (.pkl). Defaults to the model's sibling file.")
    parser.add_argument("--families", default="0,1,2,3,4",
                        help="Comma-separated footprint-family indices to request, one "
                             f"coaster each (0..{footprint.FAMILY_N - 1}; see footprint.FAMILIES "
                             "for names/bands)")
    parser.add_argument("--start-phase", type=int, default=6,
                        help="Curriculum phase for the policy env (default 6, matching run_model.py)")
    parser.add_argument("--game-speed", type=int, default=8,
                        help="Game speed while the policy builds/tests each candidate "
                             "(8 = training speed; the gallery replay always runs at 8 "
                             "then drops to 1 at the end, matching build_gallery.py)")
    parser.add_argument("--sample", action="store_true",
                        help="Sample actions instead of deterministic argmax")
    parser.add_argument("--dy", type=int, default=20,
                        help="Gallery slot spacing in tiles (see build_gallery.gallery_slots)")
    parser.add_argument("--rate-wait", type=float, default=20.0,
                        help="seconds to wait at the end for gallery test trains to rate")
    return parser.parse_args(argv)


def main():
    import time

    args = parse_args()
    families = parse_families(args.families)

    single_env = make_inference_env(args.port, start_phase=args.start_phase,
                                    game_speed=args.game_speed)
    wrapper = find_curriculum_wrapper(single_env)
    if wrapper is None:
        raise SystemExit("could not find ImprovedPhasedCurriculumWrapper in the env chain")
    base_env = wrapper._get_base_env()
    recorder = track_recorder(wrapper, base_env)

    env = DummyVecEnv([lambda: single_env])
    stats_path = args.vecnormalize or _vecnormalize_path(args.model)
    if os.path.exists(stats_path):
        print(f"Loading VecNormalize stats from {stats_path}")
        env = VecNormalize.load(stats_path, env)
    else:
        print(f"VecNormalize stats not found at {stats_path}; running without learned "
              f"obs normalization (results may be degraded).")
        env = VecNormalize(env, norm_obs=True, norm_reward=False, norm_obs_keys=["scalars"])
    env.training = False
    env.norm_reward = False

    model = MaskablePPO.load(args.model, env=env)

    rows = []
    for z in families:
        row = build_one_family(env, model, wrapper, recorder, z, deterministic=not args.sample)
        rows.append(row)
        print(f"[{row['requested']:12s}] built={row['built']:12s} "
              f"{'HIT' if row['hit'] else 'MISS'} "
              f"{'closed' if row['closed'] else 'did not close'} "
              f"E={row['excitement']:.2f} pieces={row['pieces']}")
    env.close()

    # Build the park: reuse build_gallery.py's proven station/replay plumbing (never
    # duplicate it). Every row gets a slot, even a non-closing build -- a missing
    # coaster from the park would be worse than one that visibly failed to close.
    api = APIController("localhost", args.port, verbose=0)
    api.set_game_speed(8)
    slots = gallery_slots(len(rows), dy=args.dy)
    for row, slot in zip(rows, slots):
        row["slot"] = slot
        if row["record"] is None:
            row["status"] = "NO TRACK"
            continue
        ride_id = build_one(api, row["record"], slot)
        row["status"] = "OK" if ride_id is not None else "FAILED"

    print(f"\nwaiting {args.rate_wait:.0f}s for gallery test trains to rate...")
    time.sleep(args.rate_wait)
    api.set_game_speed(1)

    print("\n" + format_summary_table(rows))
    print("\nSlots (fly to these y-rows in-game; station row x=61..56):")
    for row in rows:
        y = row["slot"][1]
        print(f"  y={y:3d}  requested={row['requested']:12s} -> {row['status']}")
    print("\nNow save the park and open it in any OpenRCT2 to inspect.")


if __name__ == "__main__":
    main()
