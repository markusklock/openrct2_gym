# Seed-Conditioned Coaster Variety — Design

**Date:** 2026-08-09
**Status:** Approved (design), pending implementation plan
**Supersedes:** the Phase-6 "Style / Variety" objective (fixed 12-turn / balance≥2 target)

## Goal

The agent should produce a **genuinely different wooden coaster each run**, while still
scoring as well as it can. Today it produces one coaster: 83% of unaided builds open with
the identical four pieces, and sampling its actions yields the same layout ±1 piece.

## Why the current design cannot deliver this

Every phase drives toward a single optimum, so reinforcement learning does what it should:
it converges on one answer. Three specific findings shaped this redesign.

1. **The agent is stuck in a local optimum, not a global one.** Mining 194k harvested
   coasters, its dominant recipe (≤5 turns, no direction changes, big hill — 103k builds)
   peaks at **E 6.11**, while winding families reach **6.43–6.45** and spirals **6.44**.
   Variety and score are not in tension; the most-built shape is not the best-scoring one.
2. **There is no latent variety to uncover.** Sampling the policy instead of taking its best
   action produced 8 "distinct" sequences that were all the same coaster (identical opening,
   identical turn count, length ±1). A random seed alone would have nothing to index into.
3. **Three separate metrics have measured something other than what they named** — S-bends
   counted as turns, the prefix descent gated on completion rather than retained style, and
   "warm builds wind at 7.9 turns" conflated agent-built with scaffold-replayed structure.
   The design below treats this as the project's characteristic failure mode.

## Approach

At each reset the agent draws a random seed `z` naming a **footprint family**. The seed
enters the observation as a one-hot. Reward = existing quality machinery + a term for
landing in the requested family. Generation: random `z` gives a different coaster each run;
chosen `z` gives the one you want.

Rejected alternative — rewarding novelty against the agent's own recent builds — requires
PPO to learn and hold a *mixed strategy*, which policy gradients drift away from (evidenced
by finding 2), and makes the reward non-stationary because the agent cannot observe its own
history. Seed-conditioning converts one unstable mixed-strategy problem into k stable
deterministic ones, which is the shape of problem PPO handles well.

## Families

Defined by the two properties identified as what the eye registers: **turn count** and
**direction-switch count** (how many times the turning direction alternates). Every family
below is proven achievable at E ≥ 6.1 by the archive — none is aspirational.

| z | family | turns | switches | archive support | best E |
|---|---|---|---|---|---|
| 0 | oval / one-way loop | ≤5 | 0 | 102,838 | 6.11 |
| 1 | spiral / helix | 6–9 | 0 | 874 | 6.44 |
| 2 | out-and-back | 6–9 | 1–2 | 27,655 | 6.41 |
| 3 | winding | 10–13 | 3–5 | 12,618 | 6.45 |
| 4 | serpentine | 14+ | 6+ | 61 | 6.20 |

Height, length and drop structure stay unconstrained — the agent optimises them freely for
score within its assigned footprint.

## Reward mechanics

`family_match ∈ [0,1]` = mean of a graded turn-band score and a graded switch-band score:
full credit inside the band, smooth falloff outside (no cliff; every leg needs a ramp or it
is never discovered). It feeds three places:

1. **Multiplicative completion gate** — replaces the current style floor. Ignoring the seed
   forfeits a large share of the completion payout. Multiplicative because additive carrots
   have repeatedly lost to reliable alternatives in this project.
2. **Discrete bonus** on family hit *and* tested E ≥ 4.5 — the "did the job asked" payment.
3. **Dense per-piece potential** — `family_match` is a pure function of the track so far, so
   it is PBRS-clean. Included because terminal-only signals have been consistently too slow
   here (the style gate ran ~900k steps without reaching unaided builds). Its weight is a
   per-phase parameter, zero in Phases 1–2 so those phases stay bit-identical.

**Removed:** the fixed 12-turn target (the target now comes from `z`) and the
handedness-balance leg (direction-switches measure the same thing better, and are meaningful
for one-way families, which *should* score zero switches without being punished).

**Curriculum gate changes meaning:** from "12 turns and balance ≥2 at E ≥ 4.5" to
**"built the family it was asked for, at E ≥ 4.5"**, tracked per seed.

**Deliberate omission:** no reward for differing from *previous* coasters. Diversity comes
entirely from `z` being random, which keeps the reward stationary.

## Curriculum

Phases 1–4 teach prerequisites a diverse builder still needs and are kept. What changes is
when shape matters:

| phase | budget | families active |
|---|---|---|
| 1–2 | 40 | none (too tight to express shape) |
| 3 | 60 | 0, 2 (oval, out-and-back) |
| 4 | 80 | + 3 (winding) |
| 5–6 | 80–120 | + 1, 4 (spiral, serpentine); P6 terminal, advancement judged per family |

The seed is in the observation from step one and **means something from step one even while
the family reward is off**: it selects which exemplar the warm-start scaffold replays. A
network given an input that predicts nothing learns to zero it out and must later unlearn
that; tying the seed to the scaffold keeps it live.

## Scaffold reuse and restart

The 194k-record library **survives the restart** — records are action sequences with no
dependency on observation format. The pool query already filters by structure; adding a
family filter gives per-family scaffolding immediately. Spiral (874) and serpentine (61) are
thin and should be thickened with `seed_p5_exemplars.py` as prep work.

Restart steps: tag the repo at the current state, archive the 22M checkpoint + VecNormalize
+ a library snapshot (so this line of work stays reproducible), then train fresh on a new
TensorBoard run, keeping the library.

**Cost:** the observation change forces a from-scratch retrain. Budget **3–5 days** to reach
the phase where variety is testable, mitigated by the Phase-3 early read below.

## Testing

Server-free, written before implementation:

- **Descriptor correctness** — a fixture per family classifies correctly; S-bends do not
  perturb it (they are no longer turns).
- **Graded match** — full credit inside the band, smooth falloff, no edge to sit on.
- **Completion-first invariant** — the family floor folds into the existing product of gates;
  worst-case completed-but-wrong-family payout still exceeds anything obtainable without
  completing. This guard has caught two regressions.
- **Economics from step one, both directions** — with `z`=winding the winding build must
  decisively beat the oval, *and* with `z`=oval the oval must beat the winding build. The
  second direction prevents "more turns is always better" from creeping back, and is the
  test whose absence allowed the S-bend exploit.
- **Warm-start filtering** — a spiral seed draws spiral exemplars.
- **PBRS hygiene** — dense term telescopes, place-then-remove nets negative, Phases 1–2
  bit-identical with the weight at zero.

**Diagnostics** (every mechanism streams its own): requested family, match score, switch
count, and **hit rate per family**.

## Success criteria

1. **Deliverable** — a gallery park with five visibly different coasters, each E ≥ 4.5, each
   produced on demand from its seed. Markus judges "visibly different".
2. **Metric** — per-family **hit rate** ≥ 0.5 for at least four of five families, where hit
   rate = (unaided episodes drawn with seed `z` that both land in family `z` **and** test at
   E ≥ 4.5) / (all unaided episodes drawn with seed `z`). Both conditions are required: a
   correctly-shaped ride that fails its test is not a hit.
3. **Early read** — at Phase 3 (~1 day), the two active families (oval and out-and-back) show
   non-zero and rising hit rates. Go/no-go before committing the full 3–5 days. Spiral was
   measured as a ~100-piece shape (median of 102 pieces across 894 library records) and cannot
   fit in Phase 3's 60-piece budget; consequently it appears only at Phase 5 onward, where the
   120-piece ceiling accommodates it. This adjustment allows Phase 3 to focus on shapes
   geometrically feasible within its constraints.

4. **Unaided quality** (added 2026-08-11, see the amendment below) — median measured
   excitement on **unaided** builds ≥ 4.5, tracked as `quality/median_excitement_cold`
   beside its sample count. This is a standalone criterion, not a by-product of criteria 1
   and 2, so quality progress stays visible even while variety is still developing.
   Baseline: **1.31** (median of a full 200-episode cold window). Corrected 2026-08-11
   from an initially-recorded 2.38 — see "a second, smaller conflation" below.

**Measurement rule:** the reward may score the whole track, but **every success claim is
judged on unaided builds only**. Four times now a number has flattered us because the
scaffold, not the agent, supplied the structure.

## Amendment, 2026-08-11: unaided quality was never demonstrated

This design was written on the premise "the agent builds a good coaster; now make it
varied." That premise was false, and the error was in the instrument rather than the policy.

`quality/median_excitement` pooled every tested episode. A *warm* episode replays a prefix of
a library exemplar (library best E 6.45), so its rating largely reflects the exemplar, not
the policy. With the cold-episode fraction at 0.25–0.35, warm rides dominated the window.
Split by source over 3,000 recent harvests:

| source | n | median | mean | max |
|---|---|---|---|---|
| cold — the agent placed every piece | 1704 | **2.38** | 1.95 | 5.81 |
| warm — a replayed exemplar | 1296 | **5.58** | 5.35 | 6.21 |

The predecessor run's headline "median excitement 5.58 at 22.5M steps" was measured the same
pooled way, so **v1's ride quality was also substantially the scaffold's**; its archive
README carries a correction. The same conflation was breaking training, not merely
reporting: the Phase-5 exploration floor releases when median excitement clears 4.0, which
warm replays cleared on their own, withdrawing exploration while the policy's own rides sat
at ~2.4.

Decision (Markus, 2026-08-11): pursue variety and unaided quality **together**, with quality
promoted to its own tracked criterion above, so success can never again be declared on
borrowed numbers. No change to the family design, the reward structure, or criteria 1–3.

### A second, smaller conflation, found the same day

The cold/warm table above is drawn from the harvested **library**, which is not a neutral
sample: `LoopLibrary.add` uses upgrade-append (`warm_start.py:210-215`), so a repeated action
sequence only re-enters with a *strictly higher* rating and the library keeps each sequence's
best. The agent builds the same oval repeatedly, so its library median is biased upward.

`quality/median_excitement_cold`, a median over the last 200 tested unaided episodes with no
filtering or dedup, reads **1.31** — and that is the number criterion 4 is judged on. The
library figures (cold 2.38 / warm 5.58) remain useful for the *relative* comparison that
exposed the original conflation, but both are optimistic in absolute terms. Criterion 4's
baseline is therefore 1.31, not 2.38.

## Risks

- **Serpentine may be unreachable** at good quality (61 examples). Response: drop to four
  families rather than chase it.
- **The network may ignore the seed** early. Mitigated by the seed selecting the scaffold
  exemplar from Phase 1.
- **Band boundaries are gameable in principle.** Far less than S-bends were: turn and switch
  counts cannot be padded with filler now that S-bends do not count as turns, and the band
  interior scores 1.0 so there is no incentive to sit on an edge.
- **Retrain cost** — 3–5 blind days, bounded by the Phase-3 early read.
