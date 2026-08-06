"""Canonical track-piece families (action ids), dependency-free so every layer can
import the SAME definition.

The distinction that matters for the P6 style legs is **heading change vs lateral
weave**:

* A turn piece (plain 1-4, banked 21-24) rotates the train's heading. Stringing them
  is what makes a layout wind.
* An S-bend (29/30) shifts the track sideways one tile and hands back the ORIGINAL
  heading -- a lateral weave, not a direction change.

Counting S-bends as "turns" was a live-observed reward exploit (Aug-6): the policy
learned to build two 180s and then stack 6-8 S-bends into a diagonal drift home,
which farmed the turn-count leg, the handedness-balance leg (29 scored left, 30
scored right, so an alternating stack manufactured both) AND the qualified gate --
61% of counted "turns" in the newest cold harvests were S-bends. The style legs
therefore count HEADING turns only; S-bends keep their own small capped leg
(struct_w_sbend / struct_sbend_target), so a few are still worth building and a
stack past the target is worth exactly nothing.

CURVED_ACTIONS (turns + S-bends) is a separate, deliberately broader family for the
PHYSICS estimate: an S-bend is curved track and really does cost turn friction.
"""

# Heading-changing turn pieces, split by handedness (the balance leg's basis).
LEFT_TURN_ACTIONS = (1, 3, 21, 23)
RIGHT_TURN_ACTIONS = (2, 4, 22, 24)
TURN_ACTIONS = LEFT_TURN_ACTIONS + RIGHT_TURN_ACTIONS

# Lateral weave: no net heading change. Own struct leg, never a "turn".
SBEND_ACTIONS = (29, 30)

# Anything with lateral curvature -- friction model only, NOT style credit.
CURVED_ACTIONS = TURN_ACTIONS + SBEND_ACTIONS
