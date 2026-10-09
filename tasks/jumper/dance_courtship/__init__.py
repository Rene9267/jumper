"""jumper.dance_courtship -- a crab courtship dance, learned by imitation.

The same task as `jumper.dance` with a different clip: the environment is
`tasks/jumper/common/dance/env.py`, and what this directory adds is the clip,
the music and the numbers.

## The clip

`media/courtship.npz`, written by `tools/make_courtship_dance.py` from the poses
in `tools/courtship_keyframes.py`. It is an original choreography, not a take
imported from rl-wbc-fsm: two front claws gesture, the body rolls and bows
nose-up, and the four support legs are solved so the feet stay planted.

The music is `media/voxel_dance_cut.wav`. Training does not read it. The
performance video `scripts/export.py` renders does, and this task's
`export_media.py` muxes it at the clip's lead-in.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.dance_courtship",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating an original crab courtship dance; the clip "
                "and its music are committed in tasks/jumper/dance_courtship/media/",
    tags=("imitation", "jumper", "dance", "flat"),
)
