"""Named poses for ``jumper.dance_courtship``, laid on the beat grid.

Edit the constants at the top. Joint angles are radians; lengths are metres.
Each claw tuple is ``(j0 yaw, j1 pitch, j2 roll, j3 wrist, j4 finger)``.

The two front claws are what the dance gestures with. The four support legs are
not posed here: ``tools/make_courtship_dance.py`` solves them so the feet stay
on the floor while the body rolls, pitches and sways. Raising ``TILT_RAD`` or
``BOW_PITCH_RAD`` asks more of those legs. The generator prints how close they
come to a stop, and how far the feet leave the floor.

Sign conventions, measured on this robot:

- Positive roll lowers the right side (right-hand rotation about +x).
- Negative pitch is nose-up. Positive pitch points the nose at the floor, which
  this dance does not do.
- ``LF_J4`` opens toward negative and shuts toward 0. ``RF_J4`` is the mirror:
  open is positive, shut is 0. Zero is nearly shut, about 10 mm of mouth left.
"""

from __future__ import annotations

import math

from tasks.jumper.common.constants import HOME, STAND_Z

# ── Music ────────────────────────────────────────────────────────────────
# "Voxel Revolution" by Kevin MacLeod, the cut in media/voxel_dance_cut.wav.
# Beat times are the ``voxel_dance_cut`` entry of voxel_dance_cut_beats.json,
# 1-indexed in the schedule below (index 0 is beat 1). 122 bpm, one beat every
# 0.4918 s, sixteen of them. The cut is sixteen beats long.
BPM = 122.0
BEAT_TIMES_S: tuple[float, ...] = (
    0.0, 0.4918, 0.9836, 1.4754, 1.9672, 2.459, 2.9508, 3.4426,
    3.9344, 4.4262, 4.918, 5.4098, 5.9016, 6.3934, 6.8852, 7.377,
)
BEAT_S = BEAT_TIMES_S[1] - BEAT_TIMES_S[0]
N_BEATS = len(BEAT_TIMES_S)
#: Sixteen beat periods. The wav is 7.86884 s; this grid ends 0.04 ms earlier.
MUSIC_S = N_BEATS * BEAT_S

# ── Timing around the music ──────────────────────────────────────────────
#: Stillness before beat 1, so the clip does not open mid-gesture. The music
#: starts when this ends (``audio_start_in_sim``).
LEAD_IN_S = 1.0
#: Return to the standing pose after the music, then a short hold there.
COOLDOWN_S = 1.0
HOLD_S = 0.40
#: How long a claw snap takes. It lands ON the beat; the motion is this long
#: just before it. 0.10 s keeps the finger under the servo's corner speed.
SNAP_S = 0.10
#: Same idea for one beckon curl, and for the flop on beat 16's offbeat.
BECKON_S = 0.10
FLOP_S = 0.18

# ── Amplitudes ───────────────────────────────────────────────────────────
# Exaggerated body tilts (beats 13 and 14). 10 degrees is the starting point;
# roll this large is untested on the robot, so raise it in the preview until
# the feet slip, then stop. The generator's foot-height line is that check.
TILT_RAD = math.radians(10.0)
# Shimmy on beats 5-6: a smaller roll plus a sideways sway, claws kept high.
SHIMMY_ROLL_RAD = math.radians(6.0)
SHIMMY_Y_M = 0.012
# Crab-twerk bows (beats 11-12): nose up, and the body lowered so the rear
# drops further. Positive number; the pose applies it as negative pitch.
BOW_PITCH_RAD = math.radians(10.0)
BOW_DROP_M = 0.006
# How wide the claws open. 1.20 rad is well inside the finger stops.
CLAW_OPEN = 1.20


def t(beat: int, fraction: float = 0.0, *, early: float = 0.0) -> float:
    """Audio time of a 1-indexed beat, plus a fraction of a beat.

    ``fraction=0.5`` is the offbeat. ``early`` moves the knot earlier so a snap
    can land on the beat instead of starting there.
    """
    if not 1 <= beat <= N_BEATS:
        raise ValueError(f"beat {beat} is outside 1..{N_BEATS}")
    return BEAT_TIMES_S[beat - 1] + fraction * BEAT_S - early


class Pose:
    """One body pose and the two claws. Support legs are solved later."""

    def __init__(
        self,
        lf: tuple[float, float, float, float, float],
        rf: tuple[float, float, float, float, float],
        *,
        y: float = 0.0,
        z: float | None = None,
        roll: float = 0.0,
        pitch: float = 0.0,
        x: float = 0.0,
        yaw: float = 0.0,
    ) -> None:
        self.lf = tuple(float(v) for v in lf)
        self.rf = tuple(float(v) for v in rf)
        self.x = float(x)
        self.y = float(y)
        self.z = STAND_Z if z is None else float(z)
        self.roll = float(roll)
        self.pitch = float(pitch)
        self.yaw = float(yaw)

    def channels(self) -> tuple[float, ...]:
        """Body (x, y, z, roll, pitch, yaw) then the ten claw joints."""
        return (
            self.x, self.y, self.z, self.roll, self.pitch, self.yaw,
            *self.lf, *self.rf,
        )


def both(
    lf: tuple[float, float, float, float, float],
    **body: float,
) -> Pose:
    """The same claw on both sides. Yaw, roll and the finger flip sign."""
    j0, j1, j2, j3, j4 = lf
    return Pose(lf, (-j0, j1, -j2, -j3, -j4), **body)


def _home_claw(side: str) -> tuple[float, float, float, float, float]:
    return tuple(HOME[f"{side}_J{i}_joint"] for i in range(5))  # type: ignore[return-value]


#: Standing, claws down, the pose the clip leaves and returns to.
HOME_POSE = Pose(_home_claw("LF"), _home_claw("RF"))

#: Halfway up, claws already wide. A straight blend from the stance to the
#: overhead pose sweeps the fingertips through each other; this waypoint keeps
#: them about 190 mm apart while they climb.
RISE_APART = both((-0.71, -2.15, -0.40, -0.30, -0.50))
#: Both claws up above the shell, fingers open. Tips sit near 0.24 m.
OVERHEAD_OPEN = both((-1.05, -2.95, -0.70, -0.30, -CLAW_OPEN))
#: Same arms, fingers shut. The clack is this change and nothing else.
OVERHEAD_SHUT = both((-1.05, -2.95, -0.70, -0.30, 0.0))

#: Beat 5: sway and lean to the robot's left, claws still overhead.
SHIMMY_LEFT = both(
    (-1.05, -2.95, -0.70, -0.30, -CLAW_OPEN),
    y=SHIMMY_Y_M,
    roll=-SHIMMY_ROLL_RAD,
)
#: Beat 6: the other way.
SHIMMY_RIGHT = both(
    (-1.05, -2.95, -0.70, -0.30, -CLAW_OPEN),
    y=-SHIMMY_Y_M,
    roll=SHIMMY_ROLL_RAD,
)

#: Beat 7: body freezes upright. The left claw has dropped to about half height;
#: the right stays up, ready to beckon.
FREEZE_HALF = Pose(
    lf=(-0.85, -1.85, -0.50, -0.90, -0.55),
    rf=(1.05, -2.95, 0.70, 0.30, CLAW_OPEN),
)
#: Beat 8: the right claw curls in toward the chest ("come here"). Twice.
BECKON = Pose(
    lf=(-0.85, -1.85, -0.50, -0.90, -0.55),
    rf=(1.05, -2.80, 1.05, 0.65, 0.90),
)

#: Beats 9-10. Both claws travel to the left, the left one higher so they pass
#: without occupying the same point, then across to the right, then back.
WIPER_LEFT = Pose(
    lf=(-0.15, -2.25, -0.45, -0.35, -0.85),
    rf=(1.90, -1.95, 0.20, -0.10, 0.75),
)
WIPER_PASS = Pose(
    lf=(-1.35, -2.30, -0.35, 0.05, -0.80),
    rf=(1.35, -1.90, 0.25, -0.15, 0.70),
)
WIPER_RIGHT = Pose(
    lf=(-1.90, -1.95, -0.20, 0.10, -0.75),
    rf=(0.15, -2.25, 0.45, 0.35, 0.85),
)
WIPER_PASS_BACK = Pose(
    lf=(-1.35, -1.90, -0.25, 0.15, -0.70),
    rf=(1.35, -2.30, 0.35, -0.05, 0.80),
)

#: Beats 11-12. Claws stay forward; only the body bows, twice.
_BOW_CLAWS = ((-0.90, -2.20, -0.40, -0.50, -0.80), (0.90, -2.20, 0.40, 0.50, 0.80))
BOW_LEVEL = Pose(*_BOW_CLAWS)
BOW = Pose(*_BOW_CLAWS, z=STAND_Z - BOW_DROP_M, pitch=-BOW_PITCH_RAD)

#: Beats 13-14. Claws wide and up so the lean reads. Right side down, then left.
TILT_CLAWS = ((-0.30, -2.40, -0.90, -0.20, -1.00), (0.30, -2.40, 0.90, 0.20, 1.00))
TILT_RIGHT = Pose(*TILT_CLAWS, roll=TILT_RAD)
TILT_LEFT = Pose(*TILT_CLAWS, roll=-TILT_RAD)

#: Halfway from the lean into the heart: claws already high and still apart, so
#: the joint blend does not sweep the tips through each other on the way in.
HEART_APPROACH = both((-1.20, -2.70, -0.50, -0.30, -0.80))
#: Beat 15. Fingertips about 11 mm apart overhead -- the pads touching. This pose
#: is the close end of a blend that shuts steadily; a nearer pinch in joint space
#: swings the tips through each other and then apart again, which reads as a miss.
#: Grippers stay partly open so the claws read as the lobes of a heart.
HEART = both((-1.40, -2.75, -0.75, -0.45, -0.70))
#: Beat 16, offbeat. The left claw drops to the stance; the right holds the heart.
FLOP = Pose(lf=_home_claw("LF"), rf=HEART.rf)


# (audio time in seconds, pose, label). Times must increase.
# Beat numbers are 1-indexed and match the director's sheet.
SCHEDULE: tuple[tuple[float, Pose, str], ...] = (
    (-LEAD_IN_S, HOME_POSE, "settle, before the music"),
    (t(1), HOME_POSE, "beat 1: start the rise"),
    (t(1, 0.5), RISE_APART, "claws rise, opened wide"),
    (t(2), OVERHEAD_OPEN, "beat 2: both claws overhead, open"),
    (t(3, early=SNAP_S), OVERHEAD_OPEN, "hold the open claws"),
    (t(3), OVERHEAD_SHUT, "beat 3: clack shut"),
    (t(4, early=SNAP_S), OVERHEAD_SHUT, "hold the shut claws"),
    (t(4), OVERHEAD_OPEN, "beat 4: clack open"),
    (t(5), SHIMMY_LEFT, "beat 5: shimmy left"),
    (t(6), SHIMMY_RIGHT, "beat 6: shimmy right"),
    (t(7), FREEZE_HALF, "beat 7: freeze, left claw halfway down"),
    (t(8, early=BECKON_S), FREEZE_HALF, "right claw ready"),
    (t(8), BECKON, "beat 8: beckon"),
    (t(8) + BECKON_S, FREEZE_HALF, "beckon returns"),
    (t(8, 0.5, early=BECKON_S), FREEZE_HALF, "ready for the offbeat"),
    (t(8, 0.5), BECKON, "beat 8 offbeat: beckon again"),
    (t(8, 0.5) + BECKON_S, FREEZE_HALF, "beckon returns"),
    (t(9), WIPER_LEFT, "beat 9: wipers to the left"),
    (t(9, 0.5), WIPER_PASS, "wipers cross, left claw higher"),
    (t(10), WIPER_RIGHT, "beat 10: wipers to the right"),
    (t(10, 0.5), WIPER_PASS_BACK, "wipers cross back, right claw higher"),
    (t(11), BOW, "beat 11: crab twerk, nose up"),
    (t(11, 0.5), BOW_LEVEL, "up between the bows"),
    (t(12), BOW, "beat 12: crab twerk again"),
    (t(12, 0.5), BOW_LEVEL, "level, before the leans"),
    (t(13), TILT_RIGHT, "beat 13: lean right"),
    (t(14), TILT_LEFT, "beat 14: lean left"),
    (t(14, 0.5), HEART_APPROACH, "claws rise, still apart"),
    (t(15), HEART, "beat 15: claw tips meet, a heart"),
    (t(16), HEART, "beat 16: the heart holds"),
    (t(16, 0.5), FLOP, "beat 16 offbeat: left claw flops"),
    (MUSIC_S, FLOP, "music ends, flop held"),
    (MUSIC_S + COOLDOWN_S, HOME_POSE, "cooldown, back to the stance"),
    (MUSIC_S + COOLDOWN_S + HOLD_S, HOME_POSE, "hold the stance"),
)
