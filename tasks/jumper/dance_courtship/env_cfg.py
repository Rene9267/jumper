"""Environment config for jumper.dance_courtship: `media/courtship.npz`.

The environment is `tasks/jumper/common/dance/env.py`; this file holds every
number. Where a number follows a rule `jumper.dance` wrote down for its own clip,
the rule is applied to this clip's measurements and the result is quoted -- read
`tasks/jumper/dance/env_cfg.py` for the reasoning behind each rule.

## This clip, measured

At the 50 Hz control rate the generator writes (10.26 s, 514 frames): per-joint
position std 0.164 rad at the median and 0.706 at the most active, RMS amplitude
0.315 rad; joint velocity RMS 1.154 rad/s, per-joint std 0.901 at the median and
1.969 at the most active, peaking at 17.7 rad/s; the base between 100.5 and
106.5 mm; roll to 10 degrees and nose-up pitch to 10 degrees.

**The reward weights are starting points**, as they are for jumper.dance: no
policy has been trained on this clip yet.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from ..common.dance.env import dance_env_cfg

#: This task's material, written by `tools/make_courtship_dance.py`.
MEDIA = Path(__file__).resolve().parent / "media"

#: Episode length, seconds. The clip is 10.26 s. jumper.dance_maze's rule for a
#: clip that short: an episode that runs off the end teleports the robot back
#: onto the reference, and a 10 s episode would do that almost every time. At
#: 5 s, an episode sampled in the first half runs clean.
EPISODE_S = 5.0


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config. See `jumper.dance`'s for the arguments."""
    return dance_env_cfg(
        media=MEDIA,
        asset=asset,
        play=play,
        episode_s=EPISODE_S,
        # ── Reference-state initialisation ────────────────────────────────
        # jumper.dance's: fractions of this robot (10 mm is a tenth of its standing
        # height), not of the clip, so they carry over unchanged.
        rsi_pose_range={
            "x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005),
            "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05),
        },
        rsi_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
        rsi_joint_range=(-0.05, 0.05),
        # ── Observation noise ─────────────────────────────────────────────
        # The tracking skeleton's, as jumper.dance has them; not measured.
        joint_pos_noise=0.01,
        joint_vel_noise=0.5,
        # ── Rewards ───────────────────────────────────────────────────────
        # Positive budget 6.5 a step, as jumper.dance: mjlab's six tracking terms
        # unchanged (5.0) plus the two joint-space terms below (1.5).
        #
        # Joint position: jumper.dance's rule is a std at 0.44 of the clip's RMS
        # amplitude, which scores a robot standing still through the clip near
        # 0.006. Here: 0.44 x 0.315 -> 0.14, standing still 0.006.
        joint_pos_std=0.14,
        joint_pos_weight=1.0,
        # Joint velocity: two thirds of the clip's velocity RMS, standing still
        # about 0.10. Here: 1.154 -> 0.77, standing still 0.11. Half the
        # position weight, so when they disagree the pose wins.
        joint_vel_std=0.77,
        joint_vel_weight=0.5,
        # Smoothness: -0.1 is calibrated at this action scale (0.25) and the
        # second difference at half of it -- properties of the robot and the
        # action space, not of the clip.
        action_rate_weight=-0.1,
        action_acc_weight=-0.05,
        # Power: 22 joints at ~0.25 N*m and this clip's ~1.15 rad/s is ~6.3 W,
        # so -0.02 is about -0.13 a step, 2% of the budget.
        power_weight=-0.02,
        # Torque above the continuous rating: anchored to the servo, not the clip
        # -- one joint at the plateau costs 0.6, 9% of the budget.
        torque_headroom_weight=-2.0,
        # Joint limits. The reference stays inside the hard stops (closest is the
        # shut finger, 100 mrad short) and inside the soft limit the penalty
        # charges (90% of each range; the same finger is 15 mrad inside that).
        joint_limit_weight=-10.0,
        # Leg-on-leg contact above 1.0 N, ~8% of the robot's worst-case push.
        self_collision_weight=-1.0,
        self_collision_force=1.0,
        # ── Terminations ──────────────────────────────────────────────────
        # Root height: this clip moves the body through a 6 mm band (100.5-106.5).
        # jumper.dance's 40 mm threshold still exceeds the whole band, so it
        # cannot fire on a policy that is merely tracking badly.
        anchor_height_error=0.04,
        # 1 - cos(tilt) = 0.3 is 45.6 degrees, "has fallen over". The dance
        # itself rolls and pitches to 10 degrees (1 - cos = 0.015).
        anchor_tilt_error=0.3,
        # A support foot 50 mm out vertically is half the standing height.
        support_foot_error=0.05,
        # ── Disturbance ───────────────────────────────────────────────────
        # jumper.dance's: gentler and rarer than mjlab's, sized to a 2 kg robot.
        push_interval_s=(3.0, 8.0),
        push_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
    )
