#!/usr/bin/env python3
"""Turn ``courtship_keyframes.py`` into the dance clip ``jumper.dance_courtship`` trains on.

    python tools/make_courtship_dance.py

The clip is the schema ``tasks/jumper/common/dance/motion.py`` reads: one float32
channel per joint and per base axis, command and ``meas_*`` both filled with the
same trajectory. There is no measured take. The poses are already a posture the
robot can hold, because the four support feet are solved onto the floor rather
than copied from a planner's body target. ``import_wbc_dances.solve_base_pose``
does the opposite -- joints in, body out -- which is right for a recording that
only has joint angles. Here the director specified the body (roll, nose-up pitch,
a sideways sway), so the body is the input and the support legs are the output.

Samples are written at the 50 Hz control rate (5 ms physics, decimation 4), which
is the rate the continuity check judges after resampling. A knot-to-knot blend is
a smoothstep, so a joint's peak speed is 1.5 times the average across that blend.

The run prints, and refuses to write if any of these fail:

- a joint faster than the servo's corner speed (``CORNER_SPEED``, 30.7 rad/s)
- a joint outside its mechanical range
- a support foot more than 2 mm off the point it stands on at ``HOME``
- a non-foot mesh under 2 mm from the floor, or a foot mesh through the floor
"""

from __future__ import annotations

import importlib.util
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MEDIA = REPO / "tasks" / "jumper" / "dance_courtship" / "media"
#: 50 Hz. ``tracking_env_cfg`` is 5 ms physics and decimation 4; the dance loader
#: resamples to that product, and an exact match is taken as a stride.
CONTROL_DT = 0.02
#: How far a solved foot may sit from its stance point before the clip is refused.
#: The loader's own coplanarity tolerance is 5 mm on the median frame. This is
#: tighter because the feet are a target of the solve, not a recording's leftover.
FOOT_TOL_M = 0.002
#: Non-foot meshes must clear the floor by this much. 2 mm, not 3: at the stance
#: the palm mesh is 2.3 mm up, and a 3 mm line would refuse the standing pose.
NONFOOT_CLEAR_M = 0.002
#: Support-foot meshes may rest on the plane. This is how far past it still counts
#: as contact rather than a corner that has gone through.
FOOT_MESH_TOL_M = 0.0002
SUPPORT = ("LM", "RM", "LR", "RR")
BODY_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")


def _load_keyframes():
    """The sibling file, loaded by path so ``tools/`` does not need to be a package."""
    path = Path(__file__).with_name("courtship_keyframes.py")
    spec = importlib.util.spec_from_file_location("courtship_keyframes", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _smoothstep(u: np.ndarray) -> np.ndarray:
    """Cubic ease in/out. Peak slope is 1.5, at the middle of the blend."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _interpolate(times: np.ndarray, knot_t: np.ndarray, knot_v: np.ndarray) -> np.ndarray:
    """Blend ``knot_v`` across ``knot_t`` and sample it at ``times``.

    Before the first knot and after the last, the pose holds. A sample that lands
    on a knot gets that knot, from either side.
    """
    out = np.empty((len(times), knot_v.shape[1]), dtype=np.float64)
    out[:] = knot_v[0]
    for i in range(len(knot_t) - 1):
        t0, t1 = float(knot_t[i]), float(knot_t[i + 1])
        mask = (times >= t0) & (times <= t1)
        if not np.any(mask):
            continue
        u = (times[mask] - t0) / (t1 - t0)
        s = _smoothstep(u)[:, None]
        out[mask] = knot_v[i] + (knot_v[i + 1] - knot_v[i]) * s
    after = times >= knot_t[-1]
    out[after] = knot_v[-1]
    return out


class _Robot:
    """The jumper, plus a 3-DoF solve that keeps each support foot where it stands."""

    def __init__(self) -> None:
        import mujoco

        from tasks.jumper.common.constants import HOME, STAND_Z, get_spec
        from tasks.jumper.common.dance.motion import LEG_JOINTS, _rpy_to_quat

        self.mujoco = mujoco
        self.home = HOME
        self.stand_z = STAND_Z
        self.leg_joints = LEG_JOINTS
        self.rpy_to_quat = _rpy_to_quat
        self.model = get_spec(None).compile()
        self.data = mujoco.MjData(self.model)
        self.names = [n for leg in sorted(LEG_JOINTS) for n in LEG_JOINTS[leg]]
        jid = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.names
        ]
        if any(i < 0 for i in jid):
            missing = [n for n, i in zip(self.names, jid) if i < 0]
            raise RuntimeError(f"model is missing joints: {missing}")
        self.q_adr = self.model.jnt_qposadr[jid]
        self.dof_adr = self.model.jnt_dofadr[jid]
        self.range = self.model.jnt_range[jid].copy()
        self.index = {n: i for i, n in enumerate(self.names)}
        self.sites = {
            leg: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, leg) for leg in SUPPORT
        }
        if any(i < 0 for i in self.sites.values()):
            raise RuntimeError(f"model is missing a support-foot site: {self.sites}")
        self.tip = {
            side: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_finger_tip_link"
            )
            for side in ("LF", "RF")
        }
        self.base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        self.targets = self._stance_feet()

    def _stance_feet(self) -> dict[str, np.ndarray]:
        q = np.array([self.home[n] for n in self.names], dtype=np.float64)
        self._set(q, np.array([0.0, 0.0, self.stand_z, 0.0, 0.0, 0.0]))
        self.mujoco.mj_forward(self.model, self.data)
        return {leg: self.data.site_xpos[sid].copy() for leg, sid in self.sites.items()}

    def _set(self, q: np.ndarray, body: np.ndarray) -> None:
        """``body`` is (x, y, z, roll, pitch, yaw), the schema's convention."""
        d = self.data
        d.qpos[:] = 0.0
        d.qpos[0:3] = body[:3]
        d.qpos[3:7] = self.rpy_to_quat(body[3:6][None, :])[0]
        d.qpos[self.q_adr] = q

    def solve_feet(
        self, q: np.ndarray, body: np.ndarray, warm: np.ndarray, foot_lift: float = 0.0,
    ) -> np.ndarray:
        """Move the twelve support joints so the feet sit on ``self.targets``.

        ``q`` carries the claw joints and is not otherwise read for the legs:
        the legs start at ``warm`` (the previous frame, or ``HOME`` on the first)
        and stay on that kinematic branch. Thirty damped steps is more than a
        20 ms frame needs; the cap is what makes a pose the legs cannot reach
        fail the foot check instead of iterating forever.

        ``foot_lift`` raises the target, in metres. A roll tips the foot mesh
        through the plane while the site, which the solve holds, stays put;
        the lift is that mesh's clearance.
        """
        out = warm.copy()
        claw = [i for i, n in enumerate(self.names) if n.startswith(("LF_", "RF_"))]
        out[claw] = q[claw]
        lift = np.array([0.0, 0.0, foot_lift])
        targets = {leg: pos + lift for leg, pos in self.targets.items()}
        damp = 1e-4 * np.eye(3)
        for _ in range(30):
            self._set(out, body)
            self.mujoco.mj_kinematics(self.model, self.data)
            worst = 0.0
            for leg, sid in self.sites.items():
                err = targets[leg] - self.data.site_xpos[sid]
                worst = max(worst, float(np.linalg.norm(err)))
                self.mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, sid)
                cols = [self.index[f"{leg}_J{j}_joint"] for j in range(3)]
                jac = self._jacp[:, self.dof_adr[cols]]
                step = jac.T @ np.linalg.solve(jac @ jac.T + damp, err)
                # A full Newton step from a cold start can leave the branch the
                # warm pose was on. Cap it; the next iteration finishes the job.
                step = np.clip(step, -0.25, 0.25)
                for col, dq in zip(cols, step):
                    lo, hi = self.range[col]
                    out[col] = float(np.clip(out[col] + dq, lo, hi))
            if worst < 2e-4:
                break
        return out

    def feet_and_tips(self, q: np.ndarray, body: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Support-foot positions [4, 3], fingertip positions [2, 3], and the
        world-z of the body's forward axis (positive means the nose is up)."""
        self._set(q, body)
        self.mujoco.mj_forward(self.model, self.data)
        feet = np.stack([self.data.site_xpos[self.sites[leg]] for leg in SUPPORT])
        tips = np.stack([self.data.xpos[self.tip[side]] for side in ("LF", "RF")])
        forward_z = float(self.data.xmat[self.base].reshape(3, 3)[2, 0])
        return feet, tips, forward_z

    def _mesh_z(self, gid: int) -> tuple[float, float]:
        """World-z of the lowest and highest vertex of a mesh geom."""
        mid = int(self.model.geom_dataid[gid])
        adr = int(self.model.mesh_vertadr[mid])
        n = int(self.model.mesh_vertnum[mid])
        v = self.model.mesh_vert[adr:adr + n]
        rot = self.data.geom_xmat[gid].reshape(3, 3)
        z = (v @ rot.T + self.data.geom_xpos[gid])[:, 2]
        return float(z.min()), float(z.max())

    def prepare_ground(self) -> None:
        """Index the meshes the floor check walks. Once per robot."""
        from tasks.jumper.common.constants import FEET

        mujoco = self.mujoco
        foot = tuple(f"{name}_" for name in FEET)
        self._ground = []
        for gid in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if not name.endswith(("_visual", "_meshcol")):
                continue
            self._ground.append((gid, name, name.startswith(foot)))

    def lowest_meshes(self, q: np.ndarray, body: np.ndarray) -> tuple[tuple, tuple]:
        """Lowest non-foot mesh and lowest foot mesh, as ``(z, name)``.

        A bounding sphere above the line skips the vertex walk. Feet and the
        other links are tracked separately: a foot is allowed to rest on the
        plane, and anything else has to clear it.
        """
        if not hasattr(self, "_ground"):
            self.prepare_ground()
        self._set(q, body)
        self.mujoco.mj_forward(self.model, self.data)
        nonfoot = (1e9, "")
        foot = (1e9, "")
        for gid, name, is_foot in self._ground:
            # rbound is a sphere around the geom, so a bottom above the best
            # mesh already found cannot be the lowest.
            bound = float(self.data.geom_xpos[gid, 2] - self.model.geom_rbound[gid])
            best = foot[0] if is_foot else nonfoot[0]
            if bound >= best:
                continue
            lo, _hi = self._mesh_z(gid)
            if is_foot:
                if lo < foot[0]:
                    foot = (lo, name)
            elif lo < nonfoot[0]:
                nonfoot = (lo, name)
        return nonfoot, foot


@dataclass
class Dance:
    """One generated clip, at ``CONTROL_DT``, joints in ``LEG_JOINTS`` order."""

    time: np.ndarray
    audio_time: np.ndarray
    phase: np.ndarray
    body: np.ndarray
    joints: np.ndarray
    feet: np.ndarray
    tips: np.ndarray
    names: list[str]
    ranges: np.ndarray
    foot_targets: np.ndarray
    labels: list[tuple[float, str]]


def _schedule(kf) -> tuple[np.ndarray, np.ndarray, list[str]]:
    times, poses, labels = [], [], []
    for when, pose, label in kf.SCHEDULE:
        times.append(float(when))
        poses.append(pose.channels())
        labels.append(label)
    knot_t = np.asarray(times, dtype=np.float64)
    if np.any(np.diff(knot_t) <= 1e-9):
        bad = int(np.argmin(np.diff(knot_t)))
        raise ValueError(
            f"schedule times do not increase at {labels[bad]!r} ({knot_t[bad]:.4f} s) "
            f"-> {labels[bad + 1]!r} ({knot_t[bad + 1]:.4f} s)"
        )
    return knot_t, np.asarray(poses, dtype=np.float64), labels


def _phase(audio_time: np.ndarray, kf) -> np.ndarray:
    """1 dance, 2 cooldown, 3 hold, matching the schema's three values."""
    phase = np.ones(len(audio_time), dtype=np.float64)
    phase[audio_time < 0.0] = 3.0
    phase[audio_time >= kf.MUSIC_S] = 2.0
    phase[audio_time >= kf.MUSIC_S + kf.COOLDOWN_S] = 3.0
    return phase


def build(robot: _Robot | None = None) -> Dance:
    """Sample the schedule, solve the feet, and return the clip. No printing."""
    kf = _load_keyframes()
    robot = _Robot() if robot is None else robot
    knot_t, knot_v, labels = _schedule(kf)
    lead = kf.LEAD_IN_S
    end = float(knot_t[-1])
    # Audio time. The grid includes the last knot even when it is not on a sample.
    n = int(np.floor((end + lead) / CONTROL_DT + 1e-9)) + 1
    sim_t = np.arange(n, dtype=np.float64) * CONTROL_DT
    audio_t = sim_t - lead
    blended = _interpolate(audio_t, knot_t, knot_v)

    claw_idx = [
        robot.index[f"{side}_J{j}_joint"] for side in ("LF", "RF") for j in range(5)
    ]
    home_q = np.array([robot.home[name] for name in robot.names], dtype=np.float64)
    joints = np.repeat(home_q[None, :], n, axis=0)
    joints[:, claw_idx] = blended[:, 6:]
    body = blended[:, :6].copy()

    lo, hi = robot.range[:, 0], robot.range[:, 1]
    # Claws are authored. A number past a stop is a typo in the keyframe file,
    # and clipping it would hide which pose.
    claw = np.asarray(claw_idx)
    over = np.nonzero(
        (joints[:, claw] < lo[claw] - 1e-8) | (joints[:, claw] > hi[claw] + 1e-8)
    )
    if len(over[0]):
        i, col = int(over[0][0]), claw[int(over[1][0])]
        raise ValueError(
            f"{robot.names[col]} asks for {joints[i, col]:+.3f} rad at t={sim_t[i]:.3f} s, "
            f"outside [{lo[col]:+.3f}, {hi[col]:+.3f}]. The pose in "
            f"courtship_keyframes.py has to move, not the clip."
        )

    feet = np.zeros((n, 4, 3))
    tips = np.zeros((n, 2, 3))
    warm = home_q.copy()
    # Full tilt lifts the feet by TILT_FOOT_CLEAR_M; a smaller roll lifts less.
    # The site stays on the floor at zero roll, which is the stance and the bows.
    roll_scale = kf.TILT_FOOT_CLEAR_M / kf.TILT_RAD
    for i in range(n):
        foot_lift = roll_scale * abs(float(body[i, 3]))
        solved = robot.solve_feet(joints[i], body[i], warm, foot_lift=foot_lift)
        joints[i] = solved
        warm = solved
        foot_i, tip_i, _ = robot.feet_and_tips(solved, body[i])
        feet[i] = foot_i
        tips[i] = tip_i

    return Dance(
        time=sim_t,
        audio_time=audio_t,
        phase=_phase(audio_t, kf),
        body=body,
        joints=joints,
        feet=feet,
        tips=tips,
        names=list(robot.names),
        ranges=robot.range.copy(),
        foot_targets=np.stack([robot.targets[leg] for leg in SUPPORT]),
        labels=list(zip(knot_t.tolist(), labels)),
    )


def assess(dance: Dance) -> str:
    """The stats the generator prints. Raises ``ValueError`` on a violation.

    The speed check is the loader's, at the control rate, so a clip that prints
    a clean report is a clip ``motion.py`` will accept. The foot check is stricter
    than the loader on purpose: these feet were solved, and a millimetre of drift
    is the solve failing rather than a contact transient.
    """
    from tasks.jumper.common.actuator import CORNER_SPEED

    lines: list[str] = []
    problems: list[str] = []
    dt = CONTROL_DT
    n = len(dance.time)
    lines.append(
        f"courtship: {n} frames at {1.0 / dt:.0f} Hz, "
        f"last sample at {dance.time[-1]:.3f} s"
    )

    speed = np.abs(np.diff(dance.joints, axis=0)) / dt
    peak = speed.max(axis=0)
    order = np.argsort(peak)[::-1]
    lines.append(f"joint speed, peak over a control step (corner {CORNER_SPEED:.2f} rad/s):")
    for col in order:
        if peak[col] < 0.5 and col != order[0]:
            continue
        margin = CORNER_SPEED - float(peak[col])
        lines.append(
            f"  {dance.names[col]:16s} {peak[col]:6.2f} rad/s   "
            f"{margin:+6.2f} rad/s from the corner"
        )
    worst = int(speed.argmax()) if speed.size else 0
    worst_col = worst % speed.shape[1] if speed.size else 0
    worst_spd = float(speed.max()) if speed.size else 0.0
    if worst_spd > CORNER_SPEED:
        frame = worst // speed.shape[1]
        problems.append(
            f"{dance.names[worst_col]} moves {worst_spd:.1f} rad/s between control steps "
            f"{frame} and {frame + 1} (t={frame * dt:.2f} s), above the servo's "
            f"{CORNER_SPEED:.1f} rad/s corner speed"
        )

    lo, hi = dance.ranges[:, 0], dance.ranges[:, 1]
    margin = np.minimum(dance.joints - lo, hi - dance.joints).min(axis=0)
    closest = int(np.argmin(margin))
    lines.append("closest approach to a joint stop:")
    near = np.argsort(margin)[:6]
    for col in near:
        lines.append(
            f"  {dance.names[col]:16s} {margin[col] * 1000:7.1f} mrad inside "
            f"[{lo[col]:+.3f}, {hi[col]:+.3f}]"
        )
    outside = np.nonzero((dance.joints < lo - 1e-6) | (dance.joints > hi + 1e-6))
    if len(outside[0]):
        i, col = int(outside[0][0]), int(outside[1][0])
        problems.append(
            f"{dance.names[col]} is {dance.joints[i, col]:+.3f} rad at t={dance.time[i]:.3f} s, "
            f"outside [{lo[col]:+.3f}, {hi[col]:+.3f}]"
        )

    err = dance.feet - dance.foot_targets
    err_norm = np.linalg.norm(err, axis=2)
    z = dance.feet[:, :, 2]
    target_z = dance.foot_targets[:, 2]
    z_err = z - target_z
    spread = z.max(axis=1) - z.min(axis=1)
    lines.append(
        "support feet (LM, RM, LR, RR), against the stance point:"
    )
    lines.append(
        f"  position error  max {err_norm.max() * 1000:.2f} mm   "
        f"median {np.median(err_norm) * 1000:.2f} mm"
    )
    lines.append(
        f"  height drift    max {np.abs(z_err).max() * 1000:.2f} mm   "
        f"coplanar spread max {spread.max() * 1000:.2f} mm "
        f"(median {np.median(spread) * 1000:.2f} mm)"
    )
    if float(err_norm.max()) > FOOT_TOL_M:
        i, leg = np.unravel_index(int(err_norm.argmax()), err_norm.shape)
        problems.append(
            f"{SUPPORT[leg]} is {err_norm[i, leg] * 1000:.1f} mm off its stance point "
            f"at t={dance.time[i]:.3f} s (tolerance {FOOT_TOL_M * 1000:.0f} mm). "
            f"The body pose is more than the legs can hold with the feet planted; "
            f"lower TILT_RAD, BOW_PITCH_RAD or the sway."
        )

    roll = np.degrees(dance.body[:, 3])
    pitch = np.degrees(dance.body[:, 4])
    lines.append(
        f"body: roll {roll.min():+.1f} to {roll.max():+.1f} deg, "
        f"pitch {pitch.min():+.1f} to {pitch.max():+.1f} deg (negative is nose-up), "
        f"sway {dance.body[:, 1].min() * 1000:+.1f} to {dance.body[:, 1].max() * 1000:+.1f} mm, "
        f"height {dance.body[:, 2].min() * 1000:.1f} to {dance.body[:, 2].max() * 1000:.1f} mm"
    )
    # The bow is the most nose-up frame. forward_z was not stored; the sign of
    # pitch is the check, and it is negative by construction of BOW.
    bow = int(np.argmin(dance.body[:, 4]))
    lines.append(
        f"  most nose-up frame t={dance.time[bow]:.2f} s, "
        f"pitch {pitch[bow]:+.1f} deg, height {dance.body[bow, 2] * 1000:.1f} mm"
    )

    # Every mesh, not only the support sites. The sites can sit on their target
    # while a foot corner or a claw cuts the plane; that is what a render shows.
    robot = _Robot()
    non_z = np.empty(n)
    foot_z = np.empty(n)
    non_name = [""] * n
    foot_name = [""] * n
    shell_top = np.empty(n)
    shell_gid = robot.mujoco.mj_name2id(
        robot.model, robot.mujoco.mjtObj.mjOBJ_GEOM, "upper_shell_link_visual"
    )
    for i in range(n):
        (nz, nn), (fz, fn) = robot.lowest_meshes(dance.joints[i], dance.body[i])
        non_z[i], non_name[i] = nz, nn
        foot_z[i], foot_name[i] = fz, fn
        _lo, shell_top[i] = robot._mesh_z(shell_gid)
    near = int(np.argmin(non_z))
    foot_near = int(np.argmin(foot_z))
    lines.append(
        "meshes against the floor (z = 0):"
    )
    lines.append(
        f"  non-foot closest {non_z[near] * 1000:.2f} mm "
        f"({non_name[near]}) at t={dance.time[near]:.2f} s, "
        f"clearance line {NONFOOT_CLEAR_M * 1000:.0f} mm"
    )
    lines.append(
        f"  foot mesh lowest {foot_z[foot_near] * 1000:.2f} mm "
        f"({foot_name[foot_near]}) at t={dance.time[foot_near]:.2f} s"
    )
    if float(non_z[near]) < NONFOOT_CLEAR_M:
        problems.append(
            f"{non_name[near]} reaches {non_z[near] * 1000:.1f} mm at "
            f"t={dance.time[near]:.2f} s, under the {NONFOOT_CLEAR_M * 1000:.0f} mm "
            f"floor clearance for a link that is not a foot"
        )
    if float(foot_z[foot_near]) < -FOOT_MESH_TOL_M:
        problems.append(
            f"{foot_name[foot_near]} is {foot_z[foot_near] * 1000:.2f} mm below the floor "
            f"at t={dance.time[foot_near]:.2f} s. The foot site is held, and the mesh "
            f"around it has crossed the plane; TILT_FOOT_CLEAR_M is that lift"
        )
    high = int(np.argmax(dance.tips[:, :, 2].max(axis=1)))
    tip_z = float(dance.tips[high, :, 2].max())
    lines.append(
        f"  claw tip {tip_z * 1000:.0f} mm at t={dance.time[high]:.2f} s, "
        f"shell top {shell_top[high] * 1000:.0f} mm, "
        f"{(tip_z - shell_top[high]) * 1000:.0f} mm above the shell"
    )
    # The global peak is the shimmy, which rolls one claw higher. Beat 2 is the
    # pose the rise is judged on: both claws overhead, before any lean.
    kf = _load_keyframes()
    for label, when in (
        ("beat 1.5 mid-rise", kf.t(1, 0.5)),
        ("beat 2 overhead", kf.t(2)),
    ):
        i = int(np.argmin(np.abs(dance.audio_time - when)))
        z = float(dance.tips[i, :, 2].max())
        lines.append(
            f"  {label}: claw tip {z * 1000:.1f} mm at t={dance.time[i]:.2f} s, "
            f"shell top {shell_top[i] * 1000:.1f} mm, "
            f"{(z - shell_top[i]) * 1000:.1f} mm above the shell"
        )

    gap = np.linalg.norm(dance.tips[:, 0] - dance.tips[:, 1], axis=1)
    closest = int(np.argmin(gap))
    # The heart is the hold from beat 15 through beat 16. The flop starts after
    # that, and its first frames pass closer than the heart itself.
    heart_hold = (dance.audio_time >= kf.t(15)) & (dance.audio_time <= kf.t(16))
    heart_gap = float(np.median(gap[heart_hold])) if np.any(heart_hold) else float("nan")
    lines.append(
        f"claw tips: closest {gap[closest] * 1000:.1f} mm apart at t={dance.time[closest]:.2f} s. "
        f"Heart hold (beats 15-16) median {heart_gap * 1000:.1f} mm. "
        f"Left tip height {dance.tips[:, 0, 2].max() * 1000:.0f} mm at most, "
        f"right {dance.tips[:, 1, 2].max() * 1000:.0f} mm."
    )

    # What the reward stds in env_cfg.py are derived from. Position RMS is the
    # root-mean-square of each joint's deviation from its own mean, then the RMS
    # across joints -- jumper.dance's "RMS amplitude".
    centered = dance.joints - dance.joints.mean(axis=0)
    per_std = dance.joints.std(axis=0)
    rms = float(np.sqrt(np.mean(centered ** 2)))
    vel = speed
    vel_rms = float(np.sqrt(np.mean(vel ** 2)))
    vel_std = vel.std(axis=0)
    lines.append(
        f"clip, for env_cfg: position std median {np.median(per_std):.3f} rad, "
        f"max {per_std.max():.3f}, RMS amplitude {rms:.3f}; "
        f"velocity RMS {vel_rms:.3f} rad/s, per-joint std median {np.median(vel_std):.3f}, "
        f"max {vel_std.max():.3f}, peak {worst_spd:.1f}"
    )

    if problems:
        raise ValueError("courtship clip refused:\n  " + "\n  ".join(problems) + "\n" + "\n".join(lines))
    return "\n".join(lines)


def pack(dance: Dance) -> dict[str, np.ndarray]:
    """The npz schema. Command and ``meas_*`` are the same solved trajectory.

    Per-sample channels are float32, as ``motion.py`` asks generators to write.
    The beat grid and the scalars stay float64, which is how ``demo.npz`` stores
    them and how ``load_source`` reads them.
    """
    kf = _load_keyframes()
    from tasks.jumper.common.dance.motion import LEG_JOINTS

    f32 = np.float32
    n = len(dance.time)
    out: dict[str, np.ndarray] = {
        "time": dance.time.astype(f32),
        "dt": np.asarray(CONTROL_DT, dtype=np.float64),
        "phase": dance.phase.astype(f32),
    }
    keys = [f"L{leg}_j{j}" for leg in sorted(LEG_JOINTS) for j in range(len(LEG_JOINTS[leg]))]
    # ``dance.joints`` is already LEG_JOINTS order, which is L0..L5. Pin it.
    expected = [name for leg in sorted(LEG_JOINTS) for name in LEG_JOINTS[leg]]
    if dance.names != expected:
        raise RuntimeError(f"joint order is {dance.names}, expected {expected}")

    for prefix in ("", "meas_"):
        for axis, name in enumerate(BODY_NAMES):
            out[f"{prefix}body_{name}"] = dance.body[:, axis].astype(f32)
        for col, key in enumerate(keys):
            out[f"{prefix}{key}"] = dance.joints[:, col].astype(f32)
    out["bpm"] = np.asarray(kf.BPM, dtype=np.float64)
    out["beat_times_audio"] = np.asarray(kf.BEAT_TIMES_S, dtype=np.float64)
    out["beat_times_sim"] = np.asarray(kf.BEAT_TIMES_S, dtype=np.float64) + kf.LEAD_IN_S
    out["audio_start_in_sim"] = np.asarray(kf.LEAD_IN_S, dtype=np.float64)
    out["source"] = np.asarray("tools/make_courtship_dance.py")
    if n != len(dance.phase):
        raise RuntimeError("phase length drifted")
    return out


def write(dance: Dance, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **pack(dance))


def _wav_line(media: Path, kf) -> str:
    wavs = sorted(p for p in media.glob("*.wav") if p.is_file())
    if len(wavs) != 1:
        return f"music: expected one wav in {media}, found {len(wavs)}"
    with wave.open(str(wavs[0])) as handle:
        got = handle.getnframes() / float(handle.getframerate())
    return (
        f"music {wavs[0].name}: {got:.4f} s on disk, "
        f"{kf.MUSIC_S:.4f} s on the 16-beat grid ({(got - kf.MUSIC_S) * 1000:+.1f} ms)"
    )


def main(argv: list[str] | None = None) -> int:
    dest = MEDIA / "courtship.npz"
    if argv:
        if len(argv) == 2 and argv[0] == "--out":
            dest = Path(argv[1])
        else:
            print("usage: python tools/make_courtship_dance.py [--out PATH]", file=sys.stderr)
            return 2
    kf = _load_keyframes()
    print("schedule (audio time, seconds from beat 1):")
    for when, label in [(w, lab) for w, _p, lab in kf.SCHEDULE]:
        print(f"  {when:7.3f}  {label}")
    dance = build()
    try:
        report = assess(dance)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    print()
    print(report)
    print(_wav_line(dest.parent if dest.parent.is_dir() else MEDIA, kf))
    write(dance, dest)
    print(f"wrote {dest.relative_to(REPO)}  ({dest.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
