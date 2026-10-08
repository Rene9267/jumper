"""The courtship generator has to refuse a clip the dance loader would accept quietly.

`tools/make_courtship_dance.py` is the only writer of `jumper.dance_courtship`'s
clip. A joint that steps past the servo's corner speed, or a support foot that
leaves the floor while the report still says the feet are planted, would train
a dance the robot cannot do and the generator would still exit 0.

The checks themselves can pass vacuously -- a threshold nothing ever reaches --
so each one has a control that breaks it on purpose and must be refused.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

REPO = Path(__file__).resolve().parents[1]
GEN_PATH = REPO / "tools" / "make_courtship_dance.py"
MEDIA = REPO / "tasks" / "jumper" / "dance_courtship" / "media"
NPZ = MEDIA / "courtship.npz"


def _generator():
    spec = importlib.util.spec_from_file_location("courtship_gen", GEN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclass processing looks the module up; it has to be registered first.
    sys.modules["courtship_gen"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generated():
    """Two builds of the same clip. The solve is the slow part, so the module shares it."""
    gen = _generator()
    robot = gen._Robot()
    first = gen.build(robot)
    second = gen.build(robot)
    return gen, first, second


def test_generated_clip_matches_the_committed_file_and_stays_inside_limits(generated) -> None:
    """Regenerating the dance is deterministic, and the committed file is that result.

    The silent failure is a hand-edited npz, or a generator that writes a different
    clip on the second run: training would then be on a dance nobody can regenerate.
    """
    gen, first, second = generated
    assert np.array_equal(first.joints, second.joints)
    assert np.array_equal(first.body, second.body)
    report = gen.assess(first)
    assert "corner" in report

    packed = gen.pack(first)
    assert NPZ.is_file(), f"missing {NPZ}; run tools/make_courtship_dance.py"
    committed = np.load(NPZ)
    assert set(committed.files) == set(packed)
    for key in packed:
        assert np.array_equal(committed[key], packed[key]), key

    wavs = list(MEDIA.glob("*.wav"))
    npzs = list(MEDIA.glob("*.npz"))
    assert len(wavs) == 1 and wavs[0].name == "voxel_dance_cut.wav"
    assert len(npzs) == 1 and npzs[0].name == "courtship.npz"

    # The rise (beats 1-2) must not sweep the tips through each other. The heart
    # hold is the close pose, about 11 mm between fingertip origins.
    kf = gen._load_keyframes()
    gap = np.linalg.norm(first.tips[:, 0] - first.tips[:, 1], axis=1)
    rise = (first.audio_time >= kf.t(1)) & (first.audio_time <= kf.t(2))
    assert float(gap[rise].min()) > 0.040
    heart = (first.audio_time >= kf.t(15)) & (first.audio_time <= kf.t(16))
    assert 0.008 < float(np.median(gap[heart])) < 0.020


def test_a_joint_faster_than_the_corner_speed_is_refused(generated) -> None:
    """The speed check can fail. A one-frame spike of 0.8 rad is 40 rad/s."""
    gen, dance, _ = generated
    col = dance.names.index("LF_J1_joint")
    joints = dance.joints.copy()
    joints[1, col] += 0.8
    with pytest.raises(ValueError, match="corner speed"):
        gen.assess(replace(dance, joints=joints))


def test_a_foot_that_leaves_its_stance_point_is_refused(generated) -> None:
    """The foot check can fail. Two centimetres is ten times the 2 mm tolerance."""
    gen, dance, _ = generated
    feet = dance.feet.copy()
    feet[0, 0, 2] += 0.02
    with pytest.raises(ValueError, match="stance point"):
        gen.assess(replace(dance, feet=feet))
