"""The crab skin is a picture, and a picture that changes the robot is silent.

`jumper_crab.xml` is generated from `jumper.xml` and selected with `--model crab`.
Nothing in training reads the texture, so a skin that also moved a mass, a
friction or a joint limit would train a different robot and report no error.
The comparison below is exact — the same arrays, not a tolerance — and the
control group shows it can fail.

The other quiet failure is the skin becoming the default, or the black display,
camera and ToF being painted over, so the machine is no longer recognisable.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
from PIL import Image

import tasks
from tasks.jumper.common.assets import CRAB_XML, JUMPER_XML

REPO = Path(__file__).resolve().parents[1]
TEXTURES = REPO / "assets" / "jumper" / "textures"
_BUILDER = REPO / "assets" / "jumper" / "tools" / "build_skin.py"


def _builder():
    spec = importlib.util.spec_from_file_location("build_skin", _BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclass looks the class up in sys.modules while the body is executing.
    sys.modules["build_skin"] = module
    spec.loader.exec_module(module)
    return module


def _names(model: mujoco.MjModel, kind: int, n: int) -> list[str]:
    return [mujoco.mj_id2name(model, kind, i) or "" for i in range(n)]


def test_crab_skin_matches_jumper_physics() -> None:
    """Mass, inertia, collision and the joint and actuator arrays are shared.

    jumper.xml has no actuator section, so both actuator arrays are empty. That
    is still the check: a skin that added one would not compare equal.
    """
    base = mujoco.MjModel.from_xml_path(str(JUMPER_XML))
    crab = mujoco.MjModel.from_xml_path(str(CRAB_XML))
    # The two files are different pictures. Without this, comparing a model to
    # itself would pass the loop below and prove nothing.
    assert not np.array_equal(base.geom_rgba, crab.geom_rgba)
    assert base.nmat == 0 and crab.nmat > 0

    kind = mujoco.mjtObj
    for obj, n in (
        (kind.mjOBJ_BODY, "nbody"),
        (kind.mjOBJ_JOINT, "njnt"),
        (kind.mjOBJ_GEOM, "ngeom"),
        (kind.mjOBJ_ACTUATOR, "nu"),
    ):
        count = getattr(base, n)
        assert count == getattr(crab, n)
        assert _names(base, obj, count) == _names(crab, obj, count)
    assert base.nq == crab.nq and base.nv == crab.nv

    mod = _builder()
    fields = mod.PHYSICS_FIELDS
    assert "geom_size" not in fields
    for name in fields:
        got = np.asarray(getattr(base, name))
        other = np.asarray(getattr(crab, name))
        assert got.shape == other.shape, name
        np.testing.assert_array_equal(got, other, err_msg=name)

    # Visual meshes are rewritten as OBJ so they can carry texcoords, and
    # geom_size is the float32 bound of those vertices. Collision geoms stay
    # the STLs and compare exact. The control rejects a millimetre, so the
    # tolerance is not a way for a skin to resize the robot.
    assert mod.geom_size_error(base, crab) is None
    names = _names(base, mujoco.mjtObj.mjOBJ_GEOM, base.ngeom)
    col = next(i for i, name in enumerate(names) if name.endswith("_meshcol"))
    saved = np.array(crab.geom_size[col], copy=True)
    crab.geom_size[col, 0] = saved[0] + 1e-4
    assert mod.geom_size_error(base, crab) is not None
    crab.geom_size[col] = saved
    vis = next(i for i, name in enumerate(names) if name == "upper_shell_link_visual")
    saved = np.array(crab.geom_size[vis], copy=True)
    crab.geom_size[vis, 2] = saved[2] + 1e-3
    assert mod.geom_size_error(base, crab) is not None
    crab.geom_size[vis] = saved

    # Control: the same comparison rejects a one-body mass change, so an empty
    # field list or a model compared with itself is not what passed above.
    bumped = np.array(base.body_mass, copy=True)
    bumped[1] += 0.05
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(bumped, np.asarray(crab.body_mass))


def test_sensors_stay_black_and_groups_take_their_materials() -> None:
    """Display, camera and ToF stay black. Everything else is a named material.

    Collision geoms take a flat rgba and no material, because a material would
    override the rgba `--strip-visual` is supposed to show.
    """
    mod = _builder()
    crab = mujoco.MjModel.from_xml_path(str(CRAB_XML))
    shell = ventral = tip = None
    for i in range(crab.ngeom):
        name = mujoco.mj_id2name(crab, mujoco.mjtObj.mjOBJ_GEOM, i)
        body = mujoco.mj_id2name(crab, mujoco.mjtObj.mjOBJ_BODY, int(crab.geom_bodyid[i]))
        role = mod.appearance(body)
        if name.endswith("_meshcol"):
            assert int(crab.geom_matid[i]) == -1, name
        elif name.endswith("_visual"):
            if role == "sensor":
                assert int(crab.geom_matid[i]) == -1, name
                np.testing.assert_allclose(crab.geom_rgba[i], [0, 0, 0, 1])
            else:
                mat = mujoco.mj_id2name(
                    crab, mujoco.mjtObj.mjOBJ_MATERIAL, int(crab.geom_matid[i]))
                assert mat == f"crab_{role}", name
        else:
            raise AssertionError(f"unexpected geom {name}")
        if name == "upper_shell_link_meshcol":
            shell = crab.geom_rgba[i, :3].mean()
        elif name == "base_link_meshcol":
            ventral = crab.geom_rgba[i, :3].mean()
        elif name == "LF_finger_tip_link_meshcol":
            tip = crab.geom_rgba[i, :3].max()
    assert tip is not None and shell is not None and ventral is not None
    # Tips darker than the carapace, the carapace darker than the underside.
    # A swapped assignment still compiles and still has materials.
    assert tip < shell < ventral


def test_the_shell_is_box_mapped_and_the_colours_are_not_black() -> None:
    """A cube map pinches the dome, and a dark rgba makes a texture read as black.

    The visual STLs have no texcoords. The skin bakes box coordinates into OBJ
    copies and samples a tileable picture with `texuniform` off. The carapace's
    darker half has to land near sRGB 90–120, and the belly, thighs and forearms
    stay a pale cream rather than white or a saturated orange.
    """
    base = mujoco.MjModel.from_xml_path(str(JUMPER_XML))
    assert int(np.max(base.mesh_texcoordnum)) == 0, "the premise: the STLs grew UVs"

    crab = mujoco.MjModel.from_xml_path(str(CRAB_XML))
    flat = int(mujoco.mjtTexture.mjTEXTURE_2D)
    assert crab.ntex == 3
    assert {int(t) for t in crab.tex_type} == {flat}
    assert not bool(np.any(crab.mat_texuniform))
    assert float(np.max(crab.mat_reflectance)) == 0.0

    mod = _builder()
    shell = mujoco.mj_name2id(crab, mujoco.mjtObj.mjOBJ_MESH, "upper_shell_link")
    assert int(crab.mesh_texcoordnum[shell]) > 1000
    sensor = mujoco.mj_name2id(crab, mujoco.mjtObj.mjOBJ_MESH, "display_module_link")
    assert int(crab.mesh_texcoordnum[sensor]) == 0
    # A collider that shares its STL with the visual keeps that STL. The texcoords
    # live on a second mesh, so rewriting the picture cannot move the hull.
    pad = mujoco.mj_name2id(crab, mujoco.mjtObj.mjOBJ_MESH, "LF_palm_pad_b_link")
    assert int(crab.mesh_texcoordnum[pad]) == 0
    assert int(crab.mesh_texcoordnum[
        mujoco.mj_name2id(crab, mujoco.mjtObj.mjOBJ_MESH, "LF_palm_pad_b_link_skin")
    ]) > 0

    assert mod.appearance("LM_thigh_link") == "leg_joint"
    assert mod.appearance("LF_forearm_link") == "claw_joint"
    assert mod.appearance("LM_calf_link") == "leg"

    for i in range(crab.ngeom):
        name = mujoco.mj_id2name(crab, mujoco.mjtObj.mjOBJ_GEOM, i)
        body = mujoco.mj_id2name(crab, mujoco.mjtObj.mjOBJ_BODY, int(crab.geom_bodyid[i]))
        if not name.endswith("_visual") or mod.appearance(body) == "sensor":
            continue
        # Near white. A dark rgba multiplies the picture and the marble goes black.
        assert float(crab.geom_rgba[i, :3].min()) > 0.85, name

    carapace = np.asarray(Image.open(TEXTURES / "carapace.png")).reshape(-1, 3)
    lum = carapace.mean(axis=1)
    dark = carapace[lum <= np.median(lum)].mean()
    assert 90.0 <= dark <= 120.0, dark
    assert float(lum.std()) > 12.0, "a flat brown would pass the mean and show no marble"

    ventral = np.asarray(Image.open(TEXTURES / "ventral.png")).reshape(-1, 3).astype(np.float64)
    hi = ventral.max(axis=1)
    sat = (hi - ventral.min(axis=1)) / np.maximum(hi, 1.0)
    assert float(sat.mean()) < 0.25, sat.mean()
    # Pale cream-ochre. Above ~190 the beach sun clips it to white plastic.
    assert 150.0 <= float(ventral.mean()) <= 185.0


def test_every_link_has_a_role() -> None:
    """A link the painter does not know would keep the factory red and look intentional."""
    mod = _builder()
    spec = mujoco.MjSpec.from_file(str(JUMPER_XML))
    seen = []
    for body in spec.bodies:
        if body.name in ("", "world"):
            continue
        seen.append(mod.appearance(body.name))
    assert seen, "the model had no links"
    assert "sensor" in seen and "shell" in seen and "tip" in seen and "leg" in seen


def test_crab_is_selectable_and_is_not_the_default() -> None:
    """Every jumper task, the dance tasks included, lists crab and still defaults to jumper."""
    specs = [s for s in tasks.all_specs() if s.id.startswith("jumper.")]
    assert len(specs) >= 10, "control: an empty registry would pass the loop"
    dance = [s.id for s in specs if s.id == "jumper.dance" or s.id.startswith("jumper.dance_")]
    assert "jumper.dance" in dance
    for spec in specs:
        assert spec.assets[0].name == "jumper"
        assert spec.resolve_asset(None) == JUMPER_XML
        assert spec.resolve_asset("crab") == CRAB_XML
    with pytest.raises(KeyError, match="crab"):
        tasks.get("jumper.dance").resolve_asset("no-such-skin")


def compare_committed_skin() -> None:
    """Rebuild the skin text and require it to match the committed files.

    Does not write them. Meant to run in a fresh interpreter: after another
    model has been compiled in-process, MuJoCo's ``to_xml`` drops
    ``content_type`` on the meshes, so the same function later in a pytest
    file would disagree with a file the script wrote on its own.
    """
    mod = _builder()
    for name, draw in mod.TEXTURES.items():
        got = draw()
        have = np.asarray(Image.open(TEXTURES / f"{name}.png"))
        if got.shape != have.shape or not np.array_equal(got, have):
            raise SystemExit(f"texture {name} does not match the generator")
    for name, got in mod.baked_meshes().items():
        path = mod.SKIN_DIR / name
        if got != path.read_bytes():
            raise SystemExit(f"mesh {name} does not match the generator")
    spec = mujoco.MjSpec.from_file(str(JUMPER_XML))
    images = {name: draw() for name, draw in mod.TEXTURES.items()}
    mod._paint(spec, images)
    fresh = mod._xml_text(spec)
    disk = CRAB_XML.read_text(encoding="utf-8")
    if fresh != disk:
        raise SystemExit("jumper_crab.xml does not match a fresh generation")


def test_committed_skin_matches_the_generator() -> None:
    """A hand-edited XML or PNG does not look hand-edited. Regenerating does."""
    probe = subprocess.run(
        [sys.executable, "-c", "import tests.test_crab_skin as t; t.compare_committed_skin()"],
        check=False, cwd=REPO, capture_output=True, text=True,
    )
    assert probe.returncode == 0, probe.stderr or probe.stdout
