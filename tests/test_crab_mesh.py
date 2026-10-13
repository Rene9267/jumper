"""Mesh v2 (Calappa overlays) and the female skin: appearance only.

`jumper_crab_mesh.xml` adds visual-only meshes to the crab skin and
`jumper_crab_femaleB.xml` swaps its textures. Neither may change the mechanism.
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

import tasks
from tasks.jumper.common.assets import CRAB_FEMALE_XML, CRAB_MESH_XML, CRAB_XML

OVERLAYS = ("calappa_carapace_visual", "LF_claw_granules_visual", "RF_claw_granules_visual")


def _load(path):
    return mujoco.MjModel.from_xml_path(str(path))


@pytest.fixture(scope="module")
def models():
    return _load(CRAB_XML), _load(CRAB_MESH_XML), _load(CRAB_FEMALE_XML)


def test_same_mechanism(models) -> None:
    base, *variants = models
    for m in variants:
        assert (m.nbody, m.njnt, m.nu, m.nq) == (base.nbody, base.njnt, base.nu, base.nq)
        np.testing.assert_array_equal(m.body_mass, base.body_mass)
        np.testing.assert_array_equal(m.body_inertia, base.body_inertia)
        np.testing.assert_array_equal(m.jnt_range, base.jnt_range)
        np.testing.assert_array_equal(m.actuator_gear, base.actuator_gear)


def test_overlays_have_no_contacts_and_no_mass(models) -> None:
    for m in models[1:]:
        for name in OVERLAYS:
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert g >= 0, name
            assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0


def test_colliding_geoms_unchanged(models) -> None:
    base, *variants = models
    def colliders(m):
        ids = [i for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]]
        return np.array([m.geom_size[i] for i in ids]), np.array([m.geom_friction[i] for i in ids])
    bs, bf = colliders(base)
    for m in variants:
        s, f = colliders(m)
        np.testing.assert_array_equal(s, bs)
        np.testing.assert_array_equal(f, bf)


def test_female_differs_only_in_textures() -> None:
    a = CRAB_MESH_XML.read_text(encoding="utf-8").splitlines()
    b = [l for l in CRAB_FEMALE_XML.read_text(encoding="utf-8").splitlines()]
    b = [l for l in b if not l.startswith(("<!-- Female", "     femaleB_legs"))]
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(a) == len(b)
    assert diff and all("<texture" in x and "<texture" in y for x, y in diff)


def test_new_models_selectable() -> None:
    spec = tasks.get("jumper.dance")
    assert spec.resolve_asset("crab_mesh") == CRAB_MESH_XML
    assert spec.resolve_asset("crab_female") == CRAB_FEMALE_XML
    assert spec.resolve_asset(None).name == "jumper.xml"
