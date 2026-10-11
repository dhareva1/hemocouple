import math
import shutil
import subprocess

import numpy as np
import pytest

from hemocouple import foam
from hemocouple.geometry import StenosisGeometry


@pytest.fixture
def spec():
    return foam.FoamCaseSpec(StenosisGeometry(3e-3, 0.5, 12e-3), upstream_length=6e-3, downstream_length=12e-3,
                             n_square=4, n_radial=3, axial_cell=1e-3)


def test_spec_sizes(spec):
    assert spec.length == pytest.approx(30e-3)
    assert spec.n_axial == 30
    assert spec.n_cells == (16 + 4 * 4 * 3) * 30
    # 16-gon inscribed in R = 3 mm
    assert spec.inlet_area == pytest.approx(0.5 * 16 * 9e-6 * math.sin(2 * math.pi / 16))
    assert spec.inlet_area < math.pi * 9e-6
    fine = spec.refined(2.0)
    assert (fine.n_square, fine.n_radial, fine.axial_cell) == (8, 6, 5e-4)


def test_case_files_and_stenosis_mapping(spec, tmp_path):
    case = tmp_path / "case"
    foam.write_case(case, spec, steady=False, delta_t=1e-4, end_time=1e-3, write_every=10)
    bmd = (case / "system" / "blockMeshDict").read_text()
    assert bmd.count("hex (") == 5 and bmd.count("arc ") == 8
    assert "timeVaryingMappedFixedValue" in (case / "0" / "U").read_text()
    assert str(case.resolve() / "constant" / "outletPressure.dat") in (case / "0" / "p").read_text()

    # Synthetic polyMesh/points: points on the wall at several z.
    z = np.linspace(0, spec.length, 31)
    pts = np.stack([np.full_like(z, spec.geometry.radius), np.zeros_like(z), z], -1)
    body = "\n".join(f"({x:.12g} {y:.12g} {zz:.12g})" for x, y, zz in pts)
    pm = case / "constant" / "polyMesh"
    pm.mkdir(parents=True)
    (pm / "points").write_text(foam.HEADER.format(cls="vectorField", obj="points") + f"\n\n{len(pts)}\n(\n{body}\n)\n\n// end\n")
    foam.apply_stenosis(case, spec)
    head, out, tail = foam._read_points(pm / "points")
    assert "FoamFile" in head and "// end" in tail
    r = np.hypot(out[:, 0], out[:, 1])
    np.testing.assert_allclose(r, spec.geometry.radius_profile(z, spec.stenosis_center), rtol=1e-10)
    assert r.min() == pytest.approx(spec.geometry.throat_radius, rel=1e-3)


def test_boundary_data_format(tmp_path, grid):
    foam.write_inlet_points(tmp_path, grid, 3e-3, n_theta=16, n_wall=8)
    n_radii = len(np.unique(grid.rho))
    n = 1 + n_radii * 16 + 8
    radius = 3e-3
    u = 2.0 * (1 - grid.rho**2)
    foam.write_inlet_profile(tmp_path, 0.0125, u, q=5e-6, inlet_area=2.5e-5, grid=grid, n_theta=16, n_wall=8)
    bd = tmp_path / "constant" / "boundaryData" / "inlet"
    pts = (bd / "points").read_text().splitlines()
    assert int(pts[0]) == n
    vals = (bd / "0.0125" / "U").read_text()
    assert vals.startswith(f"{n}\n(")
    assert vals.strip().splitlines()[-1] == "(0 0 0.2)"


def test_upsampled_profile_preserves_axisymmetric_rings(grid):
    u = 2.0 * (1 - grid.rho**2) + 0.1 * grid.rho
    dense = foam.upsample_profile(grid, u, n_theta=32)
    radii = np.unique(grid.rho)
    expected = np.broadcast_to((2.0 * (1 - radii**2) + 0.1 * radii)[:, None], (len(radii), 32))
    np.testing.assert_allclose(dense[1:].reshape(len(radii), 32), expected)
    assert dense[0] == pytest.approx(2.0 * (1 - radii[0] ** 2) + 0.1 * radii[0])


def _have_foam():
    if shutil.which("docker") is None:
        return False
    out = subprocess.run(["docker", "images", "-q", foam.FOAM_IMAGE], capture_output=True, text=True)
    return out.returncode == 0 and bool(out.stdout.strip())


@pytest.mark.skipif(not _have_foam(), reason="needs Docker and the OpenFOAM image")
def test_blockmesh_and_checkmesh(spec, tmp_path_factory):
    root = tmp_path_factory.mktemp("foam")
    runner = foam.DockerFoam(root, name="hemocouple-foam-test").start()
    try:
        case = root / "case"
        foam.write_case(case, spec, steady=False, delta_t=1e-4, end_time=1e-3, write_every=10)
        log = foam.build_mesh(runner, case, spec)
        assert f"cells:            {spec.n_cells}" in log
        assert "Mesh OK" in log
    finally:
        runner.stop()


def test_radial_grading_written_for_outer_blocks_only(tmp_path):
    spec = foam.FoamCaseSpec(StenosisGeometry(3e-3, 0.5, 12e-3), 6e-3, 12e-3, n_square=4, n_radial=3,
                             axial_cell=1e-3, radial_grading=4.0)
    assert spec.refined(1.5).radial_grading == 4.0
    foam.write_case(tmp_path / "c", spec, steady=False, delta_t=1e-4, end_time=1e-3, write_every=10)
    hexes = [l for l in (tmp_path / "c" / "system" / "blockMeshDict").read_text().splitlines() if "hex (" in l]
    assert hexes[0].endswith("simpleGrading (1 1 1)")
    assert all(h.endswith("simpleGrading (1 4 1)") for h in hexes[1:])


def test_accept_keeps_drifted_time_directory(spec, grid, tmp_path):
    """OpenFOAM may write t_next as e.g. 0.161206678063 when 0.161206678064 was requested (12-digit names)."""
    dt = 0.80603339032 / 100
    loc = foam.FoamLocal3D(None, tmp_path / "c", spec, grid, coupling_dt=dt, substeps=20,
                           distal_resistance=2e9, distal_compliance=1e-10, q0=6e-6, p_distal0=1.2e4)
    loc.t = 19 * dt
    for name in ("0", "0.15314634416", "0.161206678063"):
        (tmp_path / "c" / name).mkdir(parents=True)
    bd = tmp_path / "c" / "constant" / "boundaryData" / "inlet"
    for name in ("0.145085", "0.153146344161", "0.161206678064"):
        (bd / name).mkdir(parents=True)
    loc._trial = dict(t=20 * dt, q=6e-6, p_distal=1.2e4, p_in=1.2e4, q_mesh=6e-6, sample={})
    loc.accept()
    assert sorted(d.name for d in (tmp_path / "c").iterdir() if d.name != "constant") == ["0", "0.161206678063"]
    assert sorted(d.name for d in bd.iterdir()) == ["0.153146344161", "0.161206678064"]


def test_accept_refuses_to_prune_when_nothing_was_written(spec, grid, tmp_path):
    dt = 0.01
    loc = foam.FoamLocal3D(None, tmp_path / "c", spec, grid, coupling_dt=dt, substeps=20,
                           distal_resistance=2e9, distal_compliance=1e-10, q0=6e-6, p_distal0=1.2e4)
    loc.t = dt
    for name in ("0", "0.01"):
        (tmp_path / "c" / name).mkdir(parents=True)
    (tmp_path / "c" / "constant" / "boundaryData" / "inlet").mkdir(parents=True)
    loc._trial = dict(t=2 * dt, q=6e-6, p_distal=1.2e4, p_in=1.2e4, q_mesh=6e-6, sample={})
    with pytest.raises(RuntimeError, match="did not write"):
        loc.accept()
    assert (tmp_path / "c" / "0.01").exists()
