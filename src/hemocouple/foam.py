"""OpenFOAM (pimpleFoam / simpleFoam) local 3D domain, run inside Docker (proposal 7.2).

Case layout: rigid straight tube, interface inlet at z = 0, smooth cosine
stenosis centred at z = upstream_length + length / 2, outlet at z = L.

- Mesh: blockMesh O-grid (centre square + 4 outer blocks), then the points are
  scaled radially by R(z) / R0 to make the stenosis.
- Inlet: timeVaryingMappedFixedValue from constant/boundaryData/inlet/<t>/U.
  The points are the reference CrossSectionGrid plus a no-slip wall ring;
  setAverage makes the discrete inlet flux exactly Q.
- Outlet: pressure from a Python-side RC Windkessel through a tableFile.
  Rigid wall + incompressible means Q_out = Q_in, so it is implicit in Q.
- Returned interface pressure: area-averaged static pressure on the inlet.

OpenFOAM works in kinematic pressure (p / rho); this module converts at its
boundary so the coupling sees Pa.

``FoamLocal3D`` implements the coupling ``LocalModel`` protocol. Each trial
restarts the solver from the last accepted time directory. This is simple and
robust for the M1 pilot, but the per-step restart overhead is what a preCICE
adapter would remove later.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .cross_section import CrossSectionGrid
from .geometry import StenosisGeometry

FOAM_IMAGE = "opencfd/openfoam-run:2412"
FOAM_BASHRC = "/usr/lib/openfoam/openfoam2412/etc/bashrc"

HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


# ---------------------------------------------------------------------------
# Running OpenFOAM in Docker
# ---------------------------------------------------------------------------
class DockerFoam:
    """Long-lived container with ``root`` bind-mounted at the same path inside."""

    def __init__(self, root: Path, image: str = FOAM_IMAGE, name: str = "hemocouple-foam", bashrc: str = FOAM_BASHRC):
        self.root, self.image, self.name, self.bashrc = Path(root).resolve(), image, name, bashrc

    def start(self) -> "DockerFoam":
        running = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name=^{self.name}$"], capture_output=True, text=True, check=True
        ).stdout.strip()
        if not running:
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
            subprocess.run(
                ["docker", "run", "-d", "--name", self.name, "--entrypoint", "sleep",
                 "-v", f"{self.root}:{self.root}", self.image, "infinity"],
                check=True, capture_output=True,
            )
        return self

    def stop(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)

    def run(self, case: Path, command: str, log: Optional[str] = None, check: bool = True) -> subprocess.CompletedProcess:
        redirect = f" > {log} 2>&1" if log else ""
        script = f"source {self.bashrc} >/dev/null 2>&1; cd {Path(case).resolve()} && {command}{redirect}"
        proc = subprocess.run(["docker", "exec", self.name, "bash", "-c", script], capture_output=True, text=True)
        if check and proc.returncode != 0:
            tail = ""
            if log:
                logfile = Path(case) / log
                tail = logfile.read_text()[-3000:] if logfile.exists() else ""
            raise RuntimeError(f"OpenFOAM command failed: {command}\n{proc.stderr[-2000:]}\n{tail}")
        return proc


# ---------------------------------------------------------------------------
# Case specification and writers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FoamCaseSpec:
    geometry: StenosisGeometry
    upstream_length: float  # m, inlet (interface) to stenosis start
    downstream_length: float  # m, stenosis end to outlet
    n_square: int = 8  # cells along each side of the centre block (and tangentially per outer block)
    n_radial: int = 8  # cells across the outer blocks
    axial_cell: float = 3.0e-4  # m
    density: float = 1060.0
    viscosity: float = 3.5e-3
    radial_grading: float = 1.0  # outer-block cell size ratio, centre side / wall side (> 1 refines toward the wall)

    @property
    def length(self) -> float:
        return self.upstream_length + self.geometry.length + self.downstream_length

    @property
    def stenosis_center(self) -> float:
        return self.upstream_length + self.geometry.length / 2

    @property
    def n_axial(self) -> int:
        return max(1, int(round(self.length / self.axial_cell)))

    @property
    def n_cells(self) -> int:
        return (self.n_square**2 + 4 * self.n_square * self.n_radial) * self.n_axial

    @property
    def inlet_area(self) -> float:
        """Discrete inlet area: polygon through the 4 * n_square wall points (faces tile it exactly)."""
        n = 4 * self.n_square
        return 0.5 * n * self.geometry.radius**2 * math.sin(2 * math.pi / n)

    def refined(self, factor: float) -> "FoamCaseSpec":
        return FoamCaseSpec(
            self.geometry, self.upstream_length, self.downstream_length,
            max(2, int(round(self.n_square * factor))), max(2, int(round(self.n_radial * factor))),
            self.axial_cell / factor, self.density, self.viscosity, self.radial_grading,
        )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _block_mesh_dict(spec: FoamCaseSpec) -> str:
    R, L = spec.geometry.radius, spec.length
    a = 0.5 * R / math.sqrt(2.0)
    b = R / math.sqrt(2.0)
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    verts = []
    for z in (0.0, L):
        verts += [(sx * a, sy * a, z) for sx, sy in corners]
        verts += [(sx * b, sy * b, z) for sx, sy in corners]
    vtxt = "\n".join(f"    ({x:.10g} {y:.10g} {z:.10g})" for x, y, z in verts)
    ns, nr, nz = spec.n_square, spec.n_radial, spec.n_axial
    blocks = [
        ((0, 1, 2, 3), (ns, ns)),
        ((4, 5, 1, 0), (ns, nr)),
        ((5, 6, 2, 1), (ns, nr)),
        ((6, 7, 3, 2), (ns, nr)),
        ((7, 4, 0, 3), (ns, nr)),
    ]
    # Outer blocks run wall -> centre in their second direction, so grading g > 1 puts the small cells at the wall.
    btxt = "\n".join(
        f"    hex ({' '.join(map(str, base))} {' '.join(str(v + 8) for v in base)}) ({n1} {n2} {nz}) "
        f"simpleGrading (1 {1.0 if i == 0 else spec.radial_grading:.6g} 1)"
        for i, (base, (n1, n2)) in enumerate(blocks)
    )
    edges = []
    for off, z in ((0, 0.0), (8, L)):
        for (i, j), (mx, my) in zip(((4, 5), (5, 6), (6, 7), (7, 4)), ((0, -R), (R, 0), (0, R), (-R, 0))):
            edges.append(f"    arc {i + off} {j + off} ({mx:.10g} {my:.10g} {z:.10g})")
    inlet = ["(0 3 2 1)", "(4 0 1 5)", "(5 1 2 6)", "(6 2 3 7)", "(7 3 0 4)"]
    outlet = ["(8 9 10 11)", "(12 13 9 8)", "(13 14 10 9)", "(14 15 11 10)", "(15 12 8 11)"]
    wall = ["(4 5 13 12)", "(5 6 14 13)", "(6 7 15 14)", "(7 4 12 15)"]

    def patch(name, kind, faces):
        f = "\n".join(f"            {x}" for x in faces)
        return f"    {name}\n    {{\n        type {kind};\n        faces\n        (\n{f}\n        );\n    }}"

    return (
        HEADER.format(cls="dictionary", obj="blockMeshDict")
        + f"scale 1;\n\nvertices\n(\n{vtxt}\n);\n\nblocks\n(\n{btxt}\n);\n\nedges\n(\n"
        + "\n".join(edges)
        + "\n);\n\nboundary\n(\n"
        + "\n".join([patch("inlet", "patch", inlet), patch("outlet", "patch", outlet), patch("wall", "wall", wall)])
        + "\n);\n"
    )


def _control_dict(
    app: str,
    delta_t: float,
    end_time: float,
    write_every: int,
    steady: bool,
    max_co: Optional[float] = None,
    write_interval_time: Optional[float] = None,
    extra_functions: str = "",
) -> str:
    """Fixed step written every ``write_every`` steps, or (max_co set) adaptive step written every ``write_interval_time``."""
    fo = """
functions
{
    inletP
    {
        type            surfaceFieldValue;
        libs            (fieldFunctionObjects);
        regionType      patch;
        name            inlet;
        operation       areaAverage;
        fields          (p);
        writeFields     false;
        log             false;
        writeControl    timeStep;
        writeInterval   1;
    }
    outletP
    {
        $inletP;
        name            outlet;
    }
    flux
    {
        type            surfaceFieldValue;
        libs            (fieldFunctionObjects);
        regionType      patch;
        name            inlet;
        operation       sum;
        fields          (phi);
        writeFields     false;
        log             false;
        writeControl    timeStep;
        writeInterval   1;
    }
""" + extra_functions + """
}
"""
    if max_co is None:
        timing = f"writeControl    timeStep;\nwriteInterval   {write_every};\n"
    else:
        timing = (f"adjustTimeStep  yes;\nmaxCo           {max_co:.6g};\nmaxDeltaT       1;\n"
                  f"writeControl    adjustableRunTime;\nwriteInterval   {write_interval_time:.12g};\n")
    return HEADER.format(cls="dictionary", obj="controlDict") + f"""
application     {app};
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {end_time:.12g};
deltaT          {delta_t:.12g};
{timing}purgeWrite      0;
writeFormat     ascii;
writePrecision  12;
writeCompression off;
timeFormat      general;
timePrecision   12;
runTimeModifiable false;
""" + fo


def _fv_schemes(steady: bool) -> str:
    ddt = "steadyState" if steady else "backward"
    div_u = "bounded Gauss linearUpwind grad(U)" if steady else "Gauss linearUpwind grad(U)"
    return HEADER.format(cls="dictionary", obj="fvSchemes") + f"""
ddtSchemes      {{ default {ddt}; }}
gradSchemes     {{ default Gauss linear; }}
divSchemes
{{
    default         none;
    div(phi,U)      {div_u};
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}
laplacianSchemes {{ default Gauss linear limited corrected 0.5; }}
interpolationSchemes {{ default linear; }}
snGradSchemes   {{ default limited corrected 0.5; }}
"""


def _fv_solution(steady: bool) -> str:
    algo = (
        """SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent      yes;
    residualControl { p 1e-6; U 1e-7; }
}
relaxationFactors { equations { U 0.9; \".*\" 0.9; } fields { p 1; } }
"""
        if steady
        else """PIMPLE
{
    nOuterCorrectors 1;
    nCorrectors     2;
    nNonOrthogonalCorrectors 1;
}
"""
    )
    return HEADER.format(cls="dictionary", obj="fvSolution") + """
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.01;
    }
    pFinal
    {
        $p;
        relTol          0;
    }
    "(U|UFinal)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-9;
        relTol          0;
    }
}
""" + algo


def _fields(outlet_table: str) -> Tuple[str, str]:
    u = HEADER.format(cls="volVectorField", obj="U") + """
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{
    inlet
    {
        type            timeVaryingMappedFixedValue;
        setAverage      true;
        offset          (0 0 0);
        mapMethod       planarInterpolation;
        perturb         1e-5;
    }
    outlet { type zeroGradient; }
    wall   { type noSlip; }
}
"""
    p = HEADER.format(cls="volScalarField", obj="p") + f"""
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{{
    inlet  {{ type zeroGradient; }}
    outlet
    {{
        type            uniformFixedValue;
        uniformValue    tableFile;
        uniformValueCoeffs
        {{
            file            "{outlet_table}";
            outOfBounds     clamp;
        }}
        value           uniform 0;
    }}
    wall   {{ type zeroGradient; }}
}}
"""
    return u, p


def write_case(case: Path, spec: FoamCaseSpec, *, steady: bool, delta_t: float, end_time: float, write_every: int) -> None:
    case = Path(case).resolve()
    if case.exists():
        shutil.rmtree(case)
    app = "simpleFoam" if steady else "pimpleFoam"
    nu = spec.viscosity / spec.density
    table = case / "constant" / "outletPressure.dat"
    _write(case / "system" / "blockMeshDict", _block_mesh_dict(spec))
    _write(case / "system" / "controlDict", _control_dict(app, delta_t, end_time, write_every, steady))
    _write(case / "system" / "fvSchemes", _fv_schemes(steady))
    _write(case / "system" / "fvSolution", _fv_solution(steady))
    _write(case / "constant" / "transportProperties",
           HEADER.format(cls="dictionary", obj="transportProperties") + f"\ntransportModel Newtonian;\nnu {nu:.10g};\n")
    _write(case / "constant" / "turbulenceProperties",
           HEADER.format(cls="dictionary", obj="turbulenceProperties") + "\nsimulationType laminar;\n")
    u, p = _fields(str(table))
    _write(case / "0" / "U", u)
    _write(case / "0" / "p", p)
    _write(table, "(\n(0 0)\n)\n")


def _read_points(path: Path) -> Tuple[str, np.ndarray, str]:
    """Split an ASCII polyMesh/points file into (header, points, trailer)."""
    text = path.read_text()
    m = re.search(r"\n(\d+)\s*\n\(", text)
    n = int(m.group(1))
    end = text.index("\n)", m.end())
    arr = np.array(" ".join(re.findall(r"\(([^()]+)\)", text[m.end() : end])).split(), dtype=float).reshape(-1, 3)
    assert len(arr) == n, (len(arr), n)
    return text[: m.start()], arr, text[end + 2 :]


def apply_stenosis(case: Path, spec: FoamCaseSpec) -> None:
    """Scale (x, y) of every mesh point by R(z) / R0."""
    path = Path(case) / "constant" / "polyMesh" / "points"
    head, pts, tail = _read_points(path)
    scale = spec.geometry.radius_profile(pts[:, 2], spec.stenosis_center) / spec.geometry.radius
    pts[:, 0] *= scale
    pts[:, 1] *= scale
    body = "\n".join(f"({x:.12g} {y:.12g} {z:.12g})" for x, y, z in pts)
    path.write_text(f"{head}\n{len(pts)}\n(\n{body}\n){tail}")


def build_mesh(foam: DockerFoam, case: Path, spec: FoamCaseSpec) -> str:
    foam.run(case, "blockMesh", log="log.blockMesh")
    apply_stenosis(case, spec)
    foam.run(case, "checkMesh -constant", log="log.checkMesh", check=False)
    return (Path(case) / "log.checkMesh").read_text()


# ---------------------------------------------------------------------------
# Boundary data and result parsing
# ---------------------------------------------------------------------------
def _fmt_time(t: float) -> str:
    return f"{t:.12g}"


def _rings(grid: CrossSectionGrid) -> Tuple[np.ndarray, int]:
    radii = np.unique(grid.rho)
    n_theta = grid.n_points // len(radii)
    assert len(radii) * n_theta == grid.n_points, "grid must be a tensor-product polar grid"
    return radii, n_theta


def upsample_profile(grid: CrossSectionGrid, u: np.ndarray, n_theta: int) -> np.ndarray:
    """Periodic linear interpolation of each ring onto n_theta angles, plus a centre value.

    OpenFOAM's planar triangulation of a coarse ring would cut corners near the
    wall and bias the mapped mean (setAverage would then distort the shape), so
    the boundary data are written on a dense polar set.
    """
    radii, nt = _rings(grid)
    order = np.lexsort((grid.theta, grid.rho))
    rings = np.asarray(u, float)[order].reshape(len(radii), nt)
    th_src = np.sort(grid.theta[order][:nt])
    th_dst = 2 * np.pi * np.arange(n_theta) / n_theta
    dense = np.stack([np.interp(th_dst, th_src, ring, period=2 * np.pi) for ring in rings])
    return np.concatenate([[rings[0].mean()], dense.ravel()])


def write_inlet_points(case: Path, grid: CrossSectionGrid, radius: float, n_theta: int = 64, n_wall: int = 128) -> None:
    radii, _ = _rings(grid)
    th = 2 * np.pi * np.arange(n_theta) / n_theta
    tw = 2 * np.pi * np.arange(n_wall) / n_wall
    rr = np.concatenate([[0.0], np.repeat(radii, n_theta), np.ones(n_wall)]) * radius
    tt = np.concatenate([[0.0], np.tile(th, len(radii)), tw])
    body = "\n".join(f"({a:.12g} {b:.12g} 0)" for a, b in zip(rr * np.cos(tt), rr * np.sin(tt)))
    _write(Path(case) / "constant" / "boundaryData" / "inlet" / "points", f"{len(rr)}\n(\n{body}\n)\n")


def write_inlet_profile(case: Path, t: float, u_axial: np.ndarray, q: float, inlet_area: float,
                        grid: CrossSectionGrid, n_theta: int = 64, n_wall: int = 128) -> None:
    """u_axial on the reference grid; wall ring gets zero. The average fixes the discrete flux to q."""
    vals = np.concatenate([upsample_profile(grid, u_axial, n_theta), np.zeros(n_wall)])
    body = "\n".join(f"(0 0 {v:.12g})" for v in vals)
    avg = q / inlet_area
    _write(
        Path(case) / "constant" / "boundaryData" / "inlet" / _fmt_time(t) / "U",
        # OpenFOAM v2412 reads the point values first, then the average used by setAverage.
        f"{len(vals)}\n(\n{body}\n)\n\n(0 0 {avg:.12g})\n",
    )


def write_outlet_pressure(case: Path, rows: Sequence[Tuple[float, float]]) -> None:
    body = "\n".join(f"({t:.12g} {p:.12g})" for t, p in rows)
    _write(Path(case) / "constant" / "outletPressure.dat", f"(\n{body}\n)\n")


def time_dirs(case: Path) -> List[Tuple[float, Path]]:
    out = []
    for d in Path(case).iterdir():
        if d.is_dir():
            try:
                out.append((float(d.name), d))
            except ValueError:
                pass
    return sorted(out)


def read_function_object(case: Path, name: str, start: Optional[float] = None) -> np.ndarray:
    """(time, value) rows from postProcessing/<name>/<start>/surfaceFieldValue.dat (latest start by default)."""
    base = Path(case) / "postProcessing" / name
    starts = sorted(((float(d.name), d) for d in base.iterdir()), key=lambda x: x[0])
    d = starts[-1][1] if start is None else min(starts, key=lambda x: abs(x[0] - start))[1]
    rows = []
    for line in (d / "surfaceFieldValue.dat").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        rows.append((float(parts[0]), float(parts[1])))
    return np.array(rows)


# ---------------------------------------------------------------------------
# Coupling adapter
# ---------------------------------------------------------------------------
class FoamLocal3D:
    """LocalModel backed by pimpleFoam. Construct, then call ``initialize``."""

    def __init__(
        self,
        foam: DockerFoam,
        case: Path,
        spec: FoamCaseSpec,
        grid: CrossSectionGrid,
        *,
        coupling_dt: float,
        substeps: int,
        distal_resistance: float,
        distal_compliance: float,
        q0: float,
        p_distal0: float,
        keep_every: int = 0,
        sample_z: Optional[float] = None,
        planes: Optional[dict] = None,
    ):
        self.foam, self.case, self.spec, self.grid = foam, Path(case).resolve(), spec, grid
        self.dt_c, self.substeps = coupling_dt, substeps
        self.rd, self.cd = distal_resistance, distal_compliance
        self.q, self.p_distal = q0, p_distal0
        self.t = 0.0
        self.keep_every = keep_every
        self._trial: Optional[dict] = None
        self._accepted = 0
        # Optional in-domain sampling (reference runs): profile on the reference grid at z = sample_z
        # and area-averaged p on named planes, read at the end of every trial.
        self.sample_z, self.planes = sample_z, dict(planes or {})
        self.samples: List[dict] = []
        self.solver_time = 0.0  # wall time of each docker exec (includes start-up and mesh read)
        self.cfd_time = 0.0  # OpenFOAM's own ExecutionTime summed over runs
        self.n_runs = 0

    def flux(self, u: np.ndarray) -> float:
        return float(self.grid.integrate(u, self.spec.geometry.radius))

    def initialize(self, u0: np.ndarray) -> str:
        write_case(self.case, self.spec, steady=False, delta_t=self.dt_c / self.substeps,
                   end_time=self.dt_c, write_every=self.substeps)
        if self.sample_z is not None or self.planes:
            extra = ""
            if self.sample_z is not None:
                extra += _probes_dict(grid_points_at(self.grid, self.spec.geometry.radius, self.sample_z), name="gridProbes")
            for name, z in self.planes.items():
                extra += _plane_average_dict(name, z)
            (self.case / "system" / "controlDict").write_text(
                _control_dict("pimpleFoam", self.dt_c / self.substeps, self.dt_c, self.substeps, False, extra_functions=extra))
        log = build_mesh(self.foam, self.case, self.spec)
        write_inlet_points(self.case, self.grid, self.spec.geometry.radius)
        write_inlet_profile(self.case, 0.0, u0, self.flux(u0), self.spec.inlet_area, self.grid)
        return log

    def _peek_distal(self, q: float) -> float:
        return (self.p_distal + self.dt_c / self.cd * q) / (1.0 + self.dt_c / (self.rd * self.cd))

    def trial(self, dt: float, t_next: float, u_inlet: np.ndarray) -> float:
        if abs(dt - self.dt_c) > 1e-12 * self.dt_c:
            raise ValueError("FoamLocal3D runs at a fixed coupling step")
        for t, d in time_dirs(self.case):
            if t > self.t + self._time_tol:
                shutil.rmtree(d)
        q = self.flux(u_inlet)
        p_distal = self._peek_distal(q)
        rho = self.spec.density
        write_inlet_profile(self.case, t_next, u_inlet, q, self.spec.inlet_area, self.grid)
        write_outlet_pressure(self.case, [(self.t, self.p_distal / rho), (t_next, p_distal / rho)])
        shutil.rmtree(self.case / "postProcessing", ignore_errors=True)
        ctrl = self.case / "system" / "controlDict"
        ctrl.write_text(re.sub(r"endTime\s+[^;]+;", f"endTime         {_fmt_time(t_next)};", ctrl.read_text(), count=1))
        tic = time.perf_counter()
        self.foam.run(self.case, "pimpleFoam", log="log.pimpleFoam")
        self.solver_time += time.perf_counter() - tic
        self.n_runs += 1
        exec_times = re.findall(r"ExecutionTime = ([0-9.eE+-]+) s", (self.case / "log.pimpleFoam").read_text())
        if exec_times:
            self.cfd_time += float(exec_times[-1])
        rows = read_function_object(self.case, "inletP")
        p_in = float(rows[np.argmin(np.abs(rows[:, 0] - t_next)), 1]) * rho
        flux_rows = read_function_object(self.case, "flux")
        q_mesh = -float(flux_rows[np.argmin(np.abs(flux_rows[:, 0] - t_next)), 1])
        sample = {}
        if self.sample_z is not None:
            gt, gu = read_probes(self.case, "U", name="gridProbes")
            sample["profile"] = gu[np.argmin(np.abs(gt - t_next)), :, 2]
        for name in self.planes:
            rows = read_function_object(self.case, name)
            sample[name] = float(rows[np.argmin(np.abs(rows[:, 0] - t_next)), 1]) * rho
        self._trial = dict(t=t_next, q=q, p_distal=p_distal, p_in=p_in, q_mesh=q_mesh, sample=sample)
        return p_in

    @property
    def _time_tol(self) -> float:
        # OpenFOAM names time directories with 12 significant digits and re-accumulates time from the
        # truncated name on restart, so written times drift in the last digit; compare on the step scale.
        return 1e-6 * self.dt_c

    def accept(self) -> None:
        tr = self._trial
        t_prev = self.t
        self.t, self.q, self.p_distal = tr["t"], tr["q"], tr["p_distal"]
        self._accepted += 1
        self.last_q_mesh = tr["q_mesh"]
        if tr["sample"]:
            self.samples.append({"t": tr["t"], "q": tr["q"], "p_in": tr["p_in"], **tr["sample"]})
        written = [t for t, _ in time_dirs(self.case) if abs(t - self.t) <= self._time_tol]
        if not written:
            raise RuntimeError(f"OpenFOAM did not write the accepted time {self.t!r}; refusing to prune {self.case}")
        for t, d in time_dirs(self.case):
            if t == 0.0 or abs(t - self.t) <= self._time_tol:
                continue
            if t < self.t and not (self.keep_every and round(t / self.dt_c) % self.keep_every == 0):
                shutil.rmtree(d)
        # Keep the previous sample as well: a restart time that drifted just below self.t still needs a bracketing sample.
        bd = self.case / "constant" / "boundaryData" / "inlet"
        for d in bd.iterdir():
            if d.is_dir():
                try:
                    if float(d.name) < t_prev - self._time_tol:
                        shutil.rmtree(d)
                except ValueError:
                    pass


# ---------------------------------------------------------------------------
# Steady runs (verification and mesh convergence)
# ---------------------------------------------------------------------------
def run_steady(
    foam_runner: DockerFoam,
    case: Path,
    spec: FoamCaseSpec,
    grid: CrossSectionGrid,
    u_inlet: np.ndarray,
    *,
    max_iterations: int = 5000,
    n_procs: int = 1,
) -> dict:
    """simpleFoam to convergence with a fixed inlet profile and p_out = 0 (kinematic)."""
    write_case(case, spec, steady=True, delta_t=1.0, end_time=max_iterations, write_every=max_iterations)
    build_mesh(foam_runner, case, spec)
    write_inlet_points(case, grid, spec.geometry.radius)
    q = float(grid.integrate(u_inlet, spec.geometry.radius))
    write_inlet_profile(case, 0.0, u_inlet, q, spec.inlet_area, grid)
    tic = time.perf_counter()
    if n_procs > 1:
        _write(case / "system" / "decomposeParDict",
               HEADER.format(cls="dictionary", obj="decomposeParDict")
               + f"\nnumberOfSubdomains {n_procs};\nmethod scotch;\n")
        foam_runner.run(case, "decomposePar -force", log="log.decomposePar")
        foam_runner.run(case, f"mpirun --allow-run-as-root --oversubscribe -np {n_procs} simpleFoam -parallel",
                        log="log.simpleFoam")
    else:
        foam_runner.run(case, "simpleFoam", log="log.simpleFoam")
    wall = time.perf_counter() - tic
    log = (case / "log.simpleFoam").read_text()
    iters = [int(m) for m in re.findall(r"\nTime = (\d+)", log)]
    rho = spec.density
    p_in = read_function_object(case, "inletP")[-1, 1] * rho
    p_out = read_function_object(case, "outletP")[-1, 1] * rho
    q_mesh = -read_function_object(case, "flux")[-1, 1]
    return {
        "cells": spec.n_cells, "q": q, "q_mesh": float(q_mesh), "p_in": float(p_in), "p_out": float(p_out),
        "dp": float(p_in - p_out), "iterations": iters[-1] if iters else None,
        "converged": "reached convergence" in log or "SIMPLE solution converged" in log,
        "wall_time_s": wall, "n_procs": n_procs,
    }


# ---------------------------------------------------------------------------
# Continuous prescribed-inflow runs (no coupling restarts; parallel, adaptive dt)
# ---------------------------------------------------------------------------
def _sampling_control(sample_interval: Optional[float]) -> str:
    """Execute and write together: by default OpenFOAM executes function objects every time step,
    and interpolated sampling of the whole field every step costs several solver iterations."""
    if sample_interval is None:
        return "writeControl    timeStep;\n        writeInterval   1;"
    return (f"executeControl  adjustableRunTime;\n        executeInterval {sample_interval:.12g};\n"
            f"        writeControl    adjustableRunTime;\n        writeInterval   {sample_interval:.12g};")


def _probes_dict(points: np.ndarray, sample_interval: Optional[float] = None, name: str = "probes") -> str:
    locs = "\n".join(f"            ({x:.10g} {y:.10g} {z:.10g})" for x, y, z in points)
    return f"""
    {name}
    {{
        type            probes;
        libs            (sampling);
        fields          (U p);
        interpolationScheme cellPoint;
        probeLocations
        (
{locs}
        );
        {_sampling_control(sample_interval)}
    }}
"""


def _plane_average_dict(name: str, z: float, sample_interval: Optional[float] = None) -> str:
    """Area-averaged p on the cross-sectional plane at axial position z."""
    return f"""
    {name}
    {{
        type            surfaceFieldValue;
        libs            (fieldFunctionObjects);
        regionType      sampledSurface;
        name            {name}Surface;
        sampledSurfaceDict
        {{
            type        plane;
            point       (0 0 {z:.12g});
            normal      (0 0 1);
            interpolate true;
        }}
        operation       areaAverage;
        fields          (p);
        writeFields     false;
        log             false;
        {_sampling_control(sample_interval)}
    }}
"""


def read_probes(case: Path, field: str = "U", name: str = "probes") -> Tuple[np.ndarray, np.ndarray]:
    """(times, values) from postProcessing/<name>/<latest start>/<field>; vector values have shape (T, n_probes, 3)."""
    base = Path(case) / "postProcessing" / name
    start = max(base.iterdir(), key=lambda d: float(d.name))
    times, rows = [], []
    for line in (start / field).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        t, rest = line.split(None, 1)
        times.append(float(t))
        if "(" in rest:
            rows.append([list(map(float, v.split())) for v in re.findall(r"\(([^()]+)\)", rest)])
        else:
            rows.append(list(map(float, rest.split())))
    return np.array(times), np.array(rows)


def grid_points_at(grid: CrossSectionGrid, radius: float, z: float) -> np.ndarray:
    return np.stack([grid.rho * radius * np.cos(grid.theta), grid.rho * radius * np.sin(grid.theta), np.full(grid.n_points, z)], -1)


def _extra_sampling(spec, grid, probe_points, grid_probe_z, planes, sample_interval) -> str:
    # Diagnostic probes: every step only when nothing else sets the cadence (transition checks).
    text = _probes_dict(probe_points, sample_interval)
    if grid_probe_z is not None:
        text += _probes_dict(grid_points_at(grid, spec.geometry.radius, grid_probe_z), sample_interval, name="gridProbes")
    for name, z in (planes or {}).items():
        text += _plane_average_dict(name, z, sample_interval)
    return text


def run_prescribed(
    foam_runner: DockerFoam,
    case: Path,
    spec: FoamCaseSpec,
    grid: CrossSectionGrid,
    profile,  # callable t -> axial velocity on the reference grid
    *,
    period: float,
    n_cycles: int,
    samples_per_cycle: int,
    distal_resistance: float,
    distal_compliance: float,
    p_distal0: float,
    probe_points: np.ndarray,
    max_co: float = 0.8,
    n_procs: int = 8,
    grid_probe_z: Optional[float] = None,
    planes: Optional[dict] = None,
    sample_interval: Optional[float] = None,
) -> dict:
    """One continuous pimpleFoam run with a prescribed inlet profile and an implicit RC Windkessel outlet.

    The inlet profiles are written at ``samples_per_cycle`` times per cycle and
    interpolated linearly in time by OpenFOAM. The outlet pressure follows from
    the prescribed flux, because Q_out = Q_in for a rigid incompressible domain.
    """
    end = period * n_cycles
    write_case(case, spec, steady=False, delta_t=1e-5, end_time=end, write_every=1)
    (Path(case) / "system" / "controlDict").write_text(
        _control_dict("pimpleFoam", 1e-5, end, 1, False, max_co=max_co, write_interval_time=period,
                      extra_functions=_extra_sampling(spec, grid, probe_points, grid_probe_z, planes, sample_interval))
    )
    build_mesh(foam_runner, case, spec)
    write_inlet_points(case, grid, spec.geometry.radius)
    times = period * np.arange(n_cycles * samples_per_cycle + 1) / samples_per_cycle
    rho, dt = spec.density, period / samples_per_cycle
    p_d, rows, flows = p_distal0, [], []
    for i, t in enumerate(times):
        u = profile(t)
        q = float(grid.integrate(u, spec.geometry.radius))
        write_inlet_profile(case, t, u, q, spec.inlet_area, grid)
        if i > 0:
            p_d = (p_d + dt / distal_compliance * q) / (1.0 + dt / (distal_resistance * distal_compliance))
        rows.append((t, p_d / rho))
        flows.append(q)
    write_outlet_pressure(case, rows)
    tic = time.perf_counter()
    if n_procs > 1:
        _write(Path(case) / "system" / "decomposeParDict",
               HEADER.format(cls="dictionary", obj="decomposeParDict") + f"\nnumberOfSubdomains {n_procs};\nmethod scotch;\n")
        foam_runner.run(case, "decomposePar -force", log="log.decomposePar")
        foam_runner.run(case, f"mpirun --allow-run-as-root --oversubscribe -np {n_procs} pimpleFoam -parallel",
                        log="log.pimpleFoam")
    else:
        foam_runner.run(case, "pimpleFoam", log="log.pimpleFoam")
    wall = time.perf_counter() - tic
    p_in = read_function_object(case, "inletP")
    p_out = read_function_object(case, "outletP")
    flux = read_function_object(case, "flux")
    tp, up = read_probes(case, "U")
    extra = {}
    if grid_probe_z is not None:
        extra["grid_t"], extra["grid_u"] = read_probes(case, "U", name="gridProbes")
    for pname in (planes or {}):
        extra[pname] = read_function_object(case, pname)
        extra[pname][:, 1] *= rho
    return {**extra, 
        "t": p_in[:, 0], "p_in": p_in[:, 1] * rho, "p_out": p_out[:, 1] * rho, "q_mesh": -flux[:, 1],
        "probe_t": tp, "probe_u": up, "sample_times": times, "sample_q": np.array(flows),
        "wall_time_s": wall, "cells": spec.n_cells, "n_procs": n_procs,
    }
