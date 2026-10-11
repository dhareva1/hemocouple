import pytest

from hemocouple.cross_section import CrossSectionGrid
from hemocouple.dataset import CaseRecord
from hemocouple.geometry import sample_cases
from hemocouple.mock import MockCase

STEPS_PER_CYCLE = 40
N_CYCLES = 4


@pytest.fixture(scope="session")
def grid():
    return CrossSectionGrid.gauss_polar(8, 4)


@pytest.fixture(scope="session")
def specs():
    return sample_cases(n_stenosis=8, n_control=2, seed=1)


@pytest.fixture(scope="session")
def records(specs, grid):
    out = []
    for spec in specs:
        case = MockCase(spec, grid)
        res = case.run(case.oracle(), "A", STEPS_PER_CYCLE, N_CYCLES, window=1, tol=1e-10, record_profiles=True)
        out.append(CaseRecord.from_result(spec, res, steps_per_cycle=STEPS_PER_CYCLE))
    return out
