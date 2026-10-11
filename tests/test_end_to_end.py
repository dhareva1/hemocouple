import numpy as np

from conftest import STEPS_PER_CYCLE
from hemocouple.dataset import split_cases
from hemocouple.mock import MockCase
from hemocouple.providers import OperatorProvider
from hemocouple.train import TrainConfig, fit_operator, load_checkpoint, save_checkpoint


def _cfg(**kw):
    base = dict(window=6, warmup_cycles=2, max_modes=6, hidden=32, depth=2, epochs=40, batch_size=64, patience=40, seed=0)
    base.update(kw)
    return TrainConfig(**base)


def test_train_and_couple(records, specs, grid, tmp_path):
    splits = split_cases(records, val_frac=0.2, test_frac=0.2, seed=0)
    by_id = {r.case_id: r for r in records}
    train = [by_id[c] for c in splits["train"]]
    val = [by_id[c] for c in splits["val"]]

    phys, rep_phys = fit_operator(train, val, grid, _cfg(project=True, lam_pressure=0.1))
    data, _ = fit_operator(train, val, grid, _cfg(project=False, lam_pressure=0.0))
    hist = rep_phys["history"]["val_data"]
    assert min(hist) < hist[0]
    assert rep_phys["info"]["pod_projection_error_val"] < 0.05

    path = tmp_path / "phys.pt"
    save_checkpoint(path, phys, rep_phys)
    phys, info = load_checkpoint(path)
    assert info["n_modes"] == rep_phys["info"]["n_modes"]

    spec = next(s for s in specs if s.case_id == splits["test"][0])
    case = MockCase(spec, grid)
    for model, exact in ((phys, True), (data, False)):
        provider = OperatorProvider(model, spec.features(), spec.geometry.radius)
        for mode in ("A", "B"):
            res = case.run(provider, mode, STEPS_PER_CYCLE, 2, window=model.init_kwargs["window"], tol=1e-8)
            assert np.isfinite(res.p_interface).all()
            mass = np.abs(res.q_local - res.q).max() / np.abs(res.q).mean()
            if exact:
                assert mass < 1e-9
            else:
                assert mass > 1e-9
