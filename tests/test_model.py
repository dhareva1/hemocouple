import math

import numpy as np
import pytest
import torch

from hemocouple.model import OperatorNet, flux_projection
from hemocouple.pod import PODBasis


def test_pod_modes_orthonormal_and_exact_in_span(grid):
    rng = np.random.default_rng(0)
    base = rng.normal(size=(3, grid.n_points))
    snaps = rng.normal(size=(50, 3)) @ base
    pod = PODBasis.fit(snaps, grid.weights, energy=1 - 1e-12)
    assert pod.n_modes == 3
    gram = pod.modes.T @ (pod.modes * grid.weights[:, None])
    np.testing.assert_allclose(gram, np.eye(3), atol=1e-10)
    assert pod.projection_error(snaps) < 1e-10
    radius = 2e-3
    c = pod.project(snaps)
    np.testing.assert_allclose(c @ pod.flux_vector(radius), grid.integrate(snaps, radius), rtol=1e-10)


def test_flux_projection_is_exact_and_idempotent():
    torch.manual_seed(0)
    c_hat = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    f = torch.randn(7, 5, dtype=torch.float64)
    q = torch.randn(7, dtype=torch.float64)
    c = flux_projection(c_hat, f, q)
    torch.testing.assert_close((f * c).sum(-1), q)
    torch.testing.assert_close(flux_projection(c, f, q), c)
    c.sum().backward()
    assert torch.isfinite(c_hat.grad).all()


@pytest.mark.parametrize("basis", ["pod", "trunk"])
def test_projected_model_conserves_flux(grid, basis):
    k, window, n_feat = 4, 5, 3
    model = OperatorNet(window, n_feat, grid.n_points, k, basis=basis, hidden=16, depth=2, project=True).double()
    rng = np.random.default_rng(1)
    modes = np.linalg.qr(rng.normal(size=(grid.n_points, k)))[0] / np.sqrt(grid.weights)[:, None]
    model.set_statistics(grid_rho=grid.rho, grid_theta=grid.theta, grid_weights=grid.weights, pod_modes=modes)
    b = 6
    radius = torch.full((b,), 3e-3, dtype=torch.float64)
    q = torch.linspace(-2e-6, 9e-6, b, dtype=torch.float64)
    out = model(torch.randn(b, window, 2, dtype=torch.float64), torch.randn(b, n_feat, dtype=torch.float64), q, radius)
    w = torch.as_tensor(grid.weights)
    flux = (out["u"] * w).sum(-1) * radius**2
    torch.testing.assert_close(flux, q, rtol=1e-10, atol=1e-18)


def test_checkpoint_roundtrip(grid):
    model = OperatorNet(4, 2, grid.n_points, 3, hidden=8, depth=1).double()
    model.set_statistics(grid_weights=grid.weights, dp_std=5.0)
    clone = OperatorNet.from_checkpoint(model.checkpoint())
    args = (torch.randn(2, 4, 2, dtype=torch.float64), torch.randn(2, 2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(clone(*args)["u"], model(*args)["u"])
    assert clone.dp_std.dtype == torch.float64


def test_young_tsai_prior_scales_quadratically(grid):
    model = OperatorNet(3, 4, grid.n_points, 2, hidden=8, depth=1, dp_prior="young_tsai").double()
    g = torch.tensor([[3e-3, 0.5, 4.0, 0.8], [3e-3, 0.75, 4.0, 0.8], [3e-3, 0.0, 4.0, 0.8]], dtype=torch.float64)
    q = torch.tensor([6e-6, 6e-6, 6e-6], dtype=torch.float64)
    r = torch.full((3,), 3e-3, dtype=torch.float64)
    yt = model.young_tsai(g, q, r)
    u = 6e-6 / (math.pi * 9e-6)
    assert yt[0].item() == pytest.approx(0.5 * 1060 * 1.52 * 1.0 * u * u)
    assert yt[1].item() == pytest.approx(0.5 * 1060 * 1.52 * 9.0 * u * u)
    assert yt[2].item() == 0.0
    assert model.young_tsai(g, -q, r)[0].item() == pytest.approx(-yt[0].item())
