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


def test_stokes_prior_is_causal_womersley(grid):
    from hemocouple.stokes import NU_BLOOD, stokes_basis, stokes_prior
    from hemocouple.waveforms import FourierWaveform
    from hemocouple.womersley import WomersleyProfile

    wave = FourierWaveform(period=0.8, mean=5e-6, coeffs=np.array([2e-6 - 1e-6j, 1e-6j, 5e-7]))
    radius, steps = 3e-3, 100
    t = np.arange(3 * steps + 1) * wave.period / steps
    lam, phi, b = stokes_basis(grid.rho)
    T = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64))  # noqa: E731
    u = stokes_prior(T(wave(t))[None], T([radius]), T([wave.period / steps]), T(grid.rho), T(lam), T(phi), T(b))[0].numpy()
    ref = WomersleyProfile(wave, radius, NU_BLOOD, grid.rho)(t[-1])
    assert np.sqrt(np.sum((u - ref) ** 2 * grid.weights) / np.sum(ref**2 * grid.weights)) < 2e-3
    assert abs(grid.integrate(u, radius) - wave(t[-1])) / wave.mean < 1e-3
    # Only past Q enters: a constant history gives exactly the Poiseuille profile.
    flat = stokes_prior(T(np.full((1, 50), 4e-6)), T([radius]), T([8e-3]), T(grid.rho), T(lam), T(phi), T(b))[0].numpy()
    np.testing.assert_allclose(flat, 4e-6 * 2 * (1 - grid.rho**2) / (math.pi * radius**2), rtol=1e-12)


def test_stokes_prior_model_keeps_exact_flux_and_roundtrips(grid):
    torch.manual_seed(0)
    model = OperatorNet(window=4, n_features=4, n_points=grid.n_points, n_modes=3, velocity_prior="stokes",
                        prior_window=30, steps_per_cycle=100).double()
    model.set_statistics(grid_rho=grid.rho, grid_theta=grid.theta, grid_weights=grid.weights,
                         pod_modes=np.linalg.qr(np.random.default_rng(0).normal(size=(grid.n_points, 3)))[0])
    model.init_prior()
    assert model.history_length == 30
    hist = torch.rand(5, 30, 2, dtype=torch.float64) * 1e-5
    g = torch.tensor([[3e-3, 0.5, 3.0, 0.8]] * 5, dtype=torch.float64)
    radius = torch.full((5,), 3e-3, dtype=torch.float64)
    out = model(hist, g, hist[:, -1, 0], radius)
    flux = radius**2 * (out["u"] @ torch.as_tensor(grid.weights))
    torch.testing.assert_close(flux, hist[:, -1, 0], rtol=1e-10, atol=0)
    again = OperatorNet.from_checkpoint(model.checkpoint()).double()
    torch.testing.assert_close(again(hist, g, hist[:, -1, 0], radius)["u"], out["u"])
