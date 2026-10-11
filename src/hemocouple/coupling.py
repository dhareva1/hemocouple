"""Partitioned 1D-3D coupling loop with the operator in the loop (proposal 7.5).

Interface convention (Dirichlet-Neumann): the 1D side sends flow Q, the local
side returns interface pressure P. Each solver exposes ``trial`` (advance one
coupling step from the last accepted state without committing) and ``accept``
(commit the last trial), so a step can be iterated to convergence.

Modes:
    "A"  operator-assisted coupling: the local 3D solver runs; the provider
         supplies its inlet velocity profile.
    "B"  interface surrogate: no 3D solve; the interface pressure is the
         reduced (1D-transferred) estimate plus the predicted correction dP.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

try:
    from typing import Protocol
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol


class UpstreamModel(Protocol):
    def trial(self, dt: float, t_next: float, p_interface: float) -> Tuple[float, float]:
        """Return (Q at interface, 1D pressure near interface)."""

    def accept(self) -> None: ...


class LocalModel(Protocol):
    def trial(self, dt: float, t_next: float, u_inlet: np.ndarray) -> float:
        """Apply the inlet profile (on the reference grid) and return interface pressure."""

    def accept(self) -> None: ...


class ReducedInterface(Protocol):
    def trial(self, dt: float, t_next: float, q: float) -> float:
        """1D-transferred interface pressure for flow q (no stenosis-specific physics)."""

    def accept(self) -> None: ...


class InterfaceProvider(Protocol):
    def predict(self, history: np.ndarray, q: float, t: float) -> Tuple[np.ndarray, float]:
        """history: (window, 2) of (Q, P_1D), last row = current iterate.

        Returns (inlet velocity on the reference grid, pressure correction dP).
        """


@dataclass
class CouplingResult:
    t: np.ndarray
    q: np.ndarray  # flow on the 1D side
    p1d: np.ndarray  # 1D pressure near the interface
    p_interface: np.ndarray  # pressure returned to 1D
    p_lin: np.ndarray  # reduced 1D-transferred estimate (nan if no reduced model)
    q_local: np.ndarray  # flux of the applied/predicted profile
    dp_pred: np.ndarray
    iterations: np.ndarray
    converged: np.ndarray
    provider_time: float
    wall_time: float
    profiles: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)


def run_coupled(
    upstream: UpstreamModel,
    provider: InterfaceProvider,
    flux: Callable[[np.ndarray], float],
    *,
    mode: str,
    dt: float,
    n_steps: int,
    window: int,
    q_init: float,
    p1_init: float,
    p_init: float,
    local: Optional[LocalModel] = None,
    reduced: Optional[ReducedInterface] = None,
    t0: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 50,
    omega0: float = 0.5,
    record_profiles: bool = False,
) -> CouplingResult:
    """Fixed-point coupling with scalar Aitken relaxation on the interface pressure."""
    if mode not in ("A", "B"):
        raise ValueError(f"mode must be 'A' or 'B', got {mode!r}")
    if mode == "A" and local is None:
        raise ValueError("mode A needs a local (3D) model")
    if mode == "B" and reduced is None:
        raise ValueError("mode B needs a reduced interface model")

    hist = deque([(q_init, p1_init)] * window, maxlen=window)
    out = {k: np.full(n_steps, np.nan) for k in ("q", "p1d", "p", "p_lin", "q_local", "dp")}
    iterations = np.zeros(n_steps, dtype=int)
    converged = np.zeros(n_steps, dtype=bool)
    profiles = [] if record_profiles else None
    provider_time = 0.0
    p_prev = p_prev2 = p_init
    start = time.perf_counter()

    for n in range(n_steps):
        t = t0 + (n + 1) * dt
        p = 2.0 * p_prev - p_prev2 if n > 0 else p_prev
        omega, r_old = omega0, None
        for k in range(1, max_iter + 1):
            q, p1 = upstream.trial(dt, t, p)
            history = np.array(list(hist)[1:] + [(q, p1)])
            tic = time.perf_counter()
            u, dp = provider.predict(history, q, t)
            provider_time += time.perf_counter() - tic
            p_new = local.trial(dt, t, u) if mode == "A" else reduced.trial(dt, t, q) + dp
            r = p_new - p
            if abs(r) <= tol * max(abs(p_new), 1.0):
                p, converged[n] = p_new, True
                break
            if r_old is not None and r != r_old:
                omega = -omega * r_old / (r - r_old)
                if not np.isfinite(omega) or omega == 0.0:
                    omega = omega0
            r_old = r
            p = p + omega * r
        iterations[n] = k

        upstream.accept()
        if mode == "A":
            local.accept()
        if reduced is not None:
            out["p_lin"][n] = reduced.trial(dt, t, q)
            reduced.accept()
        hist.append((q, p1))
        out["q"][n], out["p1d"][n], out["p"][n] = q, p1, p
        out["q_local"][n], out["dp"][n] = flux(u), dp
        if record_profiles:
            profiles.append(np.array(u, copy=True))
        p_prev2, p_prev = p_prev, p

    return CouplingResult(
        t=t0 + dt * np.arange(1, n_steps + 1),
        q=out["q"],
        p1d=out["p1d"],
        p_interface=out["p"],
        p_lin=out["p_lin"],
        q_local=out["q_local"],
        dp_pred=out["dp"],
        iterations=iterations,
        converged=converged,
        provider_time=provider_time,
        wall_time=time.perf_counter() - start,
        profiles=np.array(profiles) if record_profiles else None,
        meta={"mode": mode, "dt": dt, "window": window},
    )
