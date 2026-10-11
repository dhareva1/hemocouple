"""Periodic pulsatile flow waveforms represented by truncated Fourier series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FourierWaveform:
    """q(t) = mean + Re sum_{n>=1} coeffs[n-1] * exp(i n omega t)."""

    period: float
    mean: float
    coeffs: np.ndarray

    @property
    def omega(self) -> float:
        return 2.0 * np.pi / self.period

    @property
    def n_harmonics(self) -> int:
        return len(self.coeffs)

    def phases(self, t) -> np.ndarray:
        """exp(i n omega t) with shape t.shape + (n_harmonics,)."""
        n = np.arange(1, self.n_harmonics + 1)
        return np.exp(1j * self.omega * np.multiply.outer(np.asarray(t, dtype=float), n))

    def __call__(self, t) -> np.ndarray:
        return self.mean + np.real(np.sum(self.phases(t) * self.coeffs, axis=-1))

    def derivative(self, t) -> np.ndarray:
        n = np.arange(1, self.n_harmonics + 1)
        return np.real(np.sum(self.phases(t) * (1j * self.omega * n * self.coeffs), axis=-1))


def pulsatile_flow(
    mean: float,
    amplitude: float,
    period: float,
    systole_peak: float = 0.15,
    n_harmonics: int = 8,
    n_samples: int = 512,
) -> FourierWaveform:
    """Arterial-like inflow: systolic peak, dicrotic notch, small diastolic wave.

    ``amplitude`` is the peak-to-peak flow of the truncated series and
    ``systole_peak`` the fraction of the cycle at which systole peaks.
    """
    s = np.arange(n_samples) / n_samples
    shape = (
        np.exp(-(((s - systole_peak) / 0.06) ** 2))
        - 0.25 * np.exp(-(((s - systole_peak - 0.22) / 0.04) ** 2))
        + 0.15 * np.exp(-(((s - systole_peak - 0.40) / 0.08) ** 2))
    )
    coeffs = 2.0 * np.fft.rfft(shape)[1 : n_harmonics + 1] / n_samples
    values = FourierWaveform(period, 0.0, coeffs)(s * period)
    scale = amplitude / (values.max() - values.min())
    return FourierWaveform(period, mean, coeffs * scale)
