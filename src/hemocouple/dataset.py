"""Case records, causal history windows, and case-level splits (proposal 7.3, 7.6).

A sample at coupling step n uses only steps n - window + 1 .. n of its own case.
Splits are made over cases, never over time steps of the same case.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .coupling import CouplingResult
from .geometry import CaseSpec


@dataclass
class CaseRecord:
    case_id: str
    spec: Dict
    features: np.ndarray
    radius: float
    dt: float
    t: np.ndarray
    q: np.ndarray
    p1d: np.ndarray
    p_interface: np.ndarray
    p_lin: np.ndarray
    profiles: np.ndarray  # (T, N) interface velocity on the reference grid
    meta: Dict = field(default_factory=dict)

    @property
    def dp(self) -> np.ndarray:
        """Pressure-correction label: P_3D,ref - P_1D,transferred."""
        return self.p_interface - self.p_lin

    @property
    def severity(self) -> float:
        return float(self.spec["geometry"]["severity"])

    @classmethod
    def from_result(cls, spec: CaseSpec, result: CouplingResult, **meta) -> "CaseRecord":
        if result.profiles is None:
            raise ValueError("reference run must be made with record_profiles=True")
        return cls(
            case_id=spec.case_id,
            spec=spec.to_dict(),
            features=spec.features(),
            radius=spec.geometry.radius,
            dt=result.meta["dt"],
            t=result.t,
            q=result.q,
            p1d=result.p1d,
            p_interface=result.p_interface,
            p_lin=result.p_lin,
            profiles=result.profiles,
            meta=meta,
        )

    def save(self, directory: Path) -> Path:
        path = Path(directory) / f"{self.case_id}.npz"
        np.savez_compressed(
            path,
            features=self.features,
            radius=self.radius,
            dt=self.dt,
            t=self.t,
            q=self.q,
            p1d=self.p1d,
            p_interface=self.p_interface,
            p_lin=self.p_lin,
            profiles=self.profiles,
            header=json.dumps({"case_id": self.case_id, "spec": self.spec, "meta": self.meta}),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "CaseRecord":
        with np.load(path) as z:
            header = json.loads(str(z["header"]))
            arrays = {k: z[k] for k in ("features", "t", "q", "p1d", "p_interface", "p_lin", "profiles")}
            return cls(
                case_id=header["case_id"],
                spec=header["spec"],
                radius=float(z["radius"]),
                dt=float(z["dt"]),
                meta=header["meta"],
                **arrays,
            )


def load_records(directory: Path, case_ids: Optional[Sequence[str]] = None) -> List[CaseRecord]:
    directory = Path(directory)
    if case_ids is None:
        paths = sorted(directory.glob("*.npz"))
    else:
        paths = [directory / f"{cid}.npz" for cid in case_ids]
    return [CaseRecord.load(p) for p in paths]


def build_samples(records: Sequence[CaseRecord], window: int, start: int = 0) -> Dict[str, np.ndarray]:
    """Stack per-step samples; ``start`` drops warm-up steps of every case."""
    parts: Dict[str, list] = {k: [] for k in ("hist", "g", "q", "radius", "u", "dp", "case_index")}
    for i, rec in enumerate(records):
        series = np.stack([rec.q, rec.p1d], axis=-1)
        idx = np.arange(max(window - 1, start), len(rec.t))
        if len(idx) == 0:
            continue
        parts["hist"].append(np.stack([series[n - window + 1 : n + 1] for n in idx]))
        parts["g"].append(np.repeat(rec.features[None], len(idx), axis=0))
        parts["q"].append(rec.q[idx])
        parts["radius"].append(np.full(len(idx), rec.radius))
        parts["u"].append(rec.profiles[idx])
        parts["dp"].append(rec.dp[idx])
        parts["case_index"].append(np.full(len(idx), i))
    return {k: np.concatenate(v) for k, v in parts.items()}


def split_cases(
    records: Sequence[CaseRecord],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    ood_severity: Optional[float] = None,
    seed: int = 0,
) -> Dict[str, List[str]]:
    """Case-level split. Cases with severity > ood_severity go to 'test_ood' only.

    Uniform controls (severity 0) and stenosis cases are shuffled separately so
    every split receives both kinds.
    """
    rng = np.random.default_rng(seed)
    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": [], "test_ood": []}
    pool = []
    for rec in records:
        if ood_severity is not None and rec.severity > ood_severity:
            splits["test_ood"].append(rec.case_id)
        else:
            pool.append(rec)
    for group in ([r for r in pool if r.severity == 0.0], [r for r in pool if r.severity > 0.0]):
        ids = [r.case_id for r in group]
        rng.shuffle(ids)
        n_test = int(round(test_frac * len(ids)))
        n_val = int(round(val_frac * len(ids)))
        splits["test"] += ids[:n_test]
        splits["val"] += ids[n_test : n_test + n_val]
        splits["train"] += ids[n_test + n_val :]
    return {k: sorted(v) for k, v in splits.items()}
