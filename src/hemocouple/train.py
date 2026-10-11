"""Offline training of the interface operator (proposal 7.3, 7.4, 7.6).

All data-derived statistics (POD basis, K, normalization, loss scales) are
computed from the training cases only. Model selection uses the validation
*data* loss, the same criterion for every variant, applied per output head:
the profile head (branch, trunk) keeps its state at the best validation
velocity error, and the dP head (dp_branch, prior scale) keeps its own at the
best validation dP error. Training stops when both have stalled for
``patience`` epochs.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .cross_section import CrossSectionGrid
from .dataset import CaseRecord, build_samples
from .losses import LossScales, LossWeights, objective
from .model import OperatorNet
from .pod import PODBasis


@dataclass
class TrainConfig:
    window: int = 20
    warmup_cycles: int = 3
    basis: str = "pod"
    energy: float = 0.99999
    max_modes: int = 16
    project: bool = True
    hidden: int = 128
    depth: int = 3
    lam_data: float = 1.0
    lam_mass: float = 0.0
    lam_pressure: float = 0.1
    lam_pde: float = 0.0
    lr: float = 1e-3
    epochs: int = 300
    batch_size: int = 256
    patience: int = 30
    seed: int = 0
    density: float = 1060.0
    dp_prior: str = "none"

    @classmethod
    def from_json(cls, path: Path) -> "TrainConfig":
        return cls(**json.loads(Path(path).read_text()))

    def loss_weights(self) -> LossWeights:
        return LossWeights(self.lam_data, self.lam_mass, self.lam_pressure, self.lam_pde)


def _std(x: np.ndarray, axis, floor: float) -> np.ndarray:
    return np.maximum(np.std(x, axis=axis), floor)


def _to_torch(samples: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=torch.float64) for k, v in samples.items() if k != "case_index"}


def _start_index(records: Sequence[CaseRecord], cfg: TrainConfig) -> int:
    steps = {int(r.meta.get("steps_per_cycle", 0)) for r in records}
    if len(steps) != 1:
        raise ValueError(f"records must share steps_per_cycle, got {steps}")
    return cfg.warmup_cycles * steps.pop()


def fit_operator(
    train_records: Sequence[CaseRecord],
    val_records: Sequence[CaseRecord],
    grid: CrossSectionGrid,
    cfg: TrainConfig,
    verbose: bool = False,
) -> Tuple[OperatorNet, Dict]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    start = _start_index(list(train_records) + list(val_records), cfg)
    train = build_samples(train_records, cfg.window, start)
    val = build_samples(val_records, cfg.window, start)
    w = grid.weights

    pod = PODBasis.fit(train["u"], w, cfg.energy, cfg.max_modes)
    coef = pod.project(train["u"])
    velocity_scale = float(np.sqrt(np.mean(np.sum(train["u"] ** 2 * w, -1) / w.sum())))
    stats = dict(
        grid_rho=grid.rho, grid_theta=grid.theta, grid_weights=w,
        pod_modes=pod.modes,
        coef_scale=_std(coef, 0, 1e-8 * velocity_scale) if cfg.basis == "pod" else np.full(pod.n_modes, velocity_scale),
        hist_mean=train["hist"].mean(axis=(0, 1)), hist_std=_std(train["hist"], (0, 1), 1e-12),
        g_mean=train["g"].mean(0), g_std=_std(train["g"], 0, 1e-12),
        dp_mean=train["dp"].mean(), dp_std=_std(train["dp"], None, 1.0),
    )
    scales = LossScales(
        velocity=velocity_scale,
        flow=float(np.std(train["q"]) + np.abs(train["q"]).mean()),
        pressure=float(stats["dp_std"]),
        flow_eps=0.05 * float(np.abs(train["q"]).mean()),
    )

    model = OperatorNet(
        window=cfg.window, n_features=train["g"].shape[1], n_points=grid.n_points, n_modes=pod.n_modes,
        basis=cfg.basis, hidden=cfg.hidden, depth=cfg.depth, project=cfg.project,
        dp_prior=cfg.dp_prior, density=cfg.density,
    ).double()
    model.set_statistics(**stats)

    tr, va = _to_torch(train), _to_torch(val)
    wt = torch.as_tensor(w, dtype=torch.float64)
    lam = cfg.loss_weights()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(cfg.patience // 3, 1))

    n = len(tr["q"])
    history: Dict[str, List[float]] = {"train_total": [], "val_data": [], "val_velocity": [], "val_dp": []}
    dp_keys = {k for k in model.state_dict() if k.startswith("dp_branch.") or k == "prior_scale"}
    best = {"velocity": [np.inf, copy.deepcopy(model.state_dict()), 0], "dp": [np.inf, copy.deepcopy(model.state_dict()), 0]}
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            batch = {k: v[idx] for k, v in tr.items()}
            out = model(batch["hist"], batch["g"], batch["q"], batch["radius"])
            loss, _ = objective(out, batch, wt, lam, scales, cfg.density)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            out = model(va["hist"], va["g"], va["q"], va["radius"])
            _, parts = objective(out, va, wt, LossWeights(1.0, 0.0, 0.0, 0.0), scales, cfg.density)
        val_data = parts["data"]
        sched.step(val_data)
        history["train_total"].append(epoch_loss / n)
        history["val_data"].append(val_data)
        history["val_velocity"].append(parts["velocity"])
        history["val_dp"].append(parts["dp"])
        if verbose and (epoch % 10 == 0 or epoch == cfg.epochs - 1):
            print(f"epoch {epoch:4d}  train {epoch_loss / n:.4e}  val velocity {parts['velocity']:.4e}  val dP {parts['dp']:.4e}")
        for head in ("velocity", "dp"):
            if parts[head] < best[head][0] * (1 - 1e-4):
                best[head] = [parts[head], copy.deepcopy(model.state_dict()), 0]
            else:
                best[head][2] += 1
        if min(best["velocity"][2], best["dp"][2]) >= cfg.patience:
            break

    merged = {k: (best["dp"][1][k] if k in dp_keys else best["velocity"][1][k]) for k in model.state_dict()}
    model.load_state_dict(merged)
    best_val = best["velocity"][0] + best["dp"][0]
    model.eval()
    info = {
        "n_modes": pod.n_modes,
        "pod_energy": pod.captured_energy,
        "pod_projection_error_train": pod.projection_error(train["u"]),
        "pod_projection_error_val": pod.projection_error(val["u"]) if len(val["u"]) else float("nan"),
        "best_val_data": float(best_val),
        "best_val_velocity": float(best["velocity"][0]),
        "best_val_dp": float(best["dp"][0]),
        "epochs_run": len(history["val_data"]),
        "n_train_samples": int(n),
        "loss_scales": asdict(scales),
        "config": asdict(cfg),
    }
    return model, {"history": history, "info": info}


def save_checkpoint(path: Path, model: OperatorNet, report: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.checkpoint(extra={"info_json": json.dumps(report["info"])}), path)


def load_checkpoint(path: Path) -> Tuple[OperatorNet, Dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return OperatorNet.from_checkpoint(ckpt).double(), json.loads(ckpt["extra"]["info_json"])
