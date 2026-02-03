# -*- coding: utf-8 -*-
"""
City-Networks (Paris/Shanghai/LA/London) Total Influence + R benchmark
for NSD vs PolyNSD (Diag/Bundle/General).

Paper-aligned defaults (City-Networks):
- max_hops=16
- num_samples=10000
- normalize=True
- average=True
- split: 10% train / 10% val / 80% test
Ref: LeonResearch/City-Networks README.  (PyG >= 2.7 provides total_influence)

This script supports:
- W&B sweep/agent mode: ONE run == ONE dataset+model+config
- Local multi-run mode

Key logged metrics:
- R_mean  (breadth of influence-weighted receptive field)
- R_curve/barT_norm over hop  (normalized per-hop influence)
- train curves (optional)

NOTE:
If your model forward returns log-probabilities (log-softmax), total_influence will
differentiate those outputs. For paper-comparability, logits are ideal, but many
models do not expose them. This script will log what it detected.
"""

import os
import sys
import time
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repo root on PYTHONPATH (wandb agent runs from repo root but keep safe)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.disc_models import (
    DiscreteDiagSheafDiffusion,
    DiscreteBundleSheafDiffusion,
    DiscreteGeneralSheafDiffusion,
    DiscreteDiagSheafDiffusionPolynomial,
    DiscreteBundleSheafDiffusionPolynomial,
    DiscreteGeneralSheafDiffusionPolynomial,
)
from utils.heterophilic import get_dataset

# --------------------------
# Optional: PyG CityNetwork + total_influence (preferred)
# --------------------------
HAS_PYG_CITY = False
HAS_TOTAL_INFLUENCE = False

try:
    from torch_geometric.datasets import CityNetwork as PyGCityNetwork  # type: ignore
    HAS_PYG_CITY = True
except Exception:
    HAS_PYG_CITY = False

try:
    from torch_geometric.utils import total_influence as pyg_total_influence  # type: ignore
    HAS_TOTAL_INFLUENCE = True
except Exception:
    HAS_TOTAL_INFLUENCE = False

# Fallback: local CityNetwork loader if you vendor it
# (e.g., you copied citynetworks.py from LeonResearch/City-Networks)
if not HAS_PYG_CITY:
    try:
        from citynetworks import CityNetwork as LocalCityNetwork  # type: ignore
        HAS_PYG_CITY = True
        PyGCityNetwork = LocalCityNetwork  # type: ignore
    except Exception:
        pass

MODEL_TABLE = {
    # NSD
    "DiagSheaf": DiscreteDiagSheafDiffusion,
    "BundleSheaf": DiscreteBundleSheafDiffusion,
    "GeneralSheaf": DiscreteGeneralSheafDiffusion,
    # PolyNSD
    "DiagSheafPolynomial": DiscreteDiagSheafDiffusionPolynomial,
    "BundleSheafPolynomial": DiscreteBundleSheafDiffusionPolynomial,
    "GeneralSheafPolynomial": DiscreteGeneralSheafDiffusionPolynomial,
}
ALL_MODELS = list(MODEL_TABLE.keys())

CITY_DATASETS = ["paris", "shanghai", "la", "london"]
DATASET_ALIASES = {"los_angeles": "la", "Los_Angeles": "la", "LA": "la"}


# --------------------------
# Reproducibility
# --------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def canonical_dataset_name(name: str) -> str:
    name = str(name).strip()
    return DATASET_ALIASES.get(name, name)


# --------------------------
# Split (paper-style 10/10/80)
# --------------------------
def apply_transductive_split(data, split_seed: int, train_ratio: float = 0.10, val_ratio: float = 0.10):
    n = int(data.num_nodes)
    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(n)

    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def _build_adj_list(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    edge = edge_index.detach().cpu().numpy()
    src = edge[0]
    dst = edge[1]
    adj = [[] for _ in range(int(num_nodes))]
    for u, v in zip(src, dst):
        u_i = int(u)
        v_i = int(v)
        adj[u_i].append(v_i)
        adj[v_i].append(u_i)
    return adj


def _bfs_dists(adj: List[List[int]], src: int, max_hops: int) -> np.ndarray:
    n = len(adj)
    dist = np.full(n, -1, dtype=np.int32)
    dist[int(src)] = 0
    q = [int(src)]
    qi = 0
    while qi < len(q):
        u = q[qi]
        qi += 1
        du = dist[u]
        if du >= max_hops:
            continue
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = du + 1
                q.append(v)
    return dist


def num_classes_from_y(y: torch.Tensor) -> int:
    return int(torch.unique(y).numel())


def looks_like_log_probs(z: torch.Tensor, tol: float = 1e-3) -> bool:
    with torch.no_grad():
        s = z.detach().float().exp().sum(dim=-1).mean().item()
    return abs(s - 1.0) < tol


# --------------------------
# Config
# --------------------------
@dataclass
class Config:
    # identity
    dataset: str = "paris"
    model: str = "DiagSheafPolynomial"
    base_seed: int = 0
    split_seed: int = 0
    device: str = "cuda:0"

    # training
    train: bool = True
    epochs: int = 300
    early_stopping: int = 50
    stop_strategy: str = "acc"  # acc | loss
    lr: float = 0.01
    weight_decay: float = 5e-4
    maps_lr: Optional[float] = None
    sheaf_decay: Optional[float] = None

    # model hparams
    d: int = 4
    layers: int = 16
    hidden_channels: int = 64
    dropout: float = 0.3
    linear: bool = False
    normalised: bool = True
    deg_normalised: bool = False
    use_epsilons: bool = True

    # PolyNSD hparams
    poly_layers_K: int = 16
    polynomial_type: str = "ChebyshevType1"
    lambda_max_choice: str = "analytic"

    # Total Influence probe (paper defaults)
    max_hops: int = 16
    num_samples: int = 10000
    normalize_influence: bool = True
    average_influence: bool = True
    vectorize: bool = True

    # Logging
    log_train_curves: bool = True
    log_probe_curve: bool = True


def short_run_name(cfg: Config) -> str:
    poly = "PolyNSD" if "Polynomial" in cfg.model else "NSD"
    K = cfg.poly_layers_K if "Polynomial" in cfg.model else 1
    return f"{cfg.dataset}-{cfg.model}-{poly}-L{cfg.layers}-K{K}-d{cfg.d}-split{cfg.split_seed}-seed{cfg.base_seed}"


# --------------------------
# Training
# --------------------------
def grouped_params_or_all(model: nn.Module):
    if hasattr(model, "grouped_parameters"):
        try:
            return model.grouped_parameters()
        except Exception:
            pass
    return [], list(model.parameters())


def fit_model(model: nn.Module, data, cfg: Config, wandb_module=None, log_curves: bool = False) -> Dict[str, Any]:
    epochs = int(cfg.epochs)
    early = int(cfg.early_stopping)
    stop_strategy = str(cfg.stop_strategy).lower()

    sheaf_params, other_params = grouped_params_or_all(model)

    lr = float(cfg.lr)
    wd = float(cfg.weight_decay)
    sheaf_wd = float(cfg.sheaf_decay if cfg.sheaf_decay is not None else wd)
    maps_lr = float(cfg.maps_lr if cfg.maps_lr is not None else lr)

    groups = []
    if len(sheaf_params) > 0:
        groups.append({"params": sheaf_params, "lr": maps_lr, "weight_decay": sheaf_wd})
    groups.append({"params": other_params, "lr": lr, "weight_decay": wd})
    opt = torch.optim.Adam(groups)

    # detect output type once (log_probs vs logits) for correct loss
    with torch.no_grad():
        z0 = model(data.x)
    output_kind = "log_probs" if looks_like_log_probs(z0) else "logits"

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    bad = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def compute_loss(z, mask):
        if output_kind == "log_probs":
            return F.nll_loss(z[mask], data.y[mask])
        return F.cross_entropy(z[mask], data.y[mask])

    def eval_acc_loss():
        z = model(data.x)
        pred = z.argmax(dim=1)
        tr_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
        va_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        te_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        tr_l = float(compute_loss(z, data.train_mask).item())
        va_l = float(compute_loss(z, data.val_mask).item())
        te_l = float(compute_loss(z, data.test_mask).item())
        return tr_acc, tr_l, va_acc, va_l, te_acc, te_l

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        opt.zero_grad(set_to_none=True)
        z = model(data.x)
        loss = compute_loss(z, data.train_mask)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            tr_acc, tr_l, va_acc, va_l, te_acc, te_l = eval_acc_loss()
        dt = time.perf_counter() - t0

        if wandb_module is not None and log_curves:
            wandb_module.log(
                {
                    "epoch": int(ep),
                    "curve/train_acc": float(tr_acc) * 100.0,
                    "curve/val_acc": float(va_acc) * 100.0,
                    "curve/test_acc": float(te_acc) * 100.0,
                    "curve/train_loss": float(tr_l),
                    "curve/val_loss": float(va_l),
                    "curve/time_epoch_s": float(dt),
                },
                step=int(ep),
            )

        improved = (va_acc > best_val_acc) if stop_strategy == "acc" else (va_l < best_val_loss)
        if improved:
            best_val_acc = float(va_acc)
            best_val_loss = float(va_l)
            best_epoch = int(ep)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= early:
                break

    model.load_state_dict(best_state)
    return {
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "train_output_kind_detected": output_kind,
    }


# --------------------------
# Total Influence / R (preferred: PyG)
# --------------------------
def compute_total_influence_and_R(
    model: nn.Module,
    data,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, float, str, Dict[str, Any]]:
    """
    Returns:
      hops: [0..H]
      avg_tot_inf: per-hop influence (maybe normalized if cfg.normalize_influence=True)
      R_mean: scalar R
      probe_output_kind_detected: logits or log_probs heuristic from model(data.x)
      extras: dict (e.g., retries used)
    """
    model.eval()
    H = int(cfg.max_hops)
    hops = np.arange(H + 1, dtype=np.int32)

    with torch.no_grad():
        z0 = model(data.x)
    output_kind = "log_probs" if looks_like_log_probs(z0) else "logits"

    extras: Dict[str, Any] = {"probe_retries": 0, "probe_vectorize_used": bool(cfg.vectorize), "probe_num_samples_used": int(cfg.num_samples)}

    if not HAS_TOTAL_INFLUENCE:
        # Fallback: compute influence via gradients per sampled target.
        num_nodes = int(data.num_nodes)
        num_samples = min(int(cfg.num_samples), num_nodes, 2000)
        extras["probe_num_samples_used"] = int(num_samples)
        extras["probe_vectorize_used"] = False
        extras["probe_fallback"] = True

        rng = np.random.default_rng(int(cfg.base_seed) + 1000 * int(cfg.split_seed))
        targets = (
            np.arange(num_nodes, dtype=np.int64)
            if num_samples >= num_nodes
            else rng.choice(num_nodes, size=num_samples, replace=False)
        )

        adj = _build_adj_list(data.edge_index, num_nodes)
        x = data.x.detach()
        x.requires_grad_(True)
        z = model(x)

        T_accum = np.zeros(H + 1, dtype=np.float64)
        R_vals: List[float] = []

        for idx, v in enumerate(targets.tolist()):
            g = z[int(v)].sum()
            grad_x = torch.autograd.grad(
                g, x, retain_graph=(idx < len(targets) - 1), create_graph=False
            )[0]
            infl = grad_x.abs().sum(dim=1).detach().cpu().numpy()

            dist = _bfs_dists(adj, int(v), H)
            T_h = np.zeros(H + 1, dtype=np.float64)
            for h in range(H + 1):
                mask = (dist == h)
                if mask.any():
                    T_h[h] = float(infl[mask].sum())

            denom = float(T_h.sum())
            if denom > 0:
                if bool(cfg.normalize_influence):
                    T_h = T_h / denom
                    Rv = float((hops * T_h).sum())
                else:
                    Rv = float((hops * T_h).sum() / denom)
                R_vals.append(Rv)
                T_accum += T_h

        if len(R_vals) == 0:
            avg = np.full(H + 1, np.nan, dtype=np.float64)
            return hops, avg, float("nan"), output_kind, extras

        if bool(cfg.average_influence):
            avg = T_accum / float(len(R_vals))
        else:
            avg = T_accum
        R_mean = float(np.mean(R_vals))
        return hops, avg, R_mean, output_kind, extras

    # OOM-safe retry policy
    num_samples_try = int(cfg.num_samples)
    vectorize_try = bool(cfg.vectorize)

    while True:
        try:
            avg_tot_inf, R = pyg_total_influence(
                model,
                data,
                max_hops=int(cfg.max_hops),
                num_samples=int(num_samples_try),
                normalize=bool(cfg.normalize_influence),
                average=bool(cfg.average_influence),
                device=str(cfg.device),
                vectorize=bool(vectorize_try),
            )
            # avg_tot_inf is a torch tensor [H+1] when average=True
            avg = avg_tot_inf.detach().cpu().numpy().astype(np.float64)
            R_mean = float(R.detach().cpu().item() if torch.is_tensor(R) else float(R))
            extras["probe_vectorize_used"] = bool(vectorize_try)
            extras["probe_num_samples_used"] = int(num_samples_try)
            return hops, avg, R_mean, output_kind, extras

        except torch.cuda.OutOfMemoryError:
            extras["probe_retries"] += 1
            torch.cuda.empty_cache()

            # First: disable vectorization
            if vectorize_try:
                vectorize_try = False
                continue

            # Then: reduce samples
            if num_samples_try > 2000:
                num_samples_try = max(2000, num_samples_try // 2)
                continue

            raise


# --------------------------
# One run
# --------------------------
def load_citynetwork(name: str, root: str):
    if HAS_PYG_CITY:
        ds = PyGCityNetwork(root=root, name=name)  # type: ignore
        return ds[0]
    dataset = get_dataset(name)
    return dataset[0]


def run_one(cfg: Config, wandb_module=None) -> Dict[str, Any]:
    dataset_name = canonical_dataset_name(cfg.dataset)
    if dataset_name not in CITY_DATASETS:
        raise ValueError(f"dataset must be one of {CITY_DATASETS}, got {dataset_name}")
    if cfg.model not in MODEL_TABLE:
        raise ValueError(f"model must be one of {ALL_MODELS}, got {cfg.model}")

    seed_eff = int(cfg.base_seed)
    set_seed(seed_eff)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Load data + apply paper-style split
    data_root = os.path.join(REPO_ROOT, "city_networks_data")
    data = load_citynetwork(dataset_name, root=data_root)
    data = apply_transductive_split(data, split_seed=int(cfg.split_seed), train_ratio=0.10, val_ratio=0.10)
    data = data.to(device)

    # Build model args
    args_dict = asdict(cfg)
    args_dict["dataset"] = dataset_name
    args_dict["graph_size"] = int(data.num_nodes)
    args_dict["input_dim"] = int(data.x.size(-1))
    args_dict["output_dim"] = int(num_classes_from_y(data.y))

    if not (bool(args_dict["normalised"]) or bool(args_dict["deg_normalised"])):
        args_dict["normalised"] = True
    if args_dict.get("sheaf_decay", None) is None:
        args_dict["sheaf_decay"] = float(args_dict["weight_decay"])

    # Model
    model_cls = MODEL_TABLE[cfg.model]
    model = model_cls(data.edge_index, args_dict).to(device)

    # Train
    train_summary = {}
    if cfg.train:
        train_summary = fit_model(
            model, data, cfg,
            wandb_module=wandb_module,
            log_curves=(wandb_module is not None and bool(cfg.log_train_curves)),
        )

    # Probe
    t0 = time.perf_counter()
    hops, avg_tot_inf, R_mean, output_kind, probe_extras = compute_total_influence_and_R(model, data, cfg)
    probe_time_s = float(time.perf_counter() - t0)

    # If normalize=False, also create a normalized view for plotting
    if bool(cfg.normalize_influence):
        barT = avg_tot_inf
        barT_norm = avg_tot_inf  # already normalized in this mode
    else:
        barT = avg_tot_inf
        barT_norm = avg_tot_inf / float(avg_tot_inf[0] + 1e-12)

    # Log per-hop curve
    if wandb_module is not None and bool(cfg.log_probe_curve):
        try:
            wandb_module.define_metric("R_curve/hop")
            wandb_module.define_metric("R_curve/*", step_metric="R_curve/hop")
        except Exception:
            pass
        for h, rawv, nv in zip(hops.tolist(), barT.tolist(), barT_norm.tolist()):
            wandb_module.log(
                {
                    "R_curve/hop": int(h),
                    "R_curve/barT": float(rawv),
                    "R_curve/barT_norm": float(nv),
                }
            )

    summary = {
        "dataset": dataset_name,
        "model": cfg.model,
        "base_seed": int(cfg.base_seed),
        "split_seed": int(cfg.split_seed),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_features": int(data.x.size(-1)),
        "num_classes": int(num_classes_from_y(data.y)),
        "R_mean": float(R_mean),
        "max_hops": int(cfg.max_hops),
        "num_samples": int(cfg.num_samples),
        "normalize_influence": bool(cfg.normalize_influence),
        "average_influence": bool(cfg.average_influence),
        "vectorize": bool(cfg.vectorize),
        "probe_time_s": float(probe_time_s),
        "probe_output_kind_detected": str(output_kind),
        **{f"train/{k}": v for k, v in train_summary.items()},
        **{f"probe/{k}": v for k, v in probe_extras.items()},
    }

    if wandb_module is not None:
        wandb_module.log(
            {
                "dataset_name": dataset_name,
                "model_name": cfg.model,
                "R_mean": float(R_mean),
                "probe_time_s": float(probe_time_s),
                "probe_output_kind_detected": str(output_kind),
                "probe/num_samples_used": int(probe_extras["probe_num_samples_used"]),
                "probe/vectorize_used": bool(probe_extras["probe_vectorize_used"]),
                "probe/retries": int(probe_extras["probe_retries"]),
            }
        )
        try:
            wandb_module.run.summary.update(summary)
        except Exception:
            pass

    return summary


# --------------------------
# CLI / W&B
# --------------------------
def cfg_from_sources(wb: Dict[str, Any], args) -> Config:
    dataset = canonical_dataset_name(str(wb.get("dataset", args.dataset)))
    model = str(wb.get("model", args.model))

    return Config(
        dataset=dataset,
        model=model,
        base_seed=int(wb.get("base_seed", args.base_seed)),
        split_seed=int(wb.get("split_seed", args.split_seed)),
        device=str(wb.get("device", args.device)),

        train=bool(wb.get("train", bool(args.train))),
        epochs=int(wb.get("epochs", args.epochs)),
        early_stopping=int(wb.get("early_stopping", args.early_stopping)),
        stop_strategy=str(wb.get("stop_strategy", args.stop_strategy)),
        lr=float(wb.get("lr", args.lr)),
        weight_decay=float(wb.get("weight_decay", args.weight_decay)),
        maps_lr=(None if wb.get("maps_lr", args.maps_lr) is None else float(wb.get("maps_lr", args.maps_lr))),
        sheaf_decay=(None if wb.get("sheaf_decay", args.sheaf_decay) is None else float(wb.get("sheaf_decay", args.sheaf_decay))),

        d=int(wb.get("d", args.d)),
        layers=int(wb.get("layers", args.layers)),
        hidden_channels=int(wb.get("hidden_channels", args.hidden_channels)),
        dropout=float(wb.get("dropout", args.dropout)),
        linear=bool(wb.get("linear", bool(args.linear))),
        normalised=bool(wb.get("normalised", bool(args.normalised))),
        deg_normalised=bool(wb.get("deg_normalised", bool(args.deg_normalised))),
        use_epsilons=bool(wb.get("use_epsilons", bool(args.use_epsilons))),

        poly_layers_K=int(wb.get("poly_layers_K", args.poly_layers_K)),
        polynomial_type=str(wb.get("polynomial_type", args.polynomial_type)),
        lambda_max_choice=str(wb.get("lambda_max_choice", args.lambda_max_choice)),

        max_hops=int(wb.get("max_hops", args.max_hops)),
        num_samples=int(wb.get("num_samples", args.num_samples)),
        normalize_influence=bool(wb.get("normalize_influence", bool(args.normalize_influence))),
        average_influence=bool(wb.get("average_influence", bool(args.average_influence))),
        vectorize=bool(wb.get("vectorize", bool(args.vectorize))),

        log_train_curves=bool(wb.get("log_train_curves", bool(args.log_train_curves))),
        log_probe_curve=bool(wb.get("log_probe_curve", bool(args.log_probe_curve))),
    )


def main():
    p = argparse.ArgumentParser()

    # W&B
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_agent", action="store_true")
    p.add_argument("--entity", type=str, default=None)
    p.add_argument("--project", type=str, default="CityNetwork_Influence_RF")

    # agent inputs (single-run)
    p.add_argument("--dataset", type=str, default="paris")
    p.add_argument("--model", type=str, default="DiagSheafPolynomial", choices=ALL_MODELS)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--split_seed", type=int, default=0)

    # local multi-run
    p.add_argument("--datasets", nargs="*", default=CITY_DATASETS)
    p.add_argument("--models", nargs="*", default=ALL_MODELS)
    p.add_argument("--base_seeds", type=int, nargs="*", default=[0])
    p.add_argument("--split_seeds", type=int, nargs="*", default=[0])

    # device
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    # train
    p.add_argument("--train", type=int, default=1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--early_stopping", type=int, default=50)
    p.add_argument("--stop_strategy", type=str, default="acc", choices=["acc", "loss"])
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--maps_lr", type=float, default=None)
    p.add_argument("--sheaf_decay", type=float, default=None)

    # model
    p.add_argument("--d", type=int, default=4)
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--hidden_channels", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--linear", type=int, default=0)
    p.add_argument("--normalised", type=int, default=1)
    p.add_argument("--deg_normalised", type=int, default=0)
    p.add_argument("--use_epsilons", type=int, default=1)

    # PolyNSD params
    p.add_argument("--poly_layers_K", type=int, default=16)
    p.add_argument("--polynomial_type", type=str, default="ChebyshevType1")
    p.add_argument("--lambda_max_choice", type=str, default="analytic")

    # probe (paper defaults)
    p.add_argument("--max_hops", type=int, default=16)
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--normalize_influence", type=int, default=1)
    p.add_argument("--average_influence", type=int, default=1)
    p.add_argument("--vectorize", type=int, default=1)

    # logging
    p.add_argument("--log_train_curves", type=int, default=1)
    p.add_argument("--log_probe_curve", type=int, default=1)

    args = p.parse_args()
    args.dataset = canonical_dataset_name(args.dataset)
    args.datasets = [canonical_dataset_name(d) for d in (args.datasets or [])]

    wandb_mod = None
    if args.wandb or args.wandb_agent:
        import wandb  # type: ignore
        wandb_mod = wandb

    if args.wandb_agent:
        run = wandb_mod.init(entity=args.entity)  # sweep controls project
        wb = dict(wandb_mod.config)
        cfg = cfg_from_sources(wb, args)

        try:
            wandb_mod.run.name = short_run_name(cfg)
        except Exception:
            pass

        summary = run_one(cfg, wandb_module=wandb_mod)
        print(f"\n[{cfg.dataset} | {cfg.model}] R_mean={summary['R_mean']:.4f}")
        wandb_mod.finish()
        return

    # local multi-run mode
    datasets = [d for d in (args.datasets or []) if d in CITY_DATASETS]
    models = [m for m in (args.models or []) if m in ALL_MODELS]

    for dname in datasets:
        for mname in models:
            for bseed in args.base_seeds:
                for sseed in args.split_seeds:
                    cfg = Config(
                        dataset=dname,
                        model=mname,
                        base_seed=int(bseed),
                        split_seed=int(sseed),
                        device=str(args.device),

                        train=bool(int(args.train)),
                        epochs=int(args.epochs),
                        early_stopping=int(args.early_stopping),
                        stop_strategy=str(args.stop_strategy),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                        maps_lr=args.maps_lr,
                        sheaf_decay=args.sheaf_decay,

                        d=int(args.d),
                        layers=int(args.layers),
                        hidden_channels=int(args.hidden_channels),
                        dropout=float(args.dropout),
                        normalised=bool(int(args.normalised)),
                        deg_normalised=bool(int(args.deg_normalised)),
                        use_epsilons=bool(int(args.use_epsilons)),

                        poly_layers_K=int(args.poly_layers_K),
                        polynomial_type=str(args.polynomial_type),
                        lambda_max_choice=str(args.lambda_max_choice),

                        max_hops=int(args.max_hops),
                        num_samples=int(args.num_samples),
                        normalize_influence=bool(int(args.normalize_influence)),
                        average_influence=bool(int(args.average_influence)),
                        vectorize=bool(int(args.vectorize)),

                        log_train_curves=bool(int(args.log_train_curves)),
                        log_probe_curve=bool(int(args.log_probe_curve)),
                    )

                    if args.wandb:
                        run = wandb_mod.init(
                            entity=args.entity,
                            project=args.project,
                            config=asdict(cfg),
                            name=short_run_name(cfg),
                        )

                    summary = run_one(cfg, wandb_module=wandb_mod if args.wandb else None)
                    print(f"[{dname} | {mname} | split_seed={sseed} | seed={bseed}] R_mean={summary['R_mean']:.4f}")

                    if args.wandb:
                        wandb_mod.finish()


if __name__ == "__main__":
    main()
