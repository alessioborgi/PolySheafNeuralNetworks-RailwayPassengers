#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_oversquashing_influence.py

Stand-alone oversquashing experiment (Point 1: gradient-based influence vs hop distance)
for:
  - 3 sheaf model families: diag / bundle / general
  - 2 variants each: NSD vs PolyNSD
  - 3 datasets: Squirrel / Chameleon / PubMed

What it does:
  1) Train each model (or load from ckpt if present)
  2) Compute I(d) = average || d score(v) / d x_u || over u at hop-distance d from v
  3) Save curves to CSV + PNG
  4) Produce a combined plot per dataset (6 curves: 3 families x 2 variants)

Intended to run inside your repo (uses your model classes if available).
Fallback to PyG dataset loaders only if your repo loaders are unavailable.

Run example:
  python exp_oversquashing_influence.py --out_dir oversquash_out --device cuda:0 --epochs 300 --early_stopping 50 \
    --max_hops 10 --num_targets 64 --fold 0 --train_if_missing 1

Notes:
  - Influence is computed in eval() mode (dropout off).
  - Score for target node v is log-prob of the true class: model(x)[v, y[v]] (your models output log_softmax).
  - For PubMed you may want fewer targets or smaller max_hops for speed.

Author: (you)
"""

import os
import sys
import csv
import math
import time
import copy
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------- repo path -------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# ------------------------------- optional repo imports -------------------------------
HAVE_REPO = True
try:
    from utils.heterophilic import get_dataset, get_fixed_splits
except Exception:
    HAVE_REPO = False
    get_dataset = None
    get_fixed_splits = None

try:
    from models.disc_models import (
        DiscreteDiagSheafDiffusion,
        DiscreteBundleSheafDiffusion,
        DiscreteGeneralSheafDiffusion,
        DiscreteDiagSheafDiffusionPolynomial,
        DiscreteBundleSheafDiffusionPolynomial,
        DiscreteGeneralSheafDiffusionPolynomial,
    )
except Exception as e:
    HAVE_REPO = False
    raise RuntimeError(
        "Could not import your repo model classes. "
        "Place this file inside your repo (e.g., exp/) and run from there."
    ) from e

# ------------------------------- optional PyG fallback loaders -------------------------------
PYG_OK = True
try:
    from torch_geometric.datasets import Planetoid, WikipediaNetwork
except Exception:
    PYG_OK = False


# =============================================================================
# Config
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def as_device(dev: str) -> torch.device:
    return torch.device(dev)


def make_default_cfg() -> Dict:
    """
    Minimal dict-like config that matches keys your models expect.
    Tweak defaults here if needed.
    """
    return dict(
        # --- core ---
        d=4,
        layers=2,
        hidden_channels=32,
        input_dim=None,      # will be filled after loading dataset
        output_dim=None,     # will be filled after loading dataset
        graph_size=None,     # will be filled after loading dataset

        # --- model behavior ---
        use_embedding=True,
        use_act=True,
        second_linear=False,
        nonlinear=True,
        linear=False,          # explicit flag expected by base classes
        sparse_learner=False,
        rotation_learner=False,  # (kept for completeness)
        time_dep=False,

        # --- dropouts ---
        input_dropout=0.0,
        dropout=0.3,
        sheaf_dropout=0.0,

        # --- normalization ---
        normalised=True,
        deg_normalised=False,

        # --- spectral toggles (your builders use these) ---
        add_hp=1,
        add_lp=0,

        # --- linear sandwiches ---
        left_weights=True,
        right_weights=True,

        # --- edge-weights for bundle/general (optional) ---
        edge_weights=True,      # in your wandb config it's often called 'edge_weights'
        use_edge_weights=True,  # some classes use this name

        # --- orth parameterization for bundle ---
        orth="householder",
        orth_trans="householder",

        # --- eps residual gates ---
        use_epsilons=True,

        # --- optimization ---
        lr=0.01,
        maps_lr=None,           # if not None, used for sheaf learner params
        weight_decay=5e-4,
        sheaf_decay=None,       # if None, set to weight_decay

        # --- poly config ---
        polynomial_type="ChebyshevType1",
        poly_layers_K=10,              # target receptive "radius"
        gegenbauer_lambda=1.0,
        jacobi_alpha=0.0,
        jacobi_beta=0.0,
        lambda_max_choice="analytic",

        # --- misc (some base classes look for these) ---
        sheaf_act="tanh",
        task="node_classification",
        cuda=0,                 # unused here; kept for compatibility
        device="cpu",
    )


# =============================================================================
# Dataset loading
# =============================================================================

def _ensure_masks(data, seed=0, train_ratio=0.6, val_ratio=0.2):
    """
    Fallback split if dataset doesn't provide masks.
    """
    N = int(data.num_nodes)
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(train_ratio * N)
    n_val = int(val_ratio * N)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def load_dataset(name: str, fold: int, seed: int, root: str = ".data"):
    """
    Prefer your repo loaders; fallback to PyG.
    Returns a single PyG Data object.
    """
    lname = name.lower()

    if HAVE_REPO and get_dataset is not None:
        ds = get_dataset(lname)
        data = ds[0]
        if get_fixed_splits is not None and hasattr(data, "train_mask"):
            # your repo fixed splits (10 folds typically)
            data = get_fixed_splits(data, lname, fold)
        else:
            data = _ensure_masks(data, seed=seed + fold)
        return data

    if not PYG_OK:
        raise RuntimeError("No repo dataset loader and PyG datasets unavailable. Install torch_geometric.")

    ensure_dir(root)

    if lname in ("pubmed", "pubmeds"):
        ds = Planetoid(root=os.path.join(root, "Planetoid"), name="PubMed")
        data = ds[0]
        return data

    if lname in ("squirrel", "chameleon"):
        # WikipediaNetwork provides Geom-GCN preprocessing; masks may differ by version.
        ds = WikipediaNetwork(root=os.path.join(root, "WikipediaNetwork"),
                              name=lname, geom_gcn_preprocess=True)
        data = ds[0]
        if not hasattr(data, "train_mask"):
            data = _ensure_masks(data, seed=seed + fold)
        return data

    raise ValueError(f"Unknown dataset {name}")


# =============================================================================
# Models (diag/bundle/general) x (nsd/poly)
# =============================================================================

MODEL_TABLE = {
    ("diag", "nsd"): DiscreteDiagSheafDiffusion,
    ("bundle", "nsd"): DiscreteBundleSheafDiffusion,
    ("general", "nsd"): DiscreteGeneralSheafDiffusion,
    ("diag", "poly"): DiscreteDiagSheafDiffusionPolynomial,
    ("bundle", "poly"): DiscreteBundleSheafDiffusionPolynomial,
    ("general", "poly"): DiscreteGeneralSheafDiffusionPolynomial,
}


def build_model(family: str, variant: str, edge_index: torch.Tensor, cfg: Dict) -> torch.nn.Module:
    cls = MODEL_TABLE[(family, variant)]
    # Your model constructors expect (edge_index, args_dict)
    model = cls(edge_index, cfg)
    return model


# =============================================================================
# Training / evaluation (matches your optimizer grouping style)
# =============================================================================

def grouped_params_or_all(model: torch.nn.Module):
    if hasattr(model, "grouped_parameters"):
        try:
            return model.grouped_parameters()
        except Exception:
            pass
    return [], list(model.parameters())


def train_one(model, optimizer, data):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(data.x)[data.train_mask]
    loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def eval_acc_loss(model, data):
    model.eval()
    logits = model(data.x)
    accs, losses = [], []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        pred = logits[mask].argmax(dim=1)
        acc = (pred == data.y[mask]).float().mean().item()
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        accs.append(acc)
        losses.append(loss)
    return accs, losses


def fit_model(model, data, cfg: Dict, epochs: int, early_stopping: int, stop_strategy: str):
    sheaf_params, other_params = grouped_params_or_all(model)

    wd = float(cfg.get("weight_decay", 0.0))
    sheaf_wd = float(cfg.get("sheaf_decay", wd if cfg.get("sheaf_decay", None) is None else cfg["sheaf_decay"]))
    lr = float(cfg.get("lr", 0.01))
    maps_lr = cfg.get("maps_lr", None)
    maps_lr = float(maps_lr) if maps_lr is not None else lr

    param_groups = []
    if len(sheaf_params) > 0:
        param_groups.append({"params": sheaf_params, "weight_decay": sheaf_wd, "lr": maps_lr})
    param_groups.append({"params": other_params, "weight_decay": wd, "lr": lr})

    optimizer = torch.optim.Adam(param_groups)

    best = {"val": -1.0, "val_loss": float("inf"), "epoch": 0}
    best_state = copy.deepcopy(model.state_dict())
    bad = 0

    for ep in range(int(epochs)):
        _ = train_one(model, optimizer, data)
        (tr, va, te), (trl, val, tel) = eval_acc_loss(model, data)

        improved = (va > best["val"]) if stop_strategy == "acc" else (val < best["val_loss"])
        if improved:
            best["val"] = float(va)
            best["val_loss"] = float(val)
            best["epoch"] = int(ep)
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1

        if bad >= int(early_stopping):
            break

    model.load_state_dict(best_state)
    (tr, va, te), (trl, val, tel) = eval_acc_loss(model, data)
    return dict(best_epoch=best["epoch"], train_acc=tr, val_acc=va, test_acc=te, train_loss=trl, val_loss=val, test_loss=tel)


# =============================================================================
# Oversquashing diagnostic: Influence vs hop distance
# =============================================================================

def build_undirected_adj(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    ei = edge_index.detach().cpu().numpy()
    src = ei[0].tolist()
    dst = ei[1].tolist()
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)
    return adj


def bfs_distances(adj: List[List[int]], src: int, max_hops: int) -> np.ndarray:
    """
    Returns dist array with -1 for >max_hops unreachable.
    """
    n = len(adj)
    dist = -np.ones(n, dtype=np.int32)
    dist[src] = 0
    q = [src]
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


@torch.no_grad()
def sample_targets(data, split: str, num_targets: int, seed: int) -> torch.Tensor:
    if split == "train":
        mask = data.train_mask
    elif split == "val":
        mask = data.val_mask
    elif split == "test":
        mask = data.test_mask
    else:
        mask = torch.ones(data.num_nodes, dtype=torch.bool, device=data.x.device)

    idx = torch.where(mask)[0].detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    idx = idx[: min(num_targets, len(idx))]
    return torch.tensor(idx, dtype=torch.long)


def influence_by_distance(
    model: torch.nn.Module,
    data,
    max_hops: int,
    num_targets: int,
    target_split: str,
    seed: int,
    device: torch.device,
    normalize_by_d0: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes I(d) and (optionally) normalized curve I(d)/I(0).

    Returns:
      dists: (D+1,)
      mean_curve: (D+1,)
      norm_curve: (D+1,)
    """
    model.eval()

    N = int(data.num_nodes)
    adj = build_undirected_adj(data.edge_index, N)

    targets = sample_targets(data, target_split, num_targets, seed=seed).tolist()

    # collect per-target I(d) so we can average robustly
    per_target = [[] for _ in range(max_hops + 1)]

    for t in targets:
        # fresh input requiring grads
        x = data.x.detach().clone().requires_grad_(True)

        # forward
        logits = model(x)  # (N, C) log-probs
        y_t = int(data.y[t].item())
        score = logits[t, y_t]

        # backward
        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        score.backward()

        grad = x.grad.detach()  # (N, F)
        gnorm = torch.norm(grad, dim=1).detach().cpu().numpy()  # (N,)

        dist = bfs_distances(adj, src=t, max_hops=max_hops)

        for d in range(max_hops + 1):
            mask = (dist == d)
            if mask.any():
                per_target[d].append(float(gnorm[mask].mean()))
            else:
                per_target[d].append(float("nan"))

    # average across targets, ignoring NaNs (no nodes at that distance)
    mean_curve = np.array([
        np.nanmean(per_target[d]) if np.any(np.isfinite(per_target[d])) else np.nan
        for d in range(max_hops + 1)
    ], dtype=np.float64)

    if normalize_by_d0 and np.isfinite(mean_curve[0]) and mean_curve[0] != 0.0:
        norm_curve = mean_curve / mean_curve[0]
    else:
        norm_curve = mean_curve.copy()

    dists = np.arange(max_hops + 1, dtype=np.int32)
    return dists, mean_curve, norm_curve


# =============================================================================
# Plot / save
# =============================================================================

def save_curve_csv(path: str, dists: np.ndarray, curve: np.ndarray, curve_norm: np.ndarray):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["distance", "influence", "influence_norm"])
        for d, v, vn in zip(dists.tolist(), curve.tolist(), curve_norm.tolist()):
            w.writerow([d, v, vn])


def plot_dataset_curves(out_png: str, title: str, curves: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """
    curves: label -> (dists, norm_curve)
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[warn] matplotlib not available; skipping plots.")
        return

    plt.figure(figsize=(8.5, 5.0))
    for label, (dists, y) in curves.items():
        plt.plot(dists, y, marker="o", linewidth=2, label=label)
    plt.xlabel("Hop distance d")
    plt.ylabel("Influence I(d) normalized by I(0)")
    plt.title(title)
    plt.yscale("log")  # log scale highlights decay (oversquashing)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="oversquash_out")
    parser.add_argument("--device", type=str, default="cpu")

    # training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--early_stopping", type=int, default=50)
    parser.add_argument("--stop_strategy", type=str, default="acc", choices=["acc", "loss"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_if_missing", type=int, default=1)  # 1: train if ckpt missing, 0: require ckpt

    # oversquashing probe
    parser.add_argument("--max_hops", type=int, default=10)
    parser.add_argument("--num_targets", type=int, default=64)
    parser.add_argument("--target_split", type=str, default="test", choices=["train", "val", "test", "all"])

    # model hyperparams (base)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--layers_nsd", type=int, default=6)     # deeper baseline (1-hop per layer)
    parser.add_argument("--layers_poly", type=int, default=2)    # shallow poly
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    # poly receptive field
    parser.add_argument("--poly_layers_K", type=int, default=10)
    parser.add_argument("--polynomial_type", type=str, default="ChebyshevType1")
    parser.add_argument("--lambda_max_choice", type=str, default="analytic", choices=["analytic", "iterative", "none"])

    args = parser.parse_args()
    ensure_dir(args.out_dir)
    set_seed(args.seed)

    device = as_device(args.device)

    datasets = ["squirrel", "chameleon", "pubmed"]
    families = ["diag", "bundle", "general"]
    variants = ["nsd", "poly"]

    # store normalized curves for combined plots
    combined: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {d: {} for d in datasets}

    for ds_name in datasets:
        # load dataset data
        data = load_dataset(ds_name, fold=args.fold, seed=args.seed, root=os.path.join(args.out_dir, ".data"))
        # ensure masks exist
        if not hasattr(data, "train_mask"):
            data = _ensure_masks(data, seed=args.seed + args.fold)

        # move to device for training/influence
        data = data.to(device)

        # fill cfg
        base_cfg = make_default_cfg()
        base_cfg.update(
            d=args.d,
            hidden_channels=args.hidden_channels,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            sheaf_decay=args.weight_decay,   # keep simple & consistent
            polynomial_type=args.polynomial_type,
            poly_layers_K=args.poly_layers_K,
            lambda_max_choice=(None if args.lambda_max_choice == "none" else args.lambda_max_choice),
            device=str(device),
        )
        base_cfg["graph_size"] = int(data.num_nodes)
        base_cfg["input_dim"] = int(data.x.size(-1))
        base_cfg["output_dim"] = int(torch.unique(data.y).numel())

        print(f"\n================ Dataset: {ds_name} | N={data.num_nodes} E={data.edge_index.size(1)} =================")

        for fam in families:
            for var in variants:
                cfg = copy.deepcopy(base_cfg)

                # “matched reach” heuristic:
                # - NSD uses more layers (sequential diffusion)
                # - Poly uses fewer layers but higher K (global polynomial reach)
                if var == "nsd":
                    cfg["layers"] = int(args.layers_nsd)
                else:
                    cfg["layers"] = int(args.layers_poly)
                    cfg["poly_layers_K"] = int(args.poly_layers_K)

                tag = f"{ds_name}_{fam}_{var}_fold{args.fold}_seed{args.seed}"
                ckpt_dir = ensure_dir(os.path.join(args.out_dir, "ckpts"))
                ckpt_path = os.path.join(ckpt_dir, f"{tag}.pt")

                # build model
                model = build_model(fam, var, data.edge_index, cfg).to(device)

                # load/train
                if os.path.exists(ckpt_path):
                    sd = torch.load(ckpt_path, map_location=device)
                    model.load_state_dict(sd)
                    print(f"[load] {tag} from {ckpt_path}")
                else:
                    if int(args.train_if_missing) != 1:
                        raise RuntimeError(f"Checkpoint missing for {tag}: {ckpt_path} (use --train_if_missing 1)")
                    print(f"[train] {tag} ...")
                    t0 = time.time()
                    summ = fit_model(
                        model=model,
                        data=data,
                        cfg=cfg,
                        epochs=args.epochs,
                        early_stopping=args.early_stopping,
                        stop_strategy=args.stop_strategy,
                    )
                    dt = time.time() - t0
                    print(f"  done in {dt:.1f}s | best_epoch={summ['best_epoch']} | "
                          f"train={summ['train_acc']:.3f} val={summ['val_acc']:.3f} test={summ['test_acc']:.3f}")
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"[save] {ckpt_path}")

                # influence curve
                print(f"[probe] influence vs distance | max_hops={args.max_hops} targets={args.num_targets} split={args.target_split}")
                dists, curve, curve_norm = influence_by_distance(
                    model=model,
                    data=data,
                    max_hops=int(args.max_hops),
                    num_targets=int(args.num_targets if ds_name != "pubmed" else min(args.num_targets, 64)),
                    target_split=args.target_split,
                    seed=args.seed + args.fold,
                    device=device,
                    normalize_by_d0=True,
                )

                # save
                curve_dir = ensure_dir(os.path.join(args.out_dir, "curves", ds_name))
                csv_path = os.path.join(curve_dir, f"{fam}_{var}.csv")
                save_curve_csv(csv_path, dists, curve, curve_norm)
                print(f"[save] curve -> {csv_path}")

                # record for combined plot
                label = f"{fam.upper()}-{var.upper()}"
                combined[ds_name][label] = (dists, curve_norm)

        # per-dataset combined plot (6 curves)
        png_path = os.path.join(args.out_dir, f"{ds_name}_oversquash_influence.png")
        plot_dataset_curves(
            out_png=png_path,
            title=f"{ds_name.upper()} | Influence decay vs hop distance (normalized) | log-scale",
            curves=combined[ds_name],
        )
        print(f"[plot] {png_path}")

    print("\nAll done. Outputs:")
    print(f"  - ckpts: {os.path.join(args.out_dir, 'ckpts')}")
    print(f"  - curves (CSV): {os.path.join(args.out_dir, 'curves')}")
    print(f"  - plots (PNG): {args.out_dir}")


if __name__ == "__main__":
    main()
