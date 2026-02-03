#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_spectral_response_grid_d_sweep.py

Appendix experiment: spectral-response diagnostics across datasets/models/layers/d.

Grid (fixed K)
--------------
- poly_layers_K fixed to 3 (no sweep in this paper)
- d swept over {2, 3, 4}
- layers swept over {2, 3, 4}
- model swept over {DiagSheafPolynomial, BundleSheafPolynomial, GeneralSheafPolynomial}
- dataset swept over {"chameleon", "squirrel", "pubmed", "citeseer"}

Outputs
-------
- One spectral-response PNG per run in --out_dir
- One CSV row per run appended to --csv_path
- Logs to Weights & Biases project "spectral_response" (one run per config + aggregate table)

Example
-------
python visualizations/plot_spectral_response.py \
  --cuda 0 \
  --epochs 200 \
  --early_stopping 200 \
  --lr 0.02 \
  --maps_lr 0.005 \
  --dropout 0.7 \
  --weight_decay 0.005 \
  --sheaf_decay 0.005 \
  --hidden_channels 20 \
  --left_weights True \
  --right_weights True \
  --normalised True \
  --deg_normalised False \
  --sparse_learner True \
  --polynomial_type ChebyshevType1 \
  --seeds 0,1,2 \
  --out_dir visualizations/figures/spectral_grid \
  --csv_path visualizations/figures/spectral_grid/results.csv
"""

import os
import sys
import pathlib
import random
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -------- repo path setup --------
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# -------- repo imports --------
try:
    from exp.parser import get_parser
    from utils.heterophilic import get_dataset, get_fixed_splits
    from models.disc_models import (
        DiscreteDiagSheafDiffusionPolynomial,
        DiscreteBundleSheafDiffusionPolynomial,
        DiscreteGeneralSheafDiffusionPolynomial,
    )
except Exception as e:
    raise RuntimeError(
        "Failed to import repo modules. Run from within the repo environment.\n"
        f"Import error: {e}"
    )

try:
    import wandb
    _WANDB_OK = True
except Exception:
    _WANDB_OK = False


# ----------------------------- polynomial bases on [-1,1] -----------------------------
def chebyshev_T(x, K):
    T0 = torch.ones_like(x)
    if K == 0:
        return [T0]
    T1 = x
    Ts = [T0, T1]
    for _ in range(1, K):
        Ts.append(2 * x * Ts[-1] - Ts[-2])
    return Ts

def chebyshev_U(x, K):
    U0 = torch.ones_like(x)
    if K == 0:
        return [U0]
    U1 = 2 * x
    Us = [U0, U1]
    for _ in range(1, K):
        Us.append(2 * x * Us[-1] - Us[-2])
    return Us

def chebyshev_V(x, K):
    V0 = torch.ones_like(x)
    if K == 0:
        return [V0]
    V1 = 2 * x - 1
    Vs = [V0, V1]
    for _ in range(1, K):
        Vs.append(2 * x * Vs[-1] - Vs[-2])
    return Vs

def chebyshev_W(x, K):
    W0 = torch.ones_like(x)
    if K == 0:
        return [W0]
    W1 = 2 * x + 1
    Ws = [W0, W1]
    for _ in range(1, K):
        Ws.append(2 * x * Ws[-1] - Ws[-2])
    return Ws

def legendre_P(x, K):
    P0 = torch.ones_like(x)
    if K == 0:
        return [P0]
    P1 = x
    Ps = [P0, P1]
    for n in range(1, K):
        Ps.append(((2*n + 1) * x * Ps[-1] - n * Ps[-2]) / (n + 1))
    return Ps

def gegenbauer_C(x, K, lam):
    C0 = torch.ones_like(x)
    if K == 0:
        return [C0]
    C1 = 2 * lam * x
    Cs = [C0, C1]
    for n in range(2, K + 1):
        Cs.append((2*(n+lam-1)*x*Cs[-1] - (n+2*lam-2)*Cs[-2]) / n)
    return Cs

def jacobi_P(x, K, alpha, beta):
    P0 = torch.ones_like(x)
    if K == 0:
        return [P0]
    P1 = 0.5 * ((2 + alpha + beta) * x + (alpha - beta))
    Ps = [P0, P1]
    for n in range(1, K):
        nn = float(n)
        A = 2*(nn+1)*(nn+alpha+beta+1)*(2*nn+alpha+beta)
        B = (2*nn+alpha+beta+1)*(alpha**2 - beta**2)
        C = (2*nn+alpha+beta)*(2*nn+alpha+beta+1)*(2*nn+alpha+beta+2)
        D = 2*(nn+alpha)*(nn+beta)*(2*nn+alpha+beta+2)
        Ps.append(((B + C * x) * Ps[-1] - D * Ps[-2]) / A)
    return Ps

def eval_basis(polynomial_type, x, K, gegenbauer_lambda=1.0, jacobi_alpha=0.0, jacobi_beta=0.0):
    pt = (polynomial_type or "ChebyshevType1").lower()
    if pt in ("chebyshev", "chebyshevtype1", "chebyshev_type1", "t1"):
        return chebyshev_T(x, K)
    if pt in ("chebyshevtype2", "chebyshev_type2", "t2", "u"):
        return chebyshev_U(x, K)
    if pt in ("chebyshevtype3", "chebyshev_type3", "t3", "v"):
        return chebyshev_V(x, K)
    if pt in ("chebyshevtype4", "chebyshev_type4", "t4", "w"):
        return chebyshev_W(x, K)
    if pt in ("legendre",):
        return legendre_P(x, K)
    if pt in ("gegenbauer",):
        return gegenbauer_C(x, K, float(gegenbauer_lambda))
    if pt in ("jacobi",):
        return jacobi_P(x, K, float(jacobi_alpha), float(jacobi_beta))
    raise ValueError(f"Unknown polynomial_type={polynomial_type}")


# ----------------------------- train / eval -----------------------------
def train_one_epoch(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x)[data.train_mask]
    loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())

@torch.no_grad()
def eval_acc_loss(model, data, mask):
    model.eval()
    logits = model(data.x)
    loss = F.nll_loss(logits[mask], data.y[mask]).detach().cpu().item()
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return float(acc), float(loss)


# ----------------------------- spectral curve + summary stats -----------------------------
@torch.no_grad()
def compute_response_curve(model, num_points=400, device="cpu"):
    lam_max = float(getattr(model, "lambda_max", 2.0))
    lam = torch.linspace(0.0, lam_max, steps=int(num_points), device=device)
    xi = 2.0 * lam / lam_max - 1.0

    polynomial_type = getattr(model, "polynomial_type", None)
    K = int(getattr(model, "K", 3))

    if hasattr(model, "poly_logits"):
        alpha = torch.softmax(model.poly_logits.detach(), dim=0)
        K = int(alpha.numel() - 1)
    elif hasattr(model, "poly_coeffs"):
        alpha = model.poly_coeffs.detach()
        K = int(alpha.numel() - 1)
    else:
        alpha = torch.ones(K + 1, device=device) / float(K + 1)

    gc_lambda = float(getattr(model, "gc_lambda", getattr(model, "gegenbauer_lambda", 1.0)))
    jac_a = float(getattr(model, "jac_alpha", getattr(model, "jacobi_alpha", 0.0)))
    jac_b = float(getattr(model, "jac_beta", getattr(model, "jacobi_beta", 0.0)))

    basis = eval_basis(polynomial_type, xi, K, gegenbauer_lambda=gc_lambda, jacobi_alpha=jac_a, jacobi_beta=jac_b)

    p = torch.zeros_like(xi)
    for k in range(K + 1):
        p = p + alpha[k] * basis[k]

    hp_alpha = float(model.hp_alpha.detach().cpu().item()) if hasattr(model, "hp_alpha") else 0.0
    hp = hp_alpha * (1.0 - lam / lam_max)
    m = p + hp

    info = dict(
        lambda_max=lam_max,
        K=K,
        polynomial_type=polynomial_type,
        hp_alpha=hp_alpha,
        alpha=alpha.detach().cpu().numpy().tolist(),
        gc_lambda=gc_lambda,
        jacobi_alpha=jac_a,
        jacobi_beta=jac_b,
    )
    return lam.detach().cpu(), m.detach().cpu(), p.detach().cpu(), hp.detach().cpu(), info

def spectral_summary(lam: torch.Tensor, m: torch.Tensor, eps: float = 1e-8) -> Dict[str, float]:
    lam_np = lam.numpy()
    m_np = m.numpy()
    lam_max = float(lam_np.max()) if lam_np.size else 1.0

    low_mask = lam_np <= (0.2 * lam_max)
    high_mask = lam_np >= (0.8 * lam_max)

    g_low = float(m_np[low_mask].mean()) if low_mask.any() else float(m_np.mean())
    g_high = float(m_np[high_mask].mean()) if high_mask.any() else float(m_np.mean())
    ratio = float(g_high / (g_low + eps))

    dm = np.diff(m_np)
    sgn = np.sign(dm)
    sgn = sgn[sgn != 0]
    sign_changes = int(np.sum(sgn[1:] * sgn[:-1] < 0)) if sgn.size >= 2 else 0

    return {
        "G_low": g_low,
        "G_high": g_high,
        "G_high_over_low": ratio,
        "nonmonotone_sign_changes": float(sign_changes),
    }

def plot_and_save(lam, m, p, hp, out_png, title_suffix=""):
    plt.figure()
    plt.plot(lam.numpy(), m.numpy(), label="m(λ)=poly+HP (approx.)")
    plt.plot(lam.numpy(), p.numpy(), label="poly p(ξ(λ))")
    plt.plot(lam.numpy(), hp.numpy(), label="HP α_hp(1-λ/λ_max) (approx.)")
    plt.xlabel("Laplacian eigenvalue λ")
    plt.ylabel("gain / multiplier")
    plt.title(f"Spectral response {title_suffix}".strip())
    plt.legend()
    plt.tight_layout()
    out_png = str(out_png)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


# ----------------------------- csv helper -----------------------------
def append_csv(csv_path: str, row: Dict[str, Any]) -> None:
    import csv
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ----------------------------- experiment grid -----------------------------
DATASETS = ["chameleon", "squirrel", "pubmed", "citeseer"]
MODELS = ["DiagSheafPolynomial", "BundleSheafPolynomial", "GeneralSheafPolynomial"]
LAYER_CHOICES = [2, 3, 4]
D_CHOICES = [2, 3, 4]
POLY_K_FIXED = 3


def route_model(model_name: str):
    if model_name == "DiagSheafPolynomial":
        return DiscreteDiagSheafDiffusionPolynomial
    if model_name == "BundleSheafPolynomial":
        return DiscreteBundleSheafDiffusionPolynomial
    if model_name == "GeneralSheafPolynomial":
        return DiscreteGeneralSheafDiffusionPolynomial
    raise ValueError(f"Unknown model: {model_name}")


def parse_args():
    repo_parser = get_parser()
    repo_args, _ = repo_parser.parse_known_args()

    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--out_dir", type=str, default="visualizations/figures/spectral_grid")
    extra.add_argument("--csv_path", type=str, default="visualizations/figures/spectral_grid/results.csv")
    extra.add_argument("--num_points", type=int, default=400)
    extra.add_argument("--single_fold", type=int, default=0)
    extra.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds, e.g. '0,1,2'")
    extra.add_argument("--wandb_project", type=str, default="spectral_response")
    extra.add_argument("--wandb_entity", type=str, default=None)
    extra.add_argument("--wandb_group", type=str, default="grid_polyK3")
    extra.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    extra.add_argument("--quiet", action="store_true")
    extra_args, _ = extra.parse_known_args()
    return repo_args, extra_args


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_args_dict(repo_args, dataset_name: str, model_name: str, layers: int, fiber_d: int, device) -> Dict[str, Any]:
    # IMPORTANT: do not shadow fiber_d with the config dict name.
    cfg = vars(repo_args).copy()

    cfg["dataset"] = dataset_name
    cfg["model"] = model_name
    cfg["layers"] = int(layers)
    cfg["d"] = int(fiber_d)

    # enforce fixed K=3 (multiple aliases for safety)
    cfg["poly_layers_K"] = int(POLY_K_FIXED)
    cfg["chebyshev_layers_K"] = int(POLY_K_FIXED)
    cfg["K"] = int(POLY_K_FIXED)

    if cfg.get("polynomial_type", None) is None:
        cfg["polynomial_type"] = "ChebyshevType1"

    cfg["device"] = device

    # Bundle model constraint
    if model_name == "BundleSheafPolynomial":
        cfg["deg_normalised"] = False

    return cfg


def run_one_config(repo_args, extra_args, dataset_name: str, model_name: str, layers: int, fiber_d: int, seed: int) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device(f"cuda:{repo_args.cuda}" if torch.cuda.is_available() else "cpu")

    dataset = get_dataset(dataset_name)
    data = dataset[0]
    data = get_fixed_splits(data, dataset_name, int(extra_args.single_fold))
    data = data.to(device)

    args_dict = build_args_dict(repo_args, dataset_name, model_name, layers, fiber_d, device)
    args_dict["graph_size"] = int(data.x.size(0))
    args_dict["input_dim"] = int(data.x.size(1))
    try:
        args_dict["output_dim"] = int(dataset.num_classes)
    except Exception:
        args_dict["output_dim"] = int(torch.unique(data.y).numel())

    model_cls = route_model(model_name)
    model = model_cls(data.edge_index, args_dict).to(device)

    lr = float(getattr(repo_args, "lr", 0.01))
    maps_lr = getattr(repo_args, "maps_lr", None)
    weight_decay = float(getattr(repo_args, "weight_decay", 0.0))
    sheaf_decay = float(getattr(repo_args, "sheaf_decay", weight_decay))

    if hasattr(model, "grouped_parameters"):
        sheaf_params, other_params = model.grouped_parameters()
        opt = torch.optim.Adam([
            {"params": sheaf_params, "lr": float(maps_lr) if maps_lr is not None else lr, "weight_decay": sheaf_decay},
            {"params": other_params, "lr": lr, "weight_decay": weight_decay},
        ])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = int(getattr(repo_args, "epochs", 200))
    early_stopping = int(getattr(repo_args, "early_stopping", 200))
    stop_strategy = str(getattr(repo_args, "stop_strategy", "acc")).lower()

    best_val = -1.0
    best_val_loss = float("inf")
    best_test = -1.0
    best_epoch = 0
    best_state = None
    bad = 0

    for ep in range(epochs):
        loss = train_one_epoch(model, opt, data)
        tr_acc, tr_loss = eval_acc_loss(model, data, data.train_mask)
        va_acc, va_loss = eval_acc_loss(model, data, data.val_mask)
        te_acc, te_loss = eval_acc_loss(model, data, data.test_mask)

        improved = (va_acc > best_val) if stop_strategy == "acc" else (va_loss < best_val_loss)
        if improved:
            best_val = float(va_acc)
            best_val_loss = float(va_loss)
            best_test = float(te_acc)
            best_epoch = int(ep)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if (not extra_args.quiet) and (ep % 20 == 0 or ep == epochs - 1):
            print(f"[{dataset_name} | {model_name} | d={fiber_d} | layers={layers} | seed={seed}] "
                  f"epoch {ep:04d} | loss {loss:.4f} | acc tr/va/te {tr_acc:.3f}/{va_acc:.3f}/{te_acc:.3f}")

        if bad >= early_stopping:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    lam, m, p, hp, info = compute_response_curve(model, num_points=int(extra_args.num_points), device=device)
    spec_stats = spectral_summary(lam, m)

    run_tag = f"{dataset_name}__{model_name}__d{fiber_d}__L{layers}__K{POLY_K_FIXED}__seed{seed}"
    out_png = os.path.join(extra_args.out_dir, f"spectral_{run_tag}.png")
    title_suffix = f"({dataset_name}, {model_name}, d={fiber_d}, layers={layers}, K={POLY_K_FIXED}, seed={seed})"
    plot_and_save(lam, m, p, hp, out_png, title_suffix=title_suffix)

    alpha = info.get("alpha", [])
    alpha_dict = {f"alpha_{k}": (alpha[k] if k < len(alpha) else None) for k in range(POLY_K_FIXED + 1)}

    row = {
        "dataset": dataset_name,
        "model": model_name,
        "d": int(fiber_d),
        "layers": int(layers),
        "poly_K": int(POLY_K_FIXED),
        "polynomial_type": info.get("polynomial_type"),
        "lambda_max": float(info.get("lambda_max", 2.0)),
        "hp_alpha": float(info.get("hp_alpha", 0.0)),
        "seed": int(seed),
        "fold": int(extra_args.single_fold),
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val),
        "best_test_acc": float(best_test),
        "out_png": out_png,
        **spec_stats,
        **alpha_dict,
    }
    return row


def main():
    repo_args, extra_args = parse_args()

    # enforce K=3 regardless of CLI
    setattr(repo_args, "poly_layers_K", POLY_K_FIXED)
    setattr(repo_args, "chebyshev_layers_K", POLY_K_FIXED)

    seeds = [int(s.strip()) for s in str(extra_args.seeds).split(",") if s.strip()]

    use_wandb = _WANDB_OK and (extra_args.wandb_mode != "disabled")
    if use_wandb:
        os.environ.setdefault("WANDB_MODE", extra_args.wandb_mode)
        os.environ.setdefault("WANDB_SILENT", "true")

    all_rows: List[Dict[str, Any]] = []

    for ds in DATASETS:
        for model in MODELS:
            for fiber_d in D_CHOICES:
                for L in LAYER_CHOICES:
                    for seed in seeds:
                        wb_run = None
                        if use_wandb:
                            wb_run = wandb.init(
                                project=extra_args.wandb_project,
                                entity=extra_args.wandb_entity,
                                group=extra_args.wandb_group,
                                name=f"{ds}-{model}-d{fiber_d}-L{L}-K{POLY_K_FIXED}-seed{seed}",
                                config={**vars(repo_args),
                                        "grid_dataset": ds,
                                        "grid_model": model,
                                        "grid_d": fiber_d,
                                        "grid_layers": L,
                                        "poly_layers_K": POLY_K_FIXED,
                                        "chebyshev_layers_K": POLY_K_FIXED,
                                        "seed": seed,
                                        "single_fold": int(extra_args.single_fold)},
                                reinit=True,
                            )

                        row = run_one_config(repo_args, extra_args, ds, model, L, fiber_d, seed)
                        all_rows.append(row)
                        append_csv(extra_args.csv_path, row)

                        if use_wandb and wb_run is not None:
                            wandb.log({k: v for k, v in row.items() if isinstance(v, (int, float))})
                            try:
                                wandb.log({"spectral_response_plot": wandb.Image(row["out_png"])})
                            except Exception:
                                pass
                            wandb.finish()

    # aggregate table run
    if use_wandb and all_rows:
        wb_run = wandb.init(
            project=extra_args.wandb_project,
            entity=extra_args.wandb_entity,
            group=extra_args.wandb_group,
            name="aggregate_table",
            config={"poly_layers_K": POLY_K_FIXED, "datasets": DATASETS, "models": MODELS, "d_grid": D_CHOICES, "layers_grid": LAYER_CHOICES},
            reinit=True,
        )
        cols = list(all_rows[0].keys())
        table = wandb.Table(columns=cols)
        for r in all_rows:
            table.add_data(*[r.get(c) for c in cols])
        wandb.log({"results_table": table})
        try:
            wandb.save(extra_args.csv_path)
        except Exception:
            pass
        wandb.finish()

    print("\nDone.")
    print("Saved CSV:", extra_args.csv_path)
    print("Saved plots in:", extra_args.out_dir)
    print(f"Total runs: {len(all_rows)}")


if __name__ == "__main__":
    main()
