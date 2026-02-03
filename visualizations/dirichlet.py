#!/usr/bin/env python3
"""
visualizations/dirichlet.py

Measures normalized Dirichlet energy during one forward pass WITHOUT modifying model code.

What it logs:
  - NSD: after each sparse diffusion application (captures BOTH torch_sparse.spmm and SparseTensor.matmul)
  - PolyNSD: after each model._poly_eval output (best-effort signature handling)

Grid:
  seeds={0,1,2}, d={2,3,4}, layers={2,3,4}
Datasets: Chameleon, Squirrel, Pubmed, Cora
Variants: Diag, Bundle, General
Plots: 3 figures (Diag/Bundle/General), each 2x2 panels of datasets, with mean ± std bands.

Run:
  python visualizations/dirichlet.py collect --outdir visualizations/results_dirichlet --device cuda:0
  python visualizations/dirichlet.py plot --outdir visualizations/results_dirichlet --figdir visualizations/figures/figs_dirichlet --max_layers 4
"""

import argparse, os, glob, json, random, sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch_sparse
from torch_sparse import SparseTensor


# ------------------------ repo path ------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ------------------------ Dirichlet energy ------------------------

@torch.no_grad()
def safe_spmm(orig_spmm, row, col, m, n, vals, mat):
    """
    Call torch_sparse.spmm robustly across installations:
      - some accept spmm(row, col, m, n, val, mat)
      - some accept spmm(index, val, m, n, mat) with index shape [2, E]
    """
    try:
        return orig_spmm(row, col, m, n, vals, mat)  # 6-arg form
    except TypeError:
        idx = torch.stack([row, col], dim=0)
        return orig_spmm(idx, vals, m, n, mat)       # 5-arg form


@torch.no_grad()
def dirichlet_energy_norm_with_spmm(orig_spmm, idx: Tuple[torch.Tensor, torch.Tensor], vals: torch.Tensor, x: torch.Tensor, eps=1e-12) -> float:
    """
    E_norm(x) = <x, Lx> / <x, x>, averaged over channels.
    Uses orig_spmm (unpatched) to avoid recursion.
    """
    row, col = idx
    m = x.size(0)
    Lx = safe_spmm(orig_spmm, row, col, m, m, vals, x)
    num = (x * Lx).sum(dim=0).mean()
    den = (x * x).sum(dim=0).mean()
    return float((num / (den + eps)).item())


def _extract_row_col_vals_from_idx_vals(idx, vals):
    """
    idx can be:
      - tuple(row, col)
      - Tensor [2, E]
    """
    if isinstance(idx, (tuple, list)) and len(idx) == 2:
        row, col = idx
        return row, col, vals
    if torch.is_tensor(idx) and idx.dim() == 2 and idx.size(0) == 2:
        row, col = idx[0], idx[1]
        return row, col, vals
    return None


# ------------------------ Monkeypatch logger ------------------------

class ForwardEnergyLogger:
    """
    Collects per-layer energies during ONE forward call.

    Hooks:
      - torch_sparse.spmm  (if used)
      - SparseTensor.matmul (if used)

    Also wraps model._poly_eval (best effort) for PolyNSD.

    It will PRINT a warning if it sees ZERO NSD events for a forward pass.
    """

    def __init__(self, max_layers: int, verbose_first_error: bool = True):
        self.max_layers = max_layers
        self.verbose_first_error = verbose_first_error
        self.reset()

        self._orig_spmm = None
        self._orig_st_matmul = None
        self._orig_poly = None
        self._model = None

        self._printed_error = False

    def reset(self):
        self.nsd_layer = 0
        self.poly_layer = 0
        self.nsd_E = np.full((self.max_layers,), np.nan, dtype=np.float64)
        self.poly_E = np.full((self.max_layers,), np.nan, dtype=np.float64)

    def _maybe_print_err(self, where: str, e: Exception):
        if self.verbose_first_error and (not self._printed_error):
            self._printed_error = True
            print(f"[dirichlet logger] first error at {where}: {type(e).__name__}: {e}")

    def _patch_spmm(self):
        self._orig_spmm = torch_sparse.spmm

        def spmm_wrapped(*args):
            # Two common call patterns:
            #  5 args: (index, value, m, n, mat)
            #  6 args: (row, col, m, n, value, mat)
            row = col = vals = out = None

            if len(args) == 5:
                idx, vals, m, n, mat = args
                if torch.is_tensor(idx) and idx.dim() == 2 and idx.size(0) == 2:
                    row, col = idx[0], idx[1]
                out = self._orig_spmm(idx, vals, m, n, mat)
            elif len(args) == 6:
                row, col, m, n, vals, mat = args
                out = self._orig_spmm(row, col, m, n, vals, mat)
            else:
                return self._orig_spmm(*args)

            # log energy on output
            if (row is not None) and (col is not None) and (vals is not None) and (self.nsd_layer < self.max_layers):
                try:
                    self.nsd_E[self.nsd_layer] = dirichlet_energy_norm_with_spmm(self._orig_spmm, (row, col), vals, out)
                except Exception as e:
                    self._maybe_print_err("spmm energy", e)

            self.nsd_layer += 1
            return out

        torch_sparse.spmm = spmm_wrapped

    def _patch_sparse_tensor_matmul(self):
        self._orig_st_matmul = SparseTensor.matmul

        def matmul_wrapped(st: SparseTensor, other, reduce: str = "sum"):
            out = self._orig_st_matmul(st, other, reduce=reduce)
            if self.nsd_layer < self.max_layers:
                try:
                    row, col, val = st.coo()
                    self.nsd_E[self.nsd_layer] = dirichlet_energy_norm_with_spmm(self._orig_spmm, (row, col), val, out)
                except Exception as e:
                    self._maybe_print_err("SparseTensor.matmul energy", e)
            self.nsd_layer += 1
            return out

        SparseTensor.matmul = matmul_wrapped

    def _patch_poly_eval(self, model):
        if not hasattr(model, "_poly_eval"):
            return
        self._model = model
        self._orig_poly = model._poly_eval

        def poly_wrapped(*a, **kw):
            out = self._orig_poly(*a, **kw)

            if self.poly_layer < self.max_layers:
                try:
                    # Best-effort extraction of (idx, vals, x) from args
                    idx = vals = None

                    if len(a) >= 2:
                        idx = a[0]
                        vals = a[1]
                    elif "idx" in kw and "vals" in kw:
                        idx, vals = kw["idx"], kw["vals"]

                    extracted = _extract_row_col_vals_from_idx_vals(idx, vals)
                    if extracted is not None:
                        row, col, v = extracted
                        self.poly_E[self.poly_layer] = dirichlet_energy_norm_with_spmm(self._orig_spmm, (row, col), v, out)
                    else:
                        # If idx is a SparseTensor
                        if isinstance(idx, SparseTensor):
                            r, c, v = idx.coo()
                            self.poly_E[self.poly_layer] = dirichlet_energy_norm_with_spmm(self._orig_spmm, (r, c), v, out)

                except Exception as e:
                    self._maybe_print_err("_poly_eval energy", e)

            self.poly_layer += 1
            return out

        model._poly_eval = poly_wrapped

    def attach(self, model):
        # patch both paths
        self._patch_spmm()
        self._patch_sparse_tensor_matmul()
        self._patch_poly_eval(model)

    def detach(self):
        if self._orig_spmm is not None:
            torch_sparse.spmm = self._orig_spmm
            self._orig_spmm = None
        if self._orig_st_matmul is not None:
            SparseTensor.matmul = self._orig_st_matmul
            self._orig_st_matmul = None
        if self._model is not None and self._orig_poly is not None:
            self._model._poly_eval = self._orig_poly
        self._model = None
        self._orig_poly = None


# ------------------------ minimal train/eval ------------------------

def train_epoch(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x)[data.train_mask]
    loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())

@torch.no_grad()
def eval_val(model, data) -> float:
    model.eval()
    logits = model(data.x)
    pred = logits[data.val_mask].max(1)[1]
    return float(pred.eq(data.y[data.val_mask]).float().mean().item())


# ------------------------ model routing ------------------------

def build_model(model_name: str, edge_index, args: Dict[str, Any]):
    from models.disc_models import (
        DiscreteDiagSheafDiffusion, DiscreteBundleSheafDiffusion, DiscreteGeneralSheafDiffusion,
        DiscreteDiagSheafDiffusionPolynomial, DiscreteBundleSheafDiffusionPolynomial, DiscreteGeneralSheafDiffusionPolynomial
    )
    if model_name == "DiagSheaf":
        return DiscreteDiagSheafDiffusion(edge_index, args)
    if model_name == "BundleSheaf":
        return DiscreteBundleSheafDiffusion(edge_index, args)
    if model_name == "GeneralSheaf":
        return DiscreteGeneralSheafDiffusion(edge_index, args)
    if model_name == "DiagSheafPolynomial":
        return DiscreteDiagSheafDiffusionPolynomial(edge_index, args)
    if model_name == "BundleSheafPolynomial":
        return DiscreteBundleSheafDiffusionPolynomial(edge_index, args)
    if model_name == "GeneralSheafPolynomial":
        return DiscreteGeneralSheafDiffusionPolynomial(edge_index, args)
    raise ValueError(model_name)

def model_names_for_variant(variant: str):
    if variant == "Diag":
        return "DiagSheaf", "DiagSheafPolynomial"
    if variant == "Bundle":
        return "BundleSheaf", "BundleSheafPolynomial"
    if variant == "General":
        return "GeneralSheaf", "GeneralSheafPolynomial"
    raise ValueError(variant)


# ------------------------ collect ------------------------

def collect_one(
    *,
    outdir: str,
    dataset_name: str,
    variant: str,
    kind: str,   # "NSD" or "PolyNSD"
    seed: int,
    d: int,
    layers: int,
    device: torch.device,
    epochs: int,
    early_stopping: int,
    base_args: Dict[str, Any],
    fold: int = 0,
):
    from utils.heterophilic import get_dataset, get_fixed_splits

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    dataset = get_dataset(dataset_name)
    data = get_fixed_splits(dataset[0], dataset_name, fold).to(device)

    args = dict(base_args)
    args["d"] = d
    args["layers"] = layers
    args["graph_size"] = int(data.x.size(0))
    args["input_dim"] = int(data.x.size(1))
    try:
        args["output_dim"] = int(dataset.num_classes)
    except Exception:
        args["output_dim"] = int(torch.unique(data.y).numel())
    args["device"] = device

    nsd_name, poly_name = model_names_for_variant(variant)
    model_name = poly_name if kind == "PolyNSD" else nsd_name

    model = build_model(model_name, data.edge_index, args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args["lr"]), weight_decay=float(args["weight_decay"]))

    best_val = -1.0
    best_state = None
    bad = 0
    for _ in range(epochs):
        train_epoch(model, opt, data)
        v = eval_val(model, data)
        if v > best_val:
            best_val = v
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= early_stopping:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # probe on one forward
    logger = ForwardEnergyLogger(max_layers=layers, verbose_first_error=True)
    logger.attach(model)
    model.eval()
    with torch.no_grad():
        _ = model(data.x)
    logger.detach()

    if kind == "NSD" and logger.nsd_layer == 0:
        print(f"[warn] no NSD sparse ops captured for {dataset_name}/{variant}/{kind} (seed={seed}, d={d}, L={layers})")

    curve = logger.poly_E if kind == "PolyNSD" else logger.nsd_E

    os.makedirs(outdir, exist_ok=True)
    fname = f"{dataset_name}__{variant}__{kind}__seed{seed}__d{d}__L{layers}.npz"
    np.savez(
        os.path.join(outdir, fname),
        dirichlet_norm=curve,
        meta=json.dumps(dict(dataset=dataset_name, variant=variant, kind=kind, seed=seed, d=d, layers=layers)),
    )
    print("Saved", fname, "curve=", curve)


def collect_grid(args):
    device = torch.device(args.device)

    base_args = dict(
        hidden_channels=32, hidden_dim=32,
        dropout=0.3, input_dropout=0.4, sheaf_dropout=0.0,
        normalised=True, deg_normalised=False,
        add_hp=1, add_lp=0,
        use_epsilons=True,
        left_weights=True, right_weights=True,
        linear=False, nonlinear=False, sparse_learner=True,
        sheaf_act="tanh",
        second_linear=False,
        use_embedding=True, use_act=True,
        edge_weights=True,
        lambda_max_choice="analytic",
        poly_layers_K=5,
        polynomial_type="ChebyshevType1",
        orth="householder", orth_trans="householder",
        lr=0.01, weight_decay=5e-4, sheaf_decay=5e-4, maps_lr=None,
    )

    variants = ["Diag", "Bundle", "General"]
    kinds = ["NSD", "PolyNSD"]

    for ds in args.datasets:
        for variant in variants:
            for seed in args.seeds:
                for d in args.ds:
                    for L in args.layers:
                        for kind in kinds:
                            try:
                                collect_one(
                                    outdir=args.outdir, dataset_name=ds, variant=variant, kind=kind,
                                    seed=seed, d=d, layers=L, device=device,
                                    epochs=args.epochs, early_stopping=args.early_stopping,
                                    base_args=base_args, fold=args.fold
                                )
                            except RuntimeError as e:
                                print(f"[warn] failed {ds} {variant} {kind} seed={seed} d={d} L={L}: {e}")


# ------------------------ plot ------------------------

def load_runs(outdir: str):
    runs = []
    for f in glob.glob(os.path.join(outdir, "*.npz")):
        z = np.load(f, allow_pickle=True)
        curve = z["dirichlet_norm"].astype(np.float64)
        meta = json.loads(str(z["meta"]))
        runs.append((meta, curve))
    return runs

def aggregate(curves: List[np.ndarray], max_layers: int):
    X = np.full((len(curves), max_layers), np.nan, dtype=np.float64)
    for i, c in enumerate(curves):
        L = min(len(c), max_layers)
        X[i, :L] = c[:L]
    return np.nanmean(X, axis=0), np.nanstd(X, axis=0)

def plot_all(args):
    datasets = ["Chameleon", "Squirrel", "Pubmed", "Cora"]
    variants = ["Diag", "Bundle", "General"]
    max_layers = int(args.max_layers)

    runs = load_runs(args.outdir)
    grouped: Dict[Tuple[str,str,str], List[np.ndarray]] = {}
    for meta, curve in runs:
        key = (meta["variant"], meta["dataset"], meta["kind"])
        grouped.setdefault(key, []).append(curve)

    os.makedirs(args.figdir, exist_ok=True)

    for variant in variants:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes = axes.flatten()

        for ax, ds in zip(axes, datasets):
            x = np.arange(1, max_layers + 1)

            nsd = grouped.get((variant, ds, "NSD"), [])
            poly = grouped.get((variant, ds, "PolyNSD"), [])

            if nsd:
                m, s = aggregate(nsd, max_layers)
                ax.plot(x, m, label=f"NSD (n={len(nsd)})")
                ax.fill_between(x, m - s, m + s, alpha=0.2)
            if poly:
                m, s = aggregate(poly, max_layers)
                ax.plot(x, m, label=f"PolyNSD (n={len(poly)})")
                ax.fill_between(x, m - s, m + s, alpha=0.2)

            ax.set_title(ds)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Dirichlet energy (normalized)")
            ax.grid(True, alpha=0.3)
            ax.legend()

        fig.suptitle(f"{variant}: Dirichlet energy after diffusion/filter (mean ± std)")
        out = os.path.join(args.figdir, f"dirichlet_{variant}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print("Saved figure:", out)


# ------------------------ CLI ------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("collect")
    pc.add_argument("--outdir", type=str, default="visualizations/results_dirichlet")
    pc.add_argument("--datasets", nargs="+", default=["Chameleon","Squirrel","Pubmed","Cora"])
    pc.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    pc.add_argument("--ds", nargs="+", type=int, default=[2,3,4])
    pc.add_argument("--layers", nargs="+", type=int, default=[2,3,4])
    pc.add_argument("--fold", type=int, default=0)
    pc.add_argument("--epochs", type=int, default=400)
    pc.add_argument("--early_stopping", type=int, default=100)
    pc.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    pp = sub.add_parser("plot")
    pp.add_argument("--outdir", type=str, default="visualizations/results_dirichlet")
    pp.add_argument("--figdir", type=str, default="visualizations/figures/figs_dirichlet")
    pp.add_argument("--max_layers", type=int, default=4)

    args = p.parse_args()
    if args.cmd == "collect":
        collect_grid(args)
    else:
        plot_all(args)

if __name__ == "__main__":
    main()
