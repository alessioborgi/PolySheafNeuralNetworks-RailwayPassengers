#!/usr/bin/env python3
"""
spectral_band_ablation.py

Causal spectral interpretability experiment:
- Train NSD vs PolyNSD (Diag/Bundle/General) on {Chameleon,Squirrel,Pubmed,Cora}
- Capture learned sheaf Laplacian L (sparse) without touching model code
- Capture final hidden representation z (input to lin2)
- Compute k smallest eigenpairs of L
- Split into low/mid/high bands and evaluate:
    * band-only accuracy
    * remove-band accuracy drop

Two commands:
  collect: run grid and save per-run JSONL
  plot:    aggregate JSONL and make bar plots of accuracy drops

Example:
  python spectral_band_ablation.py collect --outdir band_results --device cuda:0
  python spectral_band_ablation.py plot --outdir band_results --figdir band_figs
"""

import argparse, json, os, random, glob
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

import torch_sparse

# repo root on path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# scipy for sparse eigensolver
try:
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
except Exception as e:
    sp = None
    eigsh = None


# ------------------------ helpers ------------------------

def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def acc_from_logits(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits[mask].max(1)[1]
    return float(pred.eq(y[mask]).float().mean().item())

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


# ------------------------ capture Laplacian + hidden rep without editing model ------------------------

class CaptureOnce:
    """Capture one (row,col,vals,m) from the first spmm call during a forward."""
    def __init__(self):
        self.row = None
        self.col = None
        self.vals = None
        self.m = None
        self._orig = None

    def attach(self):
        self._orig = torch_sparse.spmm

        def wrapped(*args):
            # supports both common signatures
            if len(args) == 5:
                idx, vals, m, n, mat = args
                row, col = idx
                out = self._orig(idx, vals, m, n, mat)
            elif len(args) == 6:
                row, col, m, n, vals, mat = args
                out = self._orig(row, col, m, n, vals, mat)
            else:
                return self._orig(*args)

            if self.row is None:
                self.row = row.detach().cpu()
                self.col = col.detach().cpu()
                self.vals = vals.detach().cpu()
                self.m = int(m)
            return out

        torch_sparse.spmm = wrapped

    def detach(self):
        if self._orig is not None:
            torch_sparse.spmm = self._orig
            self._orig = None


class Lin2InputHook:
    """Capture the input to model.lin2 (z shape: [N, hidden_dim])."""
    def __init__(self):
        self.z = None
        self.h = None

    def attach(self, lin2_module: torch.nn.Module):
        def pre_hook(mod, inp):
            # inp is tuple
            self.z = inp[0].detach()
        self.h = lin2_module.register_forward_pre_hook(pre_hook)

    def detach(self):
        if self.h is not None:
            self.h.remove()
            self.h = None


# ------------------------ train ------------------------

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
    return acc_from_logits(logits, data.y, data.val_mask)

def train_with_early_stopping(model, data, lr: float, weight_decay: float, epochs: int, early_stopping: int):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
    return model


# ------------------------ spectral band ops ------------------------

def scipy_laplacian_from_edges(row: torch.Tensor, col: torch.Tensor, vals: torch.Tensor, n: int):
    assert sp is not None
    r = row.numpy()
    c = col.numpy()
    v = vals.numpy()
    L = sp.coo_matrix((v, (r, c)), shape=(n, n)).tocsr()
    return L

def eig_basis(L_csr, k: int, which: str = "SM"):
    """
    which=SM: smallest magnitude (for Laplacians -> near 0 modes)
    returns evals (k,), evecs (n,k)
    """
    assert eigsh is not None
    k = min(k, L_csr.shape[0] - 2) if L_csr.shape[0] > 2 else 1
    evals, evecs = eigsh(L_csr, k=k, which=which)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]
    return evals, evecs

def split_bands(evals: np.ndarray, tau1_frac: float, tau2_frac: float):
    lam_max = float(np.max(evals)) if len(evals) else 1.0
    tau1 = tau1_frac * lam_max
    tau2 = tau2_frac * lam_max
    low = np.where(evals <= tau1)[0]
    mid = np.where((evals > tau1) & (evals <= tau2))[0]
    high = np.where(evals > tau2)[0]
    return low, mid, high, tau1, tau2, lam_max

@torch.no_grad()
def project_reconstruct(U: torch.Tensor, x: torch.Tensor, idxs: np.ndarray):
    """
    U: (n,k) float tensor
    x: (n,C)
    idxs: indices in [0,k)
    return x_band: (n,C)
    """
    if idxs.size == 0:
        return torch.zeros_like(x)
    Ub = U[:, idxs]                 # (n, kb)
    coeff = Ub.T @ x                # (kb, C)
    return Ub @ coeff               # (n, C)


# ------------------------ one run ------------------------

def run_one(
    *,
    dataset_name: str,
    variant: str,
    kind: str,        # NSD or PolyNSD
    seed: int,
    d: int,
    layers: int,
    device: torch.device,
    base_args: Dict[str, Any],
    epochs: int,
    early_stopping: int,
    k_eigs: int,
    tau1_frac: float,
    tau2_frac: float,
    fold: int = 0,
):
    from utils.heterophilic import get_dataset, get_fixed_splits

    if sp is None or eigsh is None:
        raise RuntimeError("scipy is required (scipy.sparse + scipy.sparse.linalg.eigsh).")

    seed_all(seed)

    dataset = get_dataset(dataset_name)
    data = get_fixed_splits(dataset[0], dataset_name, fold).to(device)

    args = dict(base_args)
    args["d"] = int(d)
    args["layers"] = int(layers)
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

    model = train_with_early_stopping(
        model, data,
        lr=float(args["lr"]),
        weight_decay=float(args["weight_decay"]),
        epochs=epochs,
        early_stopping=early_stopping,
    )

    # Capture lin2 input z and one Laplacian L via one forward pass
    capL = CaptureOnce()
    hook = Lin2InputHook()
    hook.attach(model.lin2)
    capL.attach()

    model.eval()
    with torch.no_grad():
        _ = model(data.x)

    capL.detach()
    hook.detach()

    if hook.z is None or capL.row is None:
        raise RuntimeError("Failed to capture z or Laplacian. (Did forward reach spmm and lin2?)")

    z = hook.z                               # (N, hidden_dim)
    N = z.size(0)

    # Recover (Nd, h) view
    hidden_dim = z.size(1)
    if hidden_dim % d != 0:
        raise RuntimeError(f"lin2 input dim {hidden_dim} not divisible by d={d}. Can't reshape to (N*d, h).")
    h = hidden_dim // d
    x = z.view(N * d, h).detach().cpu()      # (Nd, h)

    # Build scipy Laplacian (Nd x Nd)
    Nd = int(capL.m)
    if Nd != N * d:
        # This can happen if model uses final_d != d; adjust if needed
        Nd = N * d  # fallback, but better to check your model.final_d convention
    L_csr = scipy_laplacian_from_edges(capL.row, capL.col, capL.vals, Nd)

    # eigenbasis
    evals, evecs = eig_basis(L_csr, k=k_eigs, which="SM")
    U = torch.from_numpy(evecs).float()      # (Nd, k)
    low, mid, high, tau1, tau2, lam_max = split_bands(evals, tau1_frac, tau2_frac)

    # baseline logits from full z using model.lin2
    with torch.no_grad():
        logits_full = model.lin2(z)  # (N, num_classes)

    base_test = acc_from_logits(logits_full, data.y, data.test_mask)
    base_val  = acc_from_logits(logits_full, data.y, data.val_mask)

    # band-only reconstructions in Nd space
    x_low  = project_reconstruct(U, x, low)
    x_mid  = project_reconstruct(U, x, mid)
    x_high = project_reconstruct(U, x, high)

    # remove-band reconstructions
    x_nolow  = x - x_low
    x_nomid  = x - x_mid
    x_nohigh = x - x_high

    def logits_from_x(x_nd: torch.Tensor) -> torch.Tensor:
        # x_nd: (Nd, h) -> z': (N, d*h) -> lin2
        z_prime = x_nd.view(N, d * h).to(device)
        return model.lin2(z_prime)

    with torch.no_grad():
        logits_low   = logits_from_x(x_low.to(device))
        logits_mid   = logits_from_x(x_mid.to(device))
        logits_high  = logits_from_x(x_high.to(device))
        logits_nolow  = logits_from_x(x_nolow.to(device))
        logits_nomid  = logits_from_x(x_nomid.to(device))
        logits_nohigh = logits_from_x(x_nohigh.to(device))

    def test_acc(logits): return acc_from_logits(logits, data.y, data.test_mask)

    out = dict(
        dataset=dataset_name, variant=variant, kind=kind,
        seed=seed, d=d, layers=layers,
        k_eigs=int(len(evals)),
        tau1=float(tau1), tau2=float(tau2), lam_max=float(lam_max),
        base_val=float(base_val), base_test=float(base_test),

        test_low_only=float(test_acc(logits_low)),
        test_mid_only=float(test_acc(logits_mid)),
        test_high_only=float(test_acc(logits_high)),

        test_remove_low=float(test_acc(logits_nolow)),
        test_remove_mid=float(test_acc(logits_nomid)),
        test_remove_high=float(test_acc(logits_nohigh)),

        drop_remove_low=float(base_test - test_acc(logits_nolow)),
        drop_remove_mid=float(base_test - test_acc(logits_nomid)),
        drop_remove_high=float(base_test - test_acc(logits_nohigh)),
    )
    return out


# ------------------------ collect/plot ------------------------

def collect(args):
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

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, "band_ablation.jsonl")

    with open(outpath, "a", encoding="utf-8") as f:
        for ds in args.datasets:
            for variant in variants:
                for kind in kinds:
                    for seed in args.seeds:
                        for d in args.ds:
                            for L in args.layers:
                                try:
                                    res = run_one(
                                        dataset_name=ds,
                                        variant=variant,
                                        kind=kind,
                                        seed=int(seed),
                                        d=int(d),
                                        layers=int(L),
                                        device=device,
                                        base_args=base_args,
                                        epochs=int(args.epochs),
                                        early_stopping=int(args.early_stopping),
                                        k_eigs=int(args.k_eigs),
                                        tau1_frac=float(args.tau1_frac),
                                        tau2_frac=float(args.tau2_frac),
                                        fold=int(args.fold),
                                    )
                                    f.write(json.dumps(res) + "\n")
                                    f.flush()
                                    print("OK:", ds, variant, kind, "seed", seed, "d", d, "L", L,
                                          "drop_high", f"{res['drop_remove_high']:.4f}")
                                except RuntimeError as e:
                                    print("[warn] failed:", ds, variant, kind, "seed", seed, "d", d, "L", L, "|", e)

    print("Saved JSONL:", outpath)


def plot(args):
    import matplotlib.pyplot as plt

    jsonl = os.path.join(args.outdir, "band_ablation.jsonl")
    if not os.path.exists(jsonl):
        raise RuntimeError(f"Missing {jsonl}. Run collect first.")

    rows = []
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # group by (variant,dataset,kind) and aggregate drops
    def group_key(r): return (r["variant"], r["dataset"], r["kind"])
    groups: Dict[Tuple[str,str,str], List[dict]] = {}
    for r in rows:
        groups.setdefault(group_key(r), []).append(r)

    os.makedirs(args.figdir, exist_ok=True)
    datasets = ["Chameleon", "Squirrel", "Pubmed", "Cora"]
    variants = ["Diag", "Bundle", "General"]

    # plot drop_remove_high as primary evidence
    for variant in variants:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes = axes.flatten()
        for ax, ds in zip(axes, datasets):
            for kind in ["NSD", "PolyNSD"]:
                g = groups.get((variant, ds, kind), [])
                if not g:
                    continue
                vals = np.array([x["drop_remove_high"] for x in g], dtype=np.float64)
                m, s = float(np.mean(vals)), float(np.std(vals))
                ax.bar(kind, m, yerr=s, capsize=4)
            ax.set_title(ds)
            ax.set_ylabel("Accuracy drop removing HIGH band")
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"{variant}: causal dependence on HIGH frequencies (mean ± std)")
        out = os.path.join(args.figdir, f"drop_remove_high_{variant}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print("Saved:", out)


# ------------------------ CLI ------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("collect")
    pc.add_argument("--outdir", type=str, required=True)
    pc.add_argument("--figdir", type=str, default="band_figs")  # unused in collect
    pc.add_argument("--datasets", nargs="+", default=["Chameleon","Squirrel","Pubmed","Cora"])
    pc.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    pc.add_argument("--ds", nargs="+", type=int, default=[2,3,4])
    pc.add_argument("--layers", nargs="+", type=int, default=[2,3,4])
    pc.add_argument("--fold", type=int, default=0)
    pc.add_argument("--epochs", type=int, default=400)
    pc.add_argument("--early_stopping", type=int, default=100)
    pc.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    pc.add_argument("--k_eigs", type=int, default=64)
    pc.add_argument("--tau1_frac", type=float, default=0.25)
    pc.add_argument("--tau2_frac", type=float, default=0.75)

    pp = sub.add_parser("plot")
    pp.add_argument("--outdir", type=str, required=True)
    pp.add_argument("--figdir", type=str, default="band_figs")

    args = p.parse_args()

    if args.cmd == "collect":
        collect(args)
    else:
        plot(args)

if __name__ == "__main__":
    main()
