# -*- coding: utf-8 -*-
"""
ChebGNN / Polynomial Spectral Filter GNN Benchmark Runner

Fixes:
- W&B sweep/agent mode: ONE run == ONE (dataset, config).
- Final aggregate metrics are SAVED with the keys you want:
    val_mean, val_std, test_mean, test_std   (PERCENT)
  plus aliases:
    val_acc, val_acc_std, test_acc, test_acc_std (PERCENT)
- No wandb.run.group assignment (read-only in recent wandb); no locked-config updates.
- Per-epoch curves logged only for split0/seed0 to keep charts readable.
"""

import os
import json
import csv
import time
import argparse
import itertools
import random
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Robust scatter_add import (prefer torch_scatter, fallback to PyG utils) ----
try:
    from torch_scatter import scatter_add  # type: ignore
except Exception:
    scatter_add = None
    try:
        from torch_geometric.utils import scatter  # type: ignore
    except Exception:
        scatter = None

from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Actor
from torch_geometric.transforms import NormalizeFeatures


# --------------------------
# Reproducibility utilities
# --------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fold_seed(base_seed: int, split_id: int, run_seed: int) -> int:
    return int(base_seed + 1000 * split_id + 17 * run_seed)


# --------------------------
# Sparse ops
# --------------------------
def _scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int) -> torch.Tensor:
    if scatter_add is not None:
        return scatter_add(src, index, dim=dim, dim_size=dim_size)
    if "scatter" in globals() and scatter is not None:
        return scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")
    raise RuntimeError("No scatter_add available. Install torch_scatter or update torch_geometric.")


def normalized_adj(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
    add_self_loops: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    row, col = edge_index
    if edge_weight is None:
        edge_weight = torch.ones(row.size(0), device=row.device)

    if add_self_loops:
        loop = torch.arange(num_nodes, device=row.device)
        loop_index = torch.stack([loop, loop], dim=0)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        edge_weight = torch.cat([edge_weight, torch.ones(num_nodes, device=row.device)], dim=0)
        row, col = edge_index

    deg = _scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)
    norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, norm


def spmm(edge_index: torch.Tensor, edge_weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    msg = x[col] * edge_weight.unsqueeze(-1)
    out = _scatter_add(msg, row, dim=0, dim_size=x.size(0))
    return out


# --------------------------
# Polynomial spectral layer
# --------------------------
class PolySpectralConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        convex_mixture: bool = True,
        use_gate: bool = True,
        highpass: bool = True,
    ):
        super().__init__()
        self.K = int(K)
        self.convex_mixture = bool(convex_mixture)
        self.use_gate = bool(use_gate)
        self.highpass = bool(highpass)

        self.lin_in = nn.Linear(in_channels, out_channels, bias=False)
        self.theta = nn.Parameter(torch.zeros(self.K + 1))
        self.lin_out = nn.Linear(out_channels, out_channels, bias=True)

        if self.use_gate:
            self.gate = nn.Linear(2 * out_channels, out_channels)

        if self.highpass:
            self.hp = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        lambda_max: float = 2.0,
    ) -> torch.Tensor:
        N = x.size(0)
        h0 = self.lin_in(x)

        edge_index, norm = normalized_adj(edge_index, N, edge_weight=edge_weight, add_self_loops=False)

        a = (2.0 / lambda_max) - 1.0
        b = 2.0 / lambda_max

        def Ltilde_mv(v: torch.Tensor) -> torch.Tensor:
            Sv = spmm(edge_index, norm, v)
            return a * v - b * Sv

        outs = [h0]
        if self.K >= 1:
            outs.append(Ltilde_mv(h0))
        for _ in range(1, self.K):
            outs.append(2.0 * Ltilde_mv(outs[-1]) - outs[-2])

        stack = torch.stack(outs, dim=0)  # [K+1, N, F]
        w = torch.softmax(self.theta, dim=0) if self.convex_mixture else self.theta
        h = (w.view(-1, 1, 1) * stack).sum(dim=0)

        if self.highpass:
            Sh0 = spmm(edge_index, norm, h0)
            Lh0 = h0 - Sh0
            h = h + torch.tanh(self.hp) * Lh0

        h = self.lin_out(h)

        if self.use_gate:
            g = torch.sigmoid(self.gate(torch.cat([h, h0], dim=-1)))
            h = g * h + (1.0 - g) * h0

        return h


class PolySpectralGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        K: int,
        layers: int,
        dropout: float,
        convex_mixture: bool,
        use_gate: bool,
        highpass: bool,
    ):
        super().__init__()
        assert layers >= 1
        self.dropout = float(dropout)

        self.convs = nn.ModuleList()
        self.convs.append(
            PolySpectralConv(
                in_channels, hidden_channels, K,
                convex_mixture=convex_mixture,
                use_gate=use_gate,
                highpass=highpass,
            )
        )
        for _ in range(layers - 1):
            self.convs.append(
                PolySpectralConv(
                    hidden_channels, hidden_channels, K,
                    convex_mixture=convex_mixture,
                    use_gate=use_gate,
                    highpass=highpass,
                )
            )

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


# --------------------------
# Dataset loading
# --------------------------
DEFAULT_BENCHMARK = ["Texas", "Wisconsin", "Cornell", "Film", "Squirrel", "Chameleon", "Cora", "CiteSeer", "PubMed"]


def load_dataset(name: str, root: str):
    name = name.strip()
    tfm = NormalizeFeatures()

    if name in ["Cora", "CiteSeer", "PubMed"]:
        ds = Planetoid(root=os.path.join(root, "Planetoid", name), name=name, transform=tfm)
        return ds[0], ds.num_features, ds.num_classes

    if name in ["Texas", "Wisconsin", "Cornell"]:
        ds = WebKB(root=os.path.join(root, "WebKB", name), name=name, transform=tfm)
        return ds[0], ds.num_features, ds.num_classes

    if name in ["Chameleon", "Squirrel"]:
        ds = WikipediaNetwork(root=os.path.join(root, "WikipediaNetwork", name), name=name, transform=tfm)
        return ds[0], ds.num_features, ds.num_classes

    if name in ["Film"]:
        ds = Actor(root=os.path.join(root, "Actor"), transform=tfm)
        return ds[0], ds.num_features, ds.num_classes

    raise ValueError(f"Unknown dataset '{name}'.")


def iter_splits(data) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not hasattr(data, "train_mask") or data.train_mask is None:
        raise RuntimeError("Dataset has no train/val/test masks.")

    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    if train_mask.dim() == 1:
        return [(train_mask, val_mask, test_mask)]

    S = train_mask.size(1)
    return [(train_mask[:, s], val_mask[:, s], test_mask[:, s]) for s in range(S)]


def acc_from_logits(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits[mask].argmax(dim=-1)
    correct = (pred == y[mask]).sum().item()
    total = int(mask.sum().item())
    return float(correct) / max(total, 1)


# --------------------------
# Config
# --------------------------
@dataclass
class Config:
    dataset: str = "Texas"
    base_seed: int = 0

    K: int = 10
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.5
    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 1000
    patience: int = 200
    convex_mixture: bool = True
    use_gate: bool = True
    highpass: bool = True


def cfg_from_sources(wb: Dict[str, Any], args) -> Config:
    # IMPORTANT: do NOT wandb.config.update() in sweeps (locked); just READ
    return Config(
        dataset=str(wb.get("dataset", args.dataset or (args.datasets[0] if args.datasets else "Texas"))),
        base_seed=int(wb.get("base_seed", args.base_seed)),
        K=int(wb.get("K", args.K)),
        hidden=int(wb.get("hidden", args.hidden)),
        layers=int(wb.get("layers", args.layers)),
        dropout=float(wb.get("dropout", args.dropout)),
        lr=float(wb.get("lr", args.lr)),
        weight_decay=float(wb.get("weight_decay", args.weight_decay)),
        epochs=int(wb.get("epochs", args.epochs)),
        patience=int(wb.get("patience", args.patience)),
        convex_mixture=bool(wb.get("convex_mixture", args.convex_mixture)),
        use_gate=bool(wb.get("use_gate", args.use_gate)),
        highpass=bool(wb.get("highpass", args.highpass)),
    )


def mean_std(xs: List[float], ddof: int = 0) -> Tuple[float, float]:
    # ddof=0 to match typical "population std" reporting in many baselines
    if len(xs) == 0:
        return 0.0, 0.0
    m = float(np.mean(xs))
    s = float(np.std(xs, ddof=ddof)) if len(xs) > 1 else 0.0
    return m, s


def short_run_name(cfg: Config) -> str:
    return (f"{cfg.dataset}"
            f"-K{cfg.K}"
            f"-h{cfg.hidden}"
            f"-L{cfg.layers}"
            f"-do{cfg.dropout}"
            f"-lr{cfg.lr}"
            f"-wd{cfg.weight_decay}"
            f"-gate{int(cfg.use_gate)}"
            f"-hp{int(cfg.highpass)}")


# --------------------------
# Train one (split, seed)
# --------------------------
def train_one_split(
    data,
    split_masks,
    cfg: Config,
    device: torch.device,
    seed: int,
    split_id: int,
    seed_idx: int,
    wandb_module=None,
) -> Dict[str, float]:
    set_seed(seed)

    x = data.x.to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)
    train_mask, val_mask, test_mask = [m.to(device) for m in split_masks]

    model = PolySpectralGNN(
        in_channels=x.size(-1),
        hidden_channels=cfg.hidden,
        out_channels=int(y.max().item() + 1),
        K=cfg.K,
        layers=cfg.layers,
        dropout=cfg.dropout,
        convex_mixture=cfg.convex_mixture,
        use_gate=cfg.use_gate,
        highpass=cfg.highpass,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = -1.0
    best_test = -1.0
    best_epoch = 0
    bad = 0

    log_curves = (wandb_module is not None) and (split_id == 0) and (seed_idx == 0)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()

        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index)
            train_acc = acc_from_logits(logits, y, train_mask)
            val_acc = acc_from_logits(logits, y, val_mask)
            test_acc = acc_from_logits(logits, y, test_mask)
            train_loss = float(F.cross_entropy(logits[train_mask], y[train_mask]).item())
            val_loss = float(F.cross_entropy(logits[val_mask], y[val_mask]).item())
            dt = time.perf_counter() - t0

        if log_curves:
            # Curves are for sanity; final metrics are logged once at the end.
            wandb_module.log({
                "epoch": int(epoch),
                "curve/train_acc": float(train_acc) * 100.0,
                "curve/val_acc": float(val_acc) * 100.0,
                "curve/train_loss": float(train_loss),
                "curve/val_loss": float(val_loss),
                "curve/time_epoch_s": float(dt),
            }, step=int(epoch))

        if val_acc > best_val + 1e-12:
            best_val = float(val_acc)
            best_test = float(test_acc)
            best_epoch = int(epoch)
            bad = 0
        else:
            bad += 1

        if bad >= cfg.patience:
            break

    return {"best_val": best_val, "best_test": best_test, "best_epoch": float(best_epoch)}


# --------------------------
# Run ONE (dataset, config) experiment
# --------------------------
def run_one_dataset_one_config(
    dataset_name: str,
    data_root: str,
    cfg: Config,
    device: torch.device,
    seeds: List[int],
    base_seed: int,
    wandb_module=None,
) -> Dict[str, Any]:
    data, num_features, num_classes = load_dataset(dataset_name, data_root)
    splits = iter_splits(data)

    # store per-split seed-averaged bests
    split_val_means: List[float] = []
    split_test_means: List[float] = []
    split_summaries: List[Dict[str, Any]] = []

    for split_id, masks in enumerate(splits):
        vals: List[float] = []
        tests: List[float] = []
        epochs: List[float] = []

        for seed_idx, s in enumerate(seeds):
            seed = fold_seed(base_seed, split_id, s)
            out = train_one_split(
                data=data,
                split_masks=masks,
                cfg=cfg,
                device=device,
                seed=seed,
                split_id=split_id,
                seed_idx=seed_idx,
                wandb_module=wandb_module,
            )
            vals.append(out["best_val"])
            tests.append(out["best_test"])
            epochs.append(out["best_epoch"])

        v_m, v_s = mean_std(vals, ddof=0)
        t_m, t_s = mean_std(tests, ddof=0)
        e_m, e_s = mean_std(epochs, ddof=0)

        split_val_means.append(v_m)
        split_test_means.append(t_m)

        split_summaries.append({
            "split": int(split_id),
            "split_val_mean": float(v_m) * 100.0,
            "split_val_std": float(v_s) * 100.0,
            "split_test_mean": float(t_m) * 100.0,
            "split_test_std": float(t_s) * 100.0,
            "best_epoch_mean": float(e_m),
            "best_epoch_std": float(e_s),
        })

        if wandb_module is not None:
            # log per-split results without polluting run summary with a "split" key
            wandb_module.log({
                f"split{split_id}/val_mean": float(v_m) * 100.0,
                f"split{split_id}/val_std": float(v_s) * 100.0,
                f"split{split_id}/test_mean": float(t_m) * 100.0,
                f"split{split_id}/test_std": float(t_s) * 100.0,
                f"split{split_id}/best_epoch_mean": float(e_m),
            })

    # FINAL AGGREGATION: mean/std across splits (after averaging seeds within each split)
    val_mean, val_std = mean_std(split_val_means, ddof=0)
    test_mean, test_std = mean_std(split_test_means, ddof=0)

    # convert to percent
    val_mean_p = float(val_mean) * 100.0
    val_std_p = float(val_std) * 100.0
    test_mean_p = float(test_mean) * 100.0
    test_std_p = float(test_std) * 100.0

    summary = {
        "dataset": dataset_name,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_features": int(num_features),
        "num_classes": int(num_classes),
        "n_splits": int(len(splits)),
        "n_seeds": int(len(seeds)),
        "n_runs": int(len(splits) * len(seeds)),
        # keys YOU want (percent)
        "val_mean": val_mean_p,
        "val_std": val_std_p,
        "test_mean": test_mean_p,
        "test_std": test_std_p,
        # aliases (percent)
        "val_acc": val_mean_p,
        "val_acc_std": val_std_p,
        "test_acc": test_mean_p,
        "test_acc_std": test_std_p,
        "splits": split_summaries,
    }

    if wandb_module is not None:
        # Log ONCE at end with the exact names used by your sweep metric / tables
        wandb_module.log({
            "dataset_name": dataset_name,
            "val_mean": val_mean_p,
            "val_std": val_std_p,
            "test_mean": test_mean_p,
            "test_std": test_std_p,
            "val_acc": val_mean_p,
            "val_acc_std": val_std_p,
            "test_acc": test_mean_p,
            "test_acc_std": test_std_p,
        })

        # Force into summary so you ALWAYS see them in the Run summary
        try:
            wandb_module.run.summary["dataset"] = dataset_name
            wandb_module.run.summary["val_mean"] = val_mean_p
            wandb_module.run.summary["val_std"] = val_std_p
            wandb_module.run.summary["test_mean"] = test_mean_p
            wandb_module.run.summary["test_std"] = test_std_p
            wandb_module.run.summary["val_acc"] = val_mean_p
            wandb_module.run.summary["val_acc_std"] = val_std_p
            wandb_module.run.summary["test_acc"] = test_mean_p
            wandb_module.run.summary["test_acc_std"] = test_std_p
        except Exception:
            pass

    return summary


def append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def write_csv(path: str, rows: List[Dict[str, Any]]):
    if len(rows) == 0:
        return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --------------------------
# Main
# --------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./runs")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Dataset selection
    p.add_argument("--dataset", type=str, default=None, help="Single dataset (recommended for sweeps/agents).")
    p.add_argument("--datasets", nargs="*", default=DEFAULT_BENCHMARK, help="Used for local non-agent runs.")

    # Hyperparams
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="*", default=[0])

    p.add_argument("--convex_mixture", action="store_true", default=True)
    p.add_argument("--no_convex_mixture", action="store_false", dest="convex_mixture")
    p.add_argument("--use_gate", action="store_true", default=True)
    p.add_argument("--no_gate", action="store_false", dest="use_gate")
    p.add_argument("--highpass", action="store_true", default=True)
    p.add_argument("--no_highpass", action="store_false", dest="highpass")

    # W&B
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--project", type=str, default="ChebGNN")  # ignored in sweep agent mode
    p.add_argument("--entity", type=str, default=None)
    p.add_argument("--wandb_agent", action="store_true")

    # Optional: save agent results to local jsonl too
    p.add_argument("--agent_results_jsonl", type=str, default=None,
                   help="If set, append ONE JSON line per agent run with final metrics.")

    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    wandb_mod = None
    if args.wandb or args.wandb_agent:
        import wandb  # type: ignore
        wandb_mod = wandb

    # --------------------------
    # Agent mode: ONE run == ONE dataset+config
    # --------------------------
    if args.wandb_agent:
        # IMPORTANT: don't pass project here; the sweep controls it
        run = wandb_mod.init(entity=args.entity)

        # Make charts use epoch as x-axis if we log curves
        try:
            wandb_mod.define_metric("epoch")
            wandb_mod.define_metric("curve/*", step_metric="epoch")
        except Exception:
            pass

        wb = dict(wandb_mod.config)
        cfg = cfg_from_sources(wb, args)

        # readable name
        try:
            wandb_mod.run.name = short_run_name(cfg)
        except Exception:
            pass

        summary = run_one_dataset_one_config(
            dataset_name=cfg.dataset,
            data_root=args.data_root,
            cfg=cfg,
            device=device,
            seeds=args.seeds,
            base_seed=cfg.base_seed,
            wandb_module=wandb_mod,
        )

        print(
            f"\n{cfg.dataset} | val={summary['val_mean']:.2f}±{summary['val_std']:.2f} "
            f"| test={summary['test_mean']:.2f}±{summary['test_std']:.2f} | runs={summary['n_runs']}"
        )

        if args.agent_results_jsonl is not None:
            append_jsonl(args.agent_results_jsonl, {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "run_name": getattr(wandb_mod.run, "name", None),
                "config": asdict(cfg),
                "metrics": {k: summary[k] for k in ["val_mean", "val_std", "test_mean", "test_std", "n_runs"]},
            })

        wandb_mod.finish()
        return

    # --------------------------
    # Local mode (optional W&B): you can loop datasets/configs and also write CSV/JSON
    # --------------------------
    datasets = [args.dataset] if args.dataset is not None else list(args.datasets)

    rows: List[Dict[str, Any]] = []
    best_json: Dict[str, Any] = {}

    for dname in datasets:
        cfg = Config(
            dataset=dname,
            base_seed=args.base_seed,
            K=args.K,
            hidden=args.hidden,
            layers=args.layers,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            convex_mixture=args.convex_mixture,
            use_gate=args.use_gate,
            highpass=args.highpass,
        )

        if args.wandb:
            run = wandb_mod.init(
                entity=args.entity,
                project=args.project,
                config=asdict(cfg),
                name=short_run_name(cfg),
            )

        summary = run_one_dataset_one_config(
            dataset_name=dname,
            data_root=args.data_root,
            cfg=cfg,
            device=device,
            seeds=args.seeds,
            base_seed=cfg.base_seed,
            wandb_module=wandb_mod if args.wandb else None,
        )

        if args.wandb:
            wandb_mod.finish()

        rows.append({
            "dataset": dname,
            "val_mean": summary["val_mean"],
            "val_std": summary["val_std"],
            "test_mean": summary["test_mean"],
            "test_std": summary["test_std"],
            "n_runs": summary["n_runs"],
            "cfg": json.dumps(asdict(cfg)),
        })

        best_json[dname] = {
            "best_cfg": asdict(cfg),
            "metrics": {
                "val_mean": summary["val_mean"],
                "val_std": summary["val_std"],
                "test_mean": summary["test_mean"],
                "test_std": summary["test_std"],
                "n_runs": summary["n_runs"],
            }
        }

        print(
            f"[{dname}] val={summary['val_mean']:.2f}±{summary['val_std']:.2f} "
            f"test={summary['test_mean']:.2f}±{summary['test_std']:.2f}"
        )

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out_dir, f"chebgnn_{ts}.csv")
    json_path = os.path.join(args.out_dir, f"chebgnn_{ts}.json")
    write_csv(csv_path, rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best_json, f, indent=2)

    print(f"\nWrote CSV:  {csv_path}")
    print(f"Wrote JSON: {json_path}")


if __name__ == "__main__":
    main()