############################ POLYSD VS NSD #############################



# ! /usr/bin/env python
# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import random
import torch
import torch.nn.functional as F
import git
import numpy as np
import wandb
from tqdm import tqdm
import torch_geometric
from torch_geometric.data import Data

# This is required here by wandb sweeps.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exp.parser import get_parser
from models.positional_encodings import append_top_k_evectors
from models.cont_models import (
    DiagSheafDiffusion, BundleSheafDiffusion, GeneralSheafDiffusion
)
from models.disc_models import (
    DiscreteDiagSheafDiffusion, DiscreteBundleSheafDiffusion, DiscreteGeneralSheafDiffusion,
    DiscreteDiagSheafDiffusionChebyshev, DiscreteBundleSheafDiffusionChebyshev, DiscreteGeneralSheafDiffusionChebyshev,
    DiscreteJointSheafDiffusionParams, DiscreteJointSheafVanillaDiffusion, DiscreteVanillaDiffusion,
    DiscreteVanillaDiffusionAlt, DiscreteJointSheafDiffusionParamsAlt,
    EquivariantDiscreteDiagSheafDiffusion, EquivariantDiscreteBundleSheafDiffusion, EquivariantDiscreteGeneralSheafDiffusion,
    EquivariantDiscreteDiagSheafDiffusionChebyshev
)
from utils.heterophilic import get_dataset, get_synthetic_dataset, get_fixed_splits


# ----------------------------- utilities -----------------------------

def precompute_sheaf_mappings(data, d, args):
    """Optional initialization for joint sheaf params models."""
    diff_model = DiscreteJointSheafVanillaDiffusion(data.edge_index, args)
    U, S, V = torch.pca_lowrank(data.x)
    x_d = torch.matmul(data.x, V[:, 0:d])
    sheaf_init = diff_model(x_d)
    return sheaf_init


def reset_wandb_env():
    """Drop all WANDB_* env vars except the essentials, helpful before sweeps/agents."""
    exclude = {"WANDB_PROJECT", "WANDB_ENTITY", "WANDB_API_KEY"}
    for k in list(os.environ.keys()):
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def _aget(args, key, default=None):
    """Safe getter that works for dict, argparse.Namespace, or wandb.Config."""
    try:
        return args[key]
    except Exception:
        return getattr(args, key, default)


def _is_cheby_model(args) -> bool:
    m = str(_aget(args, "model", "")).lower()
    return "chebyshev" in m


def _parse_cheby_Ks(args):
    """
    Returns (sumK, Ks_list) used to estimate #matvecs per forward for PolySD.
    Accepts:
      - chebyshev_layers_K (int)
      - chebyshev_layers_Ks (list-like or "[...]" string)
    """
    L = int(_aget(args, "layers", 1) or 1)
    K_default = int(_aget(args, "chebyshev_layers_K", 1) or 1)
    Ks_raw = _aget(args, "chebyshev_layers_Ks", None)

    if Ks_raw is None:
        Ks_list = [K_default] * L
    else:
        if isinstance(Ks_raw, (list, tuple)):
            Ks_list = [max(1, int(k)) for k in Ks_raw]
        else:
            try:
                s = str(Ks_raw).strip()
                if s.startswith("[") and s.endswith("]"):
                    Ks_list = [max(1, int(x.strip())) for x in s[1:-1].split(",") if x.strip()]
                else:
                    Ks_list = [max(1, int(s))] * L
            except Exception:
                Ks_list = [K_default] * L

        # Adjust to length L
        if len(Ks_list) < L:
            Ks_list = Ks_list + [Ks_list[-1]] * (L - len(Ks_list))
        elif len(Ks_list) > L:
            Ks_list = Ks_list[:L]

    sumK = int(sum(Ks_list))
    return sumK, Ks_list


# ----------------------------- train / eval -----------------------------

def train_one_epoch(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x)[data.train_mask]
    loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.detach()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    logits = model(data.x)
    accs, losses = [], []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].argmax(dim=1)
        acc = pred.eq(data.y[mask]).sum().item() / int(mask.sum().item())
        loss = F.nll_loss(logits[mask], data.y[mask])
        accs.append(acc)
        losses.append(loss.detach().cpu())
    return accs, losses  # ([train_acc,val_acc,test_acc], [train_loss,val_loss,test_loss])


def generate_splits(data, train=0.6, val=0.2):
    """Unused in your current pipeline but kept for completeness."""
    n = data.x.shape[0]
    perm = torch.randperm(n)
    ti = int(train * n)
    vi = ti + int(val * n)
    train_mask = np.full(n, False); train_mask[perm[:ti]] = True
    val_mask = np.full(n, False);   val_mask[perm[ti:vi]] = True
    test_mask = np.full(n, False);  test_mask[perm[vi:]] = True
    return Data(x=data.x, edge_index=data.edge_index, y=data.y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


# ----------------------------- core runner -----------------------------

def run_exp(args, dataset, model_cls, fold: int):
    """
    One fold of training/eval with:
      - param counting + W&B logging (once per run)
      - PolySD vs NSD matvec budget logging
      - early stopping on val-{acc|loss}
      - Laplacian / epsilons debug prints for discrete models
    """
    # ---- data / split / device
    data = dataset[0]
    data = get_fixed_splits(data, _aget(args, 'dataset'), fold)
    device = _aget(args, 'device')
    data = data.to(device)

    # ---- build model (optionally with sheaf init)
    if model_cls == DiscreteJointSheafDiffusionParams:
        sheaf_init = precompute_sheaf_mappings(data, _aget(args, 'd'), args) if bool(_aget(args, 'sheaf_init', False)) else []
        model = model_cls(data.edge_index, args, sheaf_init=sheaf_init)
    else:
        model = model_cls(data.edge_index, args)
    model = model.to(device)

    # ---- grouped params & optimizer
    sheaf_learner_params, other_params = model.grouped_parameters()

    # ---- NEW: parameter counting + budget logging
    num_total = sum(p.numel() for p in model.parameters() if getattr(p, "requires_grad", True))
    num_sheaf = sum(p.numel() for p in sheaf_learner_params if getattr(p, "requires_grad", True))
    num_other = sum(p.numel() for p in other_params if getattr(p, "requires_grad", True))
    try:
        assert num_sheaf + num_other == num_total
    except AssertionError:
        pass

    L = int(_aget(args, "layers", 1) or 1)
    if _is_cheby_model(args):
        sumK, Ks_list = _parse_cheby_Ks(args)
        approx_matvecs_per_forward = int(sumK)   # ~K matvecs per layer (Chebyshev recursion)
        cheby_K_logged = Ks_list if len(Ks_list) > 1 else Ks_list[0]
        budget_tag = "Cheby"
    else:
        approx_matvecs_per_forward = int(L)      # ~1 matvec per layer (NSD)
        cheby_K_logged = None
        budget_tag = "NSD"

    print(f"#params total={num_total:,} | sheaf={num_sheaf:,} | other={num_other:,}")
    print(f"Approx matvecs/forward ≈ {approx_matvecs_per_forward} ({budget_tag})")

    if fold == 0:
        try:
            wandb.log({
                "model/num_params": int(num_total),
                "model/num_params_sheaf": int(num_sheaf),
                "model/num_params_other": int(num_other),
                "model/layers": L,
                "model/hidden_channels": _aget(args, "hidden_channels"),
                "model/d": _aget(args, "d"),
                "model/final_d": _aget(args, "final_d"),
                "model/chebyshev_K": cheby_K_logged,
                "model/approx_matvecs_per_forward": approx_matvecs_per_forward,
                "data/graph_size": int(_aget(args, "graph_size", getattr(data, "num_nodes", data.x.size(0)))),
            })
        except Exception as e:
            print(f"[warn] wandb.log (param/budget) failed: {e}")

    # ---- optimizer
    maps_lr = _aget(args, 'maps_lr', None)
    base_lr = float(_aget(args, 'lr'))
    opt = torch.optim.Adam([
        {
            'params': sheaf_learner_params,
            'weight_decay': float(_aget(args, 'sheaf_decay', _aget(args, 'weight_decay', 0.0)) or 0.0),
            'lr': float(maps_lr if maps_lr is not None else base_lr),
        },
        {
            'params': other_params,
            'weight_decay': float(_aget(args, 'weight_decay', 0.0) or 0.0),
            'lr': float(base_lr),
        },
    ])

    # ---- training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    test_acc = 0.0
    best_epoch = 0
    bad_counter = 0
    early_stopping = int(_aget(args, 'early_stopping', 50) or 50)
    stop_strategy = str(_aget(args, 'stop_strategy', 'acc') or 'acc').lower()
    epochs = int(_aget(args, 'epochs', 200))

    for epoch in range(epochs):
        train_one_epoch(model, opt, data)

        (train_acc, val_acc, tmp_test_acc), (train_loss, val_loss, tmp_test_loss) = evaluate(model, data)

        # W&B epoch logs (fold 0 to keep tidy)
        if fold == 0:
            try:
                wandb.log({
                    f'fold{fold}/train_acc': train_acc,
                    f'fold{fold}/val_acc': val_acc,
                    f'fold{fold}/test_acc_snap': tmp_test_acc,
                    f'fold{fold}/train_loss': float(train_loss),
                    f'fold{fold}/val_loss': float(val_loss),
                    f'fold{fold}/test_loss_snap': float(tmp_test_loss),
                    'epoch': epoch,
                }, step=epoch)
            except Exception as e:
                print(f"[warn] wandb.log (epoch) failed: {e}")

        improved = (val_acc > best_val_acc) if (stop_strategy == 'acc') else (val_loss < best_val_loss)
        if improved:
            best_val_acc = val_acc
            best_val_loss = float(val_loss)
            test_acc = tmp_test_acc
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter >= early_stopping:
            break

    # ---- post-run prints & debugging
    print(f"Fold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Test acc: {test_acc:.4f} | Best val acc: {best_val_acc:.4f}")

    if "ODE" not in str(_aget(args, 'model', '')):
        # Laplacian stats
        try:
            for i, sl in enumerate(getattr(model, "sheaf_learners", [])):
                L_max = sl.L.detach().max().item()
                L_min = sl.L.detach().min().item()
                L_avg = sl.L.detach().mean().item()
                L_abs_avg = sl.L.detach().abs().mean().item()
                print(f"Laplacian {i}: Max {L_max:.4f} | Min {L_min:.4f} | Avg {L_avg:.4f} | AbsAvg {L_abs_avg:.4f}")
        except Exception as e:
            print(f"[warn] Laplacian debug failed: {e}")

        # Epsilons / DualEpsilons
        try:
            eps = getattr(model, "epsilons", None)
            if eps is not None:
                with np.printoptions(precision=3, suppress=True):
                    L_eps = int(_aget(args, 'layers', len(eps)))
                    for i in range(0, L_eps):
                        print(f"Epsilons {i}: {eps[i].detach().cpu().numpy().flatten()}")
            if model_cls == DiscreteJointSheafDiffusionParams:
                dual_eps = getattr(model, "dual_epsilons", None)
                if dual_eps is not None:
                    with np.printoptions(precision=3, suppress=True):
                        for i in range(0, int(_aget(args, 'layers', len(dual_eps))) - 1):
                            print(f"DualEpsilons {i}: {dual_eps[i].detach().cpu().numpy().flatten()}")
        except Exception as e:
            print(f"[warn] epsilon debug failed: {e}")

    # ---- final W&B logs
    try:
        wandb.log({'best_test_acc': test_acc, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch})
    except Exception as e:
        print(f"[warn] wandb.log (final) failed: {e}")

    keep_running = bool(test_acc >= float(_aget(args, 'min_acc', 0.0) or 0.0))
    return test_acc, best_val_acc, keep_running


# ----------------------------- main -----------------------------

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # ---- select model class
    if args.model == 'DiagSheafODE':
        model_cls = DiagSheafDiffusion
    elif args.model == 'BundleSheafODE':
        model_cls = BundleSheafDiffusion
    elif args.model == 'GeneralSheafODE':
        model_cls = GeneralSheafDiffusion
    elif args.model == 'DiagSheaf':
        model_cls = DiscreteDiagSheafDiffusion
    elif args.model == 'BundleSheaf':
        model_cls = DiscreteBundleSheafDiffusion
    elif args.model == 'GeneralSheaf':
        model_cls = DiscreteGeneralSheafDiffusion
    # -------- Experimental (PolySD) --------
    elif args.model == 'DiagSheafChebyshev':
        model_cls = DiscreteDiagSheafDiffusionChebyshev
    elif args.model == 'BundleSheafChebyshev':
        model_cls = DiscreteBundleSheafDiffusionChebyshev
    elif args.model == 'GeneralSheafChebyshev':
        model_cls = DiscreteGeneralSheafDiffusionChebyshev
    # -------- Equivariant --------
    elif args.model == 'EquivariantDiagSheaf':
        model_cls = EquivariantDiscreteDiagSheafDiffusion
    elif args.model == 'EquivariantBundleSheaf':
        model_cls = EquivariantDiscreteBundleSheafDiffusion
    elif args.model == 'EquivariantGeneralSheaf':
        model_cls = EquivariantDiscreteGeneralSheafDiffusion
    elif args.model == 'EquivariantDiagSheafChebyshev':
        model_cls = EquivariantDiscreteDiagSheafDiffusionChebyshev
    # -------- Joint / Vanilla --------
    elif args.model == 'JointSheafParams':
        model_cls = DiscreteJointSheafDiffusionParams
    elif args.model == 'JointSheafParamsAlt':
        model_cls = DiscreteJointSheafDiffusionParamsAlt
    elif args.model == 'JointSheafVanilla':
        model_cls = DiscreteJointSheafVanillaDiffusion
    elif args.model == 'VanillaSheaf':
        model_cls = DiscreteVanillaDiffusion
    elif args.model == 'ConvSheaf':
        model_cls = DiscreteVanillaDiffusionAlt
    else:
        raise ValueError(f'Unknown model {args.model}')

    # ---- dataset
    if args.dataset == "synthetic_exp":
        dataset = get_synthetic_dataset(args.dataset, args)
    else:
        dataset = get_dataset(args.dataset)

    # optional positional encodings
    if args.evectors > 0:
        dataset = append_top_k_evectors(dataset, args.evectors)

    # ---- enrich args
    args.sha = sha
    args.graph_size = dataset[0].x.size(0)
    args.input_dim = dataset[0].x.shape[1]
    try:
        args.output_dim = dataset.num_classes
    except Exception:
        args.output_dim = torch.unique(dataset[0].y).shape[0]
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    assert args.normalised or args.deg_normalised
    if args.sheaf_decay is None:
        args.sheaf_decay = args.weight_decay

    # ---- seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---- W&B init
    print(f"Running with wandb account: {args.entity}")
    print(args)
    wandb.init(project=f"{args.dataset.capitalize()}_BestResults_PolySD_vs_Standard", config=vars(args), entity=args.entity)

    # ---- K-fold (or repeated) runs
    results = []
    for fold in tqdm(range(args.folds)):
        test_acc, best_val_acc, keep_running = run_exp(wandb.config, dataset, model_cls, fold)
        results.append([test_acc, best_val_acc])
        if not keep_running:
            break

    # ---- aggregate & log
    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100
    wandb_results = {'test_acc': test_acc_mean, 'val_acc': val_acc_mean, 'test_acc_std': test_acc_std}
    wandb.log(wandb_results)
    wandb.finish()

    # ---- finale
    model_name = args.model if args.evectors == 0 else f"{args.model}+LP{args.evectors}"
    print(f'{model_name} on {args.dataset} | SHA: {sha}')
    print(f'Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Val acc: {val_acc_mean:.4f}')
