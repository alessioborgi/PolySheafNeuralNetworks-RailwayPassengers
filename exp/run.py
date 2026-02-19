import sys
import os
import random
import time
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
import git
import numpy as np
import wandb
from tqdm import tqdm
from torch_geometric.data import Data

# This is required here by wandb sweeps.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exp.parser import get_parser
from models.positional_encodings import append_top_k_evectors
from models.cont_models import (
    DiagSheafDiffusion, BundleSheafDiffusion, GeneralSheafDiffusion,
    DiagSheafDiffusion_Polynomial, BundleSheafDiffusion_Polynomial, GeneralSheafDiffusion_Polynomial
)
from models.disc_models import (
    DiscreteDiagSheafDiffusion, DiscreteBundleSheafDiffusion, DiscreteGeneralSheafDiffusion,
    DiscreteDiagSheafDiffusionPolynomial, DiscreteBundleSheafDiffusionPolynomial, DiscreteGeneralSheafDiffusionPolynomial
)
from utils.heterophilic import get_dataset, get_synthetic_dataset, get_fixed_splits

# reproducibility utilities (we will use them ONLY in resource_analysis mode)
from utils.reproducibility import set_reproducible, fold_seed, truthy

# resource analysis utilities (only used when resource_analysis=True)
from utils.resource_analysis import (
    ResourceMonitor,
    device_cuda_index,
    profiler_available,
    train_step_with_optional_flops,
    maybe_profile_macs_torchprofile,
)


# ----------------------------- helpers -----------------------------
def aget(args, key, default=None):
    # works with dict, argparse.Namespace, wandb.Config
    if isinstance(args, dict):
        return args.get(key, default)
    try:
        return args[key]
    except Exception:
        return getattr(args, key, default)


def normalize_device(dev):
    if isinstance(dev, torch.device):
        return dev
    return torch.device(str(dev))


# ----------------------------- train / eval -----------------------------
def train(model, optimizer, data, task="classification"):
    model.train()
    optimizer.zero_grad()
    out = model(data.x)[data.train_mask]
    if task == "regression":
        loss = F.l1_loss(out.squeeze(-1), data.y[data.train_mask].float())
    else:
        loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    del out


def test(model, data, task="classification"):
    model.eval()
    with torch.no_grad():
        logits = model(data.x)
        accs, losses, preds = [], [], []
        for _, mask in data("train_mask", "val_mask", "test_mask"):
            if task == "regression":
                pred = logits[mask].squeeze(-1)
                # Use negative MAE as "acc" so higher is better (consistent with early stopping)
                mae = F.l1_loss(pred, data.y[mask].float())
                acc = -mae.item()
                loss = F.l1_loss(pred, data.y[mask].float())
            else:
                pred = logits[mask].max(1)[1]
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                loss = F.nll_loss(logits[mask], data.y[mask])
            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses


# =====================================================================
#  CLASSIC FOLD RUN (matches your old behavior when resource_analysis=False)
# =====================================================================
def run_exp_classic(args, dataset, model_cls, fold: int) -> Tuple[float, float, bool]:
    """
    This is intentionally kept as close as possible to your original script:
      - no per-fold reproducibility changes here (relies on global seeding in main)
      - only fold 0 logs per-epoch with step=epoch
      - discrete debug prints identical pattern
      - fold summary logs: best_test_acc/best_val_acc/best_epoch (same keys)
    """
    data = dataset[0]
    data = get_fixed_splits(data, aget(args, "dataset"), fold)
    print(f"data splits for fold {fold}:")
    print(f"  train: {data.train_mask.sum().item()} samples")
    print(f"  val: {data.val_mask.sum().item()} samples")
    print(f"  test: {data.test_mask.sum().item()} samples")

    data = data.to(aget(args, "device"))

    model = model_cls(data.edge_index, args).to(aget(args, "device"))

    sheaf_learner_params, other_params = model.grouped_parameters()
    optimizer = torch.optim.Adam([
        {
            "params": sheaf_learner_params,
            "weight_decay": aget(args, "sheaf_decay"),
            "lr": aget(args, "maps_lr") if aget(args, "maps_lr") is not None else aget(args, "lr"),
        },
        {
            "params": other_params,
            "weight_decay": aget(args, "weight_decay"),
            "lr": aget(args, "lr"),
        }
    ])

    best_val_acc = 0.0
    best_val_loss = float("inf")
    test_acc = 0.0
    best_epoch = 0
    bad_counter = 0

    epochs = int(aget(args, "epochs", 200))
    early_stopping = int(aget(args, "early_stopping", 50))
    stop_strategy = str(aget(args, "stop_strategy", "acc"))

    for epoch in range(epochs):
        task = str(aget(args, "task", "classification"))
        train(model, optimizer, data, task=task)
        [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test(model, data, task=task)

        # EXACTLY AS BEFORE: only fold 0, and with step=epoch
        if fold == 0:
            wandb.log({
                f"fold{fold}_train_acc": float(train_acc),
                f"fold{fold}_train_loss": float(train_loss),
                f"fold{fold}_val_acc": float(val_acc),
                f"fold{fold}_val_loss": float(val_loss),
                f"fold{fold}_tmp_test_acc": float(tmp_test_acc),
                f"fold{fold}_tmp_test_loss": float(tmp_test_loss),
            }, step=epoch)

        new_best = (val_acc > best_val_acc) if (stop_strategy == "acc") else (val_loss < best_val_loss)
        if new_best:
            best_val_acc = float(val_acc)
            best_val_loss = float(val_loss)
            test_acc = float(tmp_test_acc)
            best_epoch = int(epoch)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == early_stopping:
            break

    print(f"Fold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Best val acc: {best_val_acc:.4f}")

    # Debugging for discrete models (same spirit as your old script)
    if "ODE" not in str(aget(args, "model", "")):
        try:
            for i in range(len(model.sheaf_learners)):
                L_max = model.sheaf_learners[i].L.detach().max().item()
                L_min = model.sheaf_learners[i].L.detach().min().item()
                L_avg = model.sheaf_learners[i].L.detach().mean().item()
                L_abs_avg = model.sheaf_learners[i].L.detach().abs().mean().item()
                print(f"Laplacian {i}: Max: {L_max:.4f}, Min: {L_min:.4f}, Avg: {L_avg:.4f}, Abs avg: {L_abs_avg:.4f}")

            with np.printoptions(precision=3, suppress=True):
                for i in range(0, int(aget(args, "layers", 0))):
                    print(f"Epsilons {i}: {model.epsilons[i].detach().cpu().numpy().flatten()}")
        except Exception as e:
            print(f"[warn] discrete debug failed: {e}")

    # EXACTLY AS BEFORE: same keys (no fold prefix)
    wandb.log({
        "best_test_acc": float(test_acc),
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
    })

    keep_running = float(test_acc) >= float(aget(args, "min_acc", 0.0))
    return float(test_acc), float(best_val_acc), keep_running


# =====================================================================
#  RESOURCE-ANALYSIS FOLD RUN (single run, global_step, fold aggregates)
# =====================================================================
def run_exp_resource(args, dataset, model_cls, fold: int) -> Tuple[float, float, bool, Dict[str, Any]]:
    # reproducibility (per fold)
    base_seed = int(aget(args, "seed", 0))
    fseed = fold_seed(base_seed, fold)

    deterministic = bool(aget(args, "deterministic", True))
    strict = truthy(os.environ.get("STRICT_DETERMINISM", "0")) or bool(aget(args, "strict_determinism", False))
    set_reproducible(fseed, deterministic=deterministic, strict=strict)

    # data
    data = dataset[0]
    data = get_fixed_splits(data, aget(args, "dataset"), fold)

    device = normalize_device(aget(args, "device", "cpu"))
    data = data.to(device)

    max_epochs = int(aget(args, "epochs", 0))
    fold_step_base = int(fold) * int(max_epochs)

    mon = None
    try:
        # monitoring
        cuda_idx = device_cuda_index(device)
        log_every_s = float(aget(args, "sys_log_every_s", 1.0))
        mon = ResourceMonitor(
            cuda_index=cuda_idx,
            log_every_s=log_every_s,
            disk_path=".",
            prefix=f"fold{fold}_sys",
        )
        mon.start()

        if torch.cuda.is_available() and str(device).startswith("cuda"):
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        # model
        model = model_cls(data.edge_index, args).to(device)

        # One-shot MACs proxy
        macs = maybe_profile_macs_torchprofile(model, data.x)
        if macs is not None:
            wandb.log({
                "global_step": fold_step_base,
                "fold": int(fold),
                "epoch": 0,
                f"fold{fold}_macs_forward_torchprofile": float(macs),
                f"fold{fold}_flops_forward_from_macs": float(2.0 * macs),
            })

        sheaf_learner_params, other_params = model.grouped_parameters()
        optimizer = torch.optim.Adam([
            {
                "params": sheaf_learner_params,
                "weight_decay": float(aget(args, "sheaf_decay")),
                "lr": float(aget(args, "maps_lr")) if aget(args, "maps_lr") is not None else float(aget(args, "lr")),
            },
            {
                "params": other_params,
                "weight_decay": float(aget(args, "weight_decay")),
                "lr": float(aget(args, "lr")),
            }
        ])

        best_val_acc = 0.0
        best_val_loss = float("inf")
        test_acc = 0.0
        best_epoch = 0
        bad_counter = 0

        epochs = int(aget(args, "epochs", 200))
        early_stopping = int(aget(args, "early_stopping", 50))
        stop_strategy = str(aget(args, "stop_strategy", "acc")).lower()

        # buffers
        t_fold0 = time.perf_counter()
        t_best = None
        val_acc_hist: List[float] = []
        val_loss_hist: List[float] = []
        step_times: List[float] = []
        flops_samples: List[float] = []

        # profile_flops may or may not exist; default True in this mode
        profile_flops_flag = bool(aget(args, "profile_flops", True))
        profile_flops = profile_flops_flag and profiler_available()
        flops_profile_epochs = int(aget(args, "flops_profile_epochs", 1))

        for epoch in range(epochs):
            global_step = fold_step_base + int(epoch)

            do_profile_now = profile_flops and (epoch < flops_profile_epochs)
            task = str(aget(args, "task", "classification"))
            flops, step_time_s = train_step_with_optional_flops(
                enabled=profile_flops,
                device=device,
                do_profile_now=do_profile_now,
                train_fn=train,
                model=model,
                optimizer=optimizer,
                data=data,
                task=task,
            )
            step_times.append(float(step_time_s))
            if flops is not None:
                flops_samples.append(float(flops))

            task = str(aget(args, "task", "classification"))
            [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test(model, data, task=task)

            # Per-epoch curves only for fold 0
            if fold == 0:
                wandb.log({
                    "global_step": global_step,
                    "fold": int(fold),
                    "epoch": int(epoch),
                    f"fold{fold}_train_acc": float(train_acc),
                    f"fold{fold}_train_loss": float(train_loss),
                    f"fold{fold}_val_acc": float(val_acc),
                    f"fold{fold}_val_loss": float(val_loss),
                    f"fold{fold}_tmp_test_acc": float(tmp_test_acc),
                    f"fold{fold}_tmp_test_loss": float(tmp_test_loss),
                })

            val_acc_hist.append(float(val_acc))
            val_loss_hist.append(float(val_loss))

            alloc_gb = reserv_gb = max_alloc_gb = max_reserv_gb = None
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                try:
                    torch.cuda.synchronize()
                    alloc_gb = torch.cuda.memory_allocated() / 1e9
                    reserv_gb = torch.cuda.memory_reserved() / 1e9
                    max_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
                    max_reserv_gb = torch.cuda.max_memory_reserved() / 1e9
                except Exception:
                    pass

            wandb.log({
                "global_step": global_step,
                "fold": int(fold),
                "epoch": int(epoch),
                f"fold{fold}_time_epoch_s": float(step_time_s) if step_time_s is not None else None,
                f"fold{fold}_flops_epoch_profiler": float(flops) if flops is not None else None,
                f"fold{fold}_torch_mem_alloc_gb": alloc_gb,
                f"fold{fold}_torch_mem_reserved_gb": reserv_gb,
                f"fold{fold}_torch_max_mem_alloc_gb": max_alloc_gb,
                f"fold{fold}_torch_max_mem_reserved_gb": max_reserv_gb,
            })

            improved = (val_acc > best_val_acc) if (stop_strategy == "acc") else (val_loss < best_val_loss)
            if improved:
                best_val_acc = float(val_acc)
                best_val_loss = float(val_loss)
                test_acc = float(tmp_test_acc)
                best_epoch = int(epoch)
                bad_counter = 0
                t_best = time.perf_counter()
            else:
                bad_counter += 1

            if bad_counter >= early_stopping:
                break

        print(f"Fold {fold} | Best epoch: {best_epoch} | Best val acc: {best_val_acc:.4f} | Test acc: {test_acc:.4f}")

        fold_summary: Dict[str, Any] = {
            "fold": int(fold),
            "fold_seed": int(fseed),
            f"fold{fold}_best_test_acc": float(test_acc),
            f"fold{fold}_best_val_acc": float(best_val_acc),
            f"fold{fold}_best_epoch": int(best_epoch),
            "deterministic_enabled": bool(deterministic),
            "strict_determinism_enabled": bool(strict),
        }

        fold_time_s = time.perf_counter() - t_fold0
        sys_agg = mon.aggregates() if mon is not None else {}

        avg_step_time_s = float(np.mean(step_times)) if step_times else None
        avg_step_time_ms = (1000.0 * avg_step_time_s) if avg_step_time_s is not None else None

        avg_flops_per_epoch = float(np.mean(flops_samples)) if flops_samples else None
        avg_gflops_per_epoch = (avg_flops_per_epoch / 1e9) if avg_flops_per_epoch is not None else None

        fold_summary.update({
            f"fold{fold}_fold_time_s": float(fold_time_s),
            f"fold{fold}_time_to_best_s": float((t_best - t_fold0) if t_best is not None else fold_time_s),
            f"fold{fold}_avg_step_time_ms": avg_step_time_ms,
            f"fold{fold}_avg_flops_per_epoch_profiler": avg_flops_per_epoch,
            f"fold{fold}_avg_gflops_per_epoch_profiler": avg_gflops_per_epoch,
            **sys_agg,
        })

        wandb.log(fold_summary)

        keep_running = float(test_acc) >= float(aget(args, "min_acc", 0.0))
        return float(test_acc), float(best_val_acc), keep_running, fold_summary

    finally:
        if mon is not None:
            try:
                mon.stop()
            except Exception:
                pass


# ----------------------------- main -----------------------------
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # ---------------- Model routing ----------------
    if args.model == "DiagSheafODE":
        model_cls = DiagSheafDiffusion
    elif args.model == "BundleSheafODE":
        model_cls = BundleSheafDiffusion
    elif args.model == "GeneralSheafODE":
        model_cls = GeneralSheafDiffusion
    elif args.model in ("DiagSheafODEPolynomial", "DiagSheafODEPoly"):
        model_cls = DiagSheafDiffusion_Polynomial
    elif args.model in ("BundleSheafODEPolynomial", "BundleSheafODEPoly"):
        model_cls = BundleSheafDiffusion_Polynomial
    elif args.model in ("GeneralSheafODEPolynomial", "GeneralSheafODEPoly"):
        model_cls = GeneralSheafDiffusion_Polynomial
    elif args.model == "DiagSheaf":
        model_cls = DiscreteDiagSheafDiffusion
    elif args.model == "BundleSheaf":
        model_cls = DiscreteBundleSheafDiffusion
    elif args.model == "GeneralSheaf":
        model_cls = DiscreteGeneralSheafDiffusion
    elif args.model == "DiagSheafPolynomial":
        model_cls = DiscreteDiagSheafDiffusionPolynomial
    elif args.model == "BundleSheafPolynomial":
        model_cls = DiscreteBundleSheafDiffusionPolynomial
    elif args.model == "GeneralSheafPolynomial":
        model_cls = DiscreteGeneralSheafDiffusionPolynomial
    else:
        raise ValueError(f"Unknown model {args.model}")

    # ---------------- Dataset ----------------
    if args.dataset == "synthetic_exp":
        dataset = get_synthetic_dataset(args.dataset, args)
    else:
        dataset = get_dataset(args.dataset)

    if args.evectors > 0:
        dataset = append_top_k_evectors(dataset, args.evectors)

    # ---------------- Enrich args ----------------
    args.sha = sha
    args.graph_size = dataset[0].x.size(0)
    args.input_dim = dataset[0].x.shape[1]
    if args.task == "regression":
        # For regression, output_dim = number of target columns (typically 1)
        args.output_dim = 1 if dataset[0].y.dim() == 1 else dataset[0].y.shape[1]
    else:
        try:
            args.output_dim = dataset.num_classes
        except Exception:
            args.output_dim = torch.unique(dataset[0].y).shape[0]
    args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    assert args.normalised or args.deg_normalised
    if args.sheaf_decay is None:
        args.sheaf_decay = args.weight_decay

    # ---------------- mode switch ----------------
    resource_analysis = bool(getattr(args, "resource_analysis", False))

    # If resource_analysis is OFF, you want the behavior EXACTLY as before:
    # use classic seeding, classic wandb step=epoch logging, etc.
    if not resource_analysis:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    else:
        # Resource mode: keep your reproducibility utilities
        strict = truthy(os.environ.get("STRICT_DETERMINISM", "0")) or bool(getattr(args, "strict_determinism", False))
        set_reproducible(int(args.seed), deterministic=bool(getattr(args, "deterministic", True)), strict=bool(strict))

    print(f"Running with wandb account: {args.entity}")
    print(args)

    # Use different default project names per mode if you want “exactly as before”
    default_project = "Chameleon_BestResults_PolySD_vs_Standard" if not resource_analysis else "Convergence_Ablation_Chameleon"
    project_name = getattr(args, "wandb_project", None) or default_project

    # ---------------- SINGLE W&B RUN ----------------
    wandb.init(
        project=project_name,
        entity=args.entity,
        config={**vars(args), "sha": sha},
        name=f"{args.model}-{args.dataset}-seed{args.seed}",
    )

    # Only define global_step metrics in resource mode (classic mode stays identical)
    if resource_analysis:
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

        try:
            wandb.config.update({
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cudnn_version": torch.backends.cudnn.version(),
                "resource_analysis": True,
                "profiler_available": bool(profiler_available()),
            }, allow_val_change=True)
        except Exception:
            pass

    results: List[List[float]] = []
    fold_summaries: List[Dict[str, Any]] = []

    for fold in tqdm(range(int(args.folds))):
        if not resource_analysis:
            test_acc, best_val_acc, keep_running = run_exp_classic(wandb.config, dataset, model_cls, fold)
            results.append([test_acc, best_val_acc])
            if not keep_running:
                break
        else:
            test_acc, best_val_acc, keep_running, fold_summary = run_exp_resource(wandb.config, dataset, model_cls, fold)
            results.append([test_acc, best_val_acc])
            fold_summaries.append(fold_summary)
            if not keep_running:
                break

    # ---------------- aggregate across folds (ACC) ----------------
    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    wandb_results = {"test_acc": float(test_acc_mean), "val_acc": float(val_acc_mean), "test_acc_std": float(test_acc_std)}
    wandb.log(wandb_results)

    # In resource mode, also log cross-fold FLOPs/time aggregates (extra keys)
    if resource_analysis:
        gflops_vals = []
        time_ms_vals = []
        for fs in fold_summaries:
            for k, v in fs.items():
                if k.endswith("_avg_gflops_per_epoch_profiler") and v is not None:
                    gflops_vals.append(float(v))
                if k.endswith("_avg_step_time_ms") and v is not None:
                    time_ms_vals.append(float(v))

        avg_gflops_10fold = float(np.mean(gflops_vals)) if len(gflops_vals) else None
        std_gflops_10fold = float(np.std(gflops_vals)) if len(gflops_vals) else None
        avg_time_ms_10fold = float(np.mean(time_ms_vals)) if len(time_ms_vals) else None
        std_time_ms_10fold = float(np.std(time_ms_vals)) if len(time_ms_vals) else None

        wandb.log({
            "cv/avg_gflops_per_epoch_profiler_mean": avg_gflops_10fold,
            "cv/avg_gflops_per_epoch_profiler_std": std_gflops_10fold,
            "cv/avg_step_time_ms_mean": avg_time_ms_10fold,
            "cv/avg_step_time_ms_std": std_time_ms_10fold,
        })

        # store in summary too (nice in tables)
        wandb.run.summary["cv/avg_gflops_per_epoch_profiler_mean"] = avg_gflops_10fold
        wandb.run.summary["cv/avg_gflops_per_epoch_profiler_std"] = std_gflops_10fold
        wandb.run.summary["cv/avg_step_time_ms_mean"] = avg_time_ms_10fold
        wandb.run.summary["cv/avg_step_time_ms_std"] = std_time_ms_10fold

    wandb.finish()

    model_name = args.model if args.evectors == 0 else f"{args.model}+LP{args.evectors}"
    print(f"{model_name} on {args.dataset} | SHA: {sha}")
    print(f"Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Val acc: {val_acc_mean:.4f}")




