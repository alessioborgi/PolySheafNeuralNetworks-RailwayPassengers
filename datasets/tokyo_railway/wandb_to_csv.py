"""Fetch the latest W&B sweep results from the Tokyo_Railway project and save to CSV."""

import wandb
import pandas as pd
import numpy as np

PROJECT = "Tokyo_Railway"
ENTITY = "sheaf_hypergraphs"  # set to your wandb username/team, or None to use the default
SHEAF_EDGE_WEIGHTS = True # if it was a sheaf-only to run to study the effect of sheaf edge weights

api = wandb.Api()
path = f"{ENTITY}/{PROJECT}" if ENTITY else PROJECT

# Get the most recent sweep
sweeps = api.project(PROJECT, entity=ENTITY).sweeps()
if not sweeps:
    raise SystemExit("No sweeps found in project " + PROJECT)

sweep = sweeps[0]  # latest sweep
print(f"Sweep: {sweep.id}  ({sweep.name or 'unnamed'})  —  {sweep.state}")

# Pull every run in the sweep
rows = []
for run in sweep.runs:
    if SHEAF_EDGE_WEIGHTS:
        row = {
            "run_name": run.name,
            "adjacency_type": run.config.get("sheaf_edge_adjacency", None),
            "mode": run.config.get("inductive", False) and "inductive" or "transductive",
            "train_mae": float(run.summary.get("rescaled_all_train_mae_mean", 0)),
            "val_mae": float(run.summary.get("rescaled_all_val_mae_mean", 0)),
            "test_mae": float(run.summary.get("rescaled_all_test_mae_mean", 0)),
            "train_mae_std": np.std([run.summary.get(f"fold{i}_rescaled_all_train_mae", 0) for i in range(3)]),
            "val_mae_std": np.std([run.summary.get(f"fold{i}_rescaled_all_val_mae", 0) for i in range(3)]),
            "test_mae_std": float(run.summary.get("rescaled_all_test_mae_std", 0)),
            "avg_train_time_s": float(run.summary.get("avg_fold_time_s", 0)),
            "std_train_time_s": float(run.summary.get("std_fold_time_s", 0)),
            "top_10_test_mae": float(run.summary.get("rescaled_top10_test_mae_mean", 0)),
            "top_100_test_mae": float(run.summary.get("rescaled_top100_test_mae_mean", 0)),
            "bottom_100_test_mae": float(run.summary.get("rescaled_bottom100_test_mae_mean", 0)),
        }
    else:
        row = {
            "run_name": run.name,
            "mode": run.config.get("inductive", False) and "inductive" or "transductive",
            "train_mae": float(run.summary.get("rescaled_all_train_mae_mean", 0)),
            "val_mae": float(run.summary.get("rescaled_all_val_mae_mean", 0)),
            "test_mae": float(run.summary.get("rescaled_all_test_mae_mean", 0)),
            "normalized_train_mae": -1/100 * float(run.summary.get("train_acc", 0)),
            "normalized_val_mae": -1/100 * float(run.summary.get("val_acc", 0)),
            "normalized_test_mae": -1/100 * float(run.summary.get("test_acc", 0)),
            "train_mae_std": np.std([run.summary.get(f"fold{i}_rescaled_all_train_mae", 0) for i in range(3)]),
            "val_mae_std": np.std([run.summary.get(f"fold{i}_rescaled_all_val_mae", 0) for i in range(3)]),
            "test_mae_std": float(run.summary.get("rescaled_all_test_mae_std", 0)),
            "normalized_train_mae_std": np.std([-1/100 * float(run.summary.get(f"fold{i}_train_acc", 0)) for i in range(3)]),
            "normalized_val_mae_std": np.std([-1/100 * float(run.summary.get(f"fold{i}_val_acc", 0)) for i in range(3)]),
            "normalized_test_mae_std": 1/100 * float(run.summary.get(f"test_acc_std", 0)),
            "avg_train_time_s": float(run.summary.get("avg_fold_time_s", 0)),
            "std_train_time_s": float(run.summary.get("std_fold_time_s", 0)),
            "top_10_test_mae": float(run.summary.get("rescaled_top10_test_mae_mean", 0)),
            "top_100_test_mae": float(run.summary.get("rescaled_top100_test_mae_mean", 0)),
            "bottom_100_test_mae": float(run.summary.get("rescaled_bottom100_test_mae_mean", 0)),
        }
    rows.append(row)

df = pd.DataFrame(rows)
out = f"sweep_{sweep.id}.csv"
df.to_csv(out, index=False)
print(f"Wrote {len(df)} runs → {out}")
