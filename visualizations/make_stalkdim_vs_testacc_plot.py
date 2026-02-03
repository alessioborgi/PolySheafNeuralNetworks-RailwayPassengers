# 2x2 figure: Test accuracy vs. Stalk dimension (one panel per dataset)
# - No seaborn, plain matplotlib
# - Lines connect model points across stalk dimensions
# - Y-axis zoom per-dataset (Cora=80–90, Film=30–40; PubMed & Texas auto-zoomed unless specified)
#
# Usage: just run this file in the same directory as your CSV, or change CSV_PATH.

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIG ---
CSV_PATH = "visualizations/csv_files/StalkDim_VS_Accuracy/Stalk_Dimension_VS_Test_Accuracy.csv"  # change if needed

# Optional manual zooms; leave None to auto-zoom around data with small padding
Y_ZOOMS: Dict[str, Optional[Tuple[float, float]]] = {
    "Cora":   (80.0, 90.0),
    "Film":   (30.0, 40.0),
    "PubMed": (89.3, 90.0),   # tweak if you want tighter/looser
    "Texas":  (86.5, 90.5),   # tweak if you want tighter/looser
}

# Dataset order in the 2×2 grid (row-major). Adjust if you prefer a different layout.
DATASET_ORDER: List[str] = ["Cora", "PubMed", "Texas", "Film"]

# --- PARSER FOR YOUR CSV FORMAT ---
def parse_stalk_csv(path: str) -> pd.DataFrame:
    """
    Parse the custom semicolon CSV:
      Line 0: title (ignored)
      Line 1: ';2;3;4;5;;' -> stalk dims
      Lines: 'Model;v2;v3;v4;v5;Dataset;'
      Values like '89.61\\pm0.46' -> take mean (left of \\pm)
    Returns long-format DataFrame with columns: Dataset, Model, StalkDim, TestAccuracy
    """
    p = Path(path)
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]

    # Extract stalk dimensions from second line
    header = lines[1]
    dims: List[int] = []
    for tok in header.split(";"):
        try:
            dims.append(int(tok))
        except Exception:
            pass
    if not dims:
        raise ValueError("Could not extract stalk dimensions from header line.")

    records = []
    for line in lines[2:]:
        parts = line.split(";")
        if len(parts) < (1 + len(dims) + 1):
            # not enough fields (model + values + dataset)
            continue
        model = parts[0].strip()
        dataset = parts[-2].strip()
        values = parts[1:1+len(dims)]
        for sd, raw in zip(dims, values):
            m = raw.split("\\pm", 1)[0]  # mean before "\pm"
            try:
                val = float(m.replace(",", "."))
            except Exception:
                val = float("nan")
            records.append(
                {"Dataset": dataset, "Model": model, "StalkDim": int(sd), "TestAccuracy": val}
            )

    df = pd.DataFrame(records).sort_values(["Dataset", "Model", "StalkDim"]).reset_index(drop=True)
    return df

def auto_zoom(min_v: float, max_v: float, pad_ratio: float = 0.05) -> Tuple[float, float]:
    span = max(1e-9, max_v - min_v)
    pad = pad_ratio * span
    return min_v - pad, max_v + pad

# --- MAIN ---
df = parse_stalk_csv(CSV_PATH)

# Ensure all four datasets are present
datasets = list(pd.unique(df["Dataset"]))
# If DATASET_ORDER has datasets not present, filter them out; append any others at the end
ordered = [d for d in DATASET_ORDER if d in datasets] + [d for d in datasets if d not in DATASET_ORDER]

# Build 2×2 figure
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
axes = axes.ravel()

xticks = [2, 3, 4, 5]

for ax, ds in zip(axes, ordered[:4]):  # limit to 4 panels
    dsub = df[df["Dataset"] == ds]
    # One line per model
    for model, msub in dsub.groupby("Model"):
        msub = msub.sort_values("StalkDim")
        ax.plot(msub["StalkDim"], msub["TestAccuracy"], marker="o", label=str(model))

    ax.set_title(ds + " — Test Accuracy vs. Stalk Dimension")
    ax.set_xlabel("Stalk Dimension (d)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xticks(xticks)
    ax.grid(True)

    # Apply zoom
    if Y_ZOOMS.get(ds) is not None:
        ymin, ymax = Y_ZOOMS[ds]
    else:
        ymin = dsub["TestAccuracy"].min()
        ymax = dsub["TestAccuracy"].max()
        ymin, ymax = auto_zoom(ymin, ymax, pad_ratio=0.03)
    ax.set_ylim(ymin, ymax)

# If fewer than 4 datasets, hide unused axes
for ax in axes[len(ordered):]:
    ax.axis("off")

# Put a single legend outside (bottom) aggregating handles from the first used axes
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True, title="Model")

fig.suptitle("Test Accuracy (%) vs. Stalk Dimension (d)", y=0.98)
fig.tight_layout(rect=[0, 0.05, 1, 0.96])  # leave space for the bottom legend

out_path = Path("visualizations/csv_files/StalkDim_VS_Accuracy/Accuracy_vs_StalkDim_2x2.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved: {out_path.resolve()}")
