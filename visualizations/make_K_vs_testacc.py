"""
3×3 Chebyshev K-sweep with method color families + horizontal offsets.
- Analytic = Reds (Diag darkest, Bundle mid, General light)
- Iterative = Blues (Diag darkest, Bundle mid, General light)
- Curves at each K are horizontally offset so they appear side-by-side
- One global legend centered at the bottom
- Robust normalization + completeness checks

pip install pandas matplotlib pillow
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

# --------------------- CONFIG ---------------------
CSV_PATH = "visualizations/csv_files/K_vs_Accuracy/K_vs_testacc.csv"  # <-- your CSV path
OUT_DIR = Path("visualizations/csv_files/K_vs_Accuracy/chebK_sweep_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREFERRED_ORDER = [
    "texas", "wisconsin", "film",
    "squirrel", "chameleon", "cornell",
    "citeseer", "pubmed", "cora"
]
K_EXPECTED = [1, 2, 4, 8, 12, 16]

# panel + canvas settings
PANEL_W_IN, PANEL_H_IN, DPI = 6, 4.7, 180
PAD_BETWEEN_PANELS = 20
LEGEND_HEIGHT_PX = 120
USE_STD_SHADING = False
# --------------------------------------------------

# --------------------- Normalization ---------------------
def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())

def canonical_model(raw: str) -> str:
    s = _clean(raw).lower()
    if "diag" in s or "diagonal" in s: return "DiagChebySD"
    if "bundle" in s:                   return "BundleChebySD"
    if "general" in s or "full" in s:   return "GeneralChebySD"
    return _clean(raw)

def canonical_method(raw: str) -> str:
    s = _clean(raw).lower()
    if "iter" in s: return "Iterative"
    if "anal" in s or "bound" in s: return "Analytic"
    return _clean(raw).title()

EXPECTED_LABELS = [
    "DiagChebySD Analytic", "DiagChebySD Iterative",
    "BundleChebySD Analytic", "BundleChebySD Iterative",
    "GeneralChebySD Analytic", "GeneralChebySD Iterative",
]

# --------------------- Styling ---------------------
# color families (method) + shades (model order: Diag -> Bundle -> General)
REDS   = mpl.colormaps["Reds"]
BLUES  = mpl.colormaps["Blues"]
SHADE_VALS = {"DiagChebySD": 0.55, "BundleChebySD": 0.70, "GeneralChebySD": 0.85}

def color_for(model_base: str, method: str):
    shade = SHADE_VALS.get(model_base, 0.70)
    return (REDS if method == "Analytic" else BLUES)(shade)

# horizontal offsets so curves at the same K are side-by-side (never on top)
# We place analytic triplet left of the tick, iterative triplet right of the tick.
GROUP_SHIFT   = 0.18  # move the whole method group left/right of the tick
INNER_SHIFT   = 0.06  # small spacing within each method group
INNER_BY_MODEL = {"DiagChebySD": -INNER_SHIFT, "BundleChebySD": 0.0, "GeneralChebySD": +INNER_SHIFT}
GROUP_BY_METHOD = {"Analytic": -GROUP_SHIFT, "Iterative": +GROUP_SHIFT}

def x_with_offset(k, model_base, method):
    return float(k) + GROUP_BY_METHOD[method] + INNER_BY_MODEL[model_base]

# --------------------- Load & checks ---------------------
path = Path(CSV_PATH)
if not path.exists():
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

df_raw = pd.read_csv(path)
req = ["dataset", "model", "lambda_max_choice", "chebyshev_layers_K", "test_acc"]
miss = [c for c in req if c not in df_raw.columns]
if miss:
    raise ValueError(f"Missing columns: {miss}\nAvailable: {list(df_raw.columns)}")

df = df_raw.copy()
df["dataset"] = df["dataset"].map(_clean).str.lower()
df["ModelBase"] = df["model"].map(canonical_model)
df["MethodPretty"] = df["lambda_max_choice"].map(canonical_method)
df["Label"] = df["ModelBase"] + " " + df["MethodPretty"]
df["K"] = pd.to_numeric(df["chebyshev_layers_K"], errors="coerce")
df["TestAccuracy"] = pd.to_numeric(df["test_acc"], errors="coerce")

# quick diagnostics
unexpected_models  = sorted(set(df["ModelBase"].unique()) - set(["DiagChebySD","BundleChebySD","GeneralChebySD"]))
unexpected_methods = sorted(set(df["MethodPretty"].unique()) - set(["Analytic","Iterative"]))
if unexpected_models:  print("[WARN] Unexpected model strings:", unexpected_models)
if unexpected_methods: print("[WARN] Unexpected lambda_max_choice strings:", unexpected_methods)

nan_rows = df[df["TestAccuracy"].isna()]
if not nan_rows.empty:
    print("[WARN] NaN test_acc rows (will be ignored):", len(nan_rows))

# confirm completeness (36 rows/dataset expected)
def report_missing(dfc: pd.DataFrame):
    ok = True
    for ds, g in dfc.groupby("dataset"):
        labs = set(g["Label"].unique())
        miss_labs = [lab for lab in EXPECTED_LABELS if lab not in labs]
        miss_per = {}
        for lab in EXPECTED_LABELS:
            sub = g[g["Label"] == lab]
            present_k = set(sub["K"].dropna().unique().tolist())
            miss_k = [k for k in K_EXPECTED if k not in present_k]
            if miss_k: miss_per[lab] = miss_k
        if miss_labs or any(miss_per.values()):
            ok = False
            print(f"[WARN] Dataset '{ds}' incomplete.")
            if miss_labs:
                print("  Missing labels:", miss_labs)
            for lab, ks in miss_per.items():
                print(f"  {lab} missing K: {ks}")
    if ok:
        print("[OK] All datasets include 6 labels × all K.")
report_missing(df)

# --------------------- Aggregate ---------------------
df_plot = df.dropna(subset=["TestAccuracy"]).copy()
agg = (df_plot.groupby(["dataset", "ModelBase", "MethodPretty", "K"])["TestAccuracy"]
              .agg(mean="mean", std="std", count="count").reset_index())

datasets = agg["dataset"].unique().tolist()
ordered = [d for d in PREFERRED_ORDER if d in datasets] + [d for d in datasets if d not in PREFERRED_ORDER]
ordered = ordered[:9]

# --------------------- Draw per-dataset panels (offset lines) ---------------------
panel_paths = []
for ds in ordered:
    dsub = agg[agg["dataset"] == ds]

    fig = plt.figure(figsize=(PANEL_W_IN, PANEL_H_IN))
    ax = plt.gca()

    # Draw Analytic first (left side), Iterative second (right side)
    for method in ["Analytic", "Iterative"]:
        for model in ["DiagChebySD", "BundleChebySD", "GeneralChebySD"]:
            msub = dsub[(dsub["ModelBase"] == model) & (dsub["MethodPretty"] == method)].sort_values("K")
            if msub.empty:
                continue
            x = [x_with_offset(k, model, method) for k in msub["K"]]
            y = msub["mean"].to_numpy()
            ax.plot(
                x, y,
                marker="o", linewidth=2.0, markersize=4.8,
                color=color_for(model, method),
                solid_capstyle="round", alpha=0.98,
            )
            if USE_STD_SHADING and np.isfinite(msub["std"]).any():
                y1 = (msub["mean"] - msub["std"]).to_numpy()
                y2 = (msub["mean"] + msub["std"]).to_numpy()
                ax.fill_between(x, y1, y2, color=color_for(model, method), alpha=0.12)

    # aesthetics
    ax.set_title(f"{ds} — Test Accuracy vs. Chebyshev Order (K)")
    ax.set_xlabel("Chebyshev Order (K)")
    ax.set_ylabel("Test Accuracy")
    ax.set_xticks(K_EXPECTED)
    ax.set_xticklabels([str(k) for k in K_EXPECTED])
    ax.grid(True, alpha=0.35)

    out_path = OUT_DIR / f"K_Sweep_{ds}.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    panel_paths.append(out_path)

# --------------------- Stitch 3×3 + one global legend ---------------------
images = [Image.open(p).convert("RGB") for p in panel_paths]
while len(images) < 9:
    blank = Image.new("RGB", (int(PANEL_W_IN * DPI), int(PANEL_H_IN * DPI)), color=(255, 255, 255))
    images.append(blank)

cols, rows = 3, 3
panel_w, panel_h = images[0].size
canvas_w = cols * panel_w + (cols - 1) * PAD_BETWEEN_PANELS
canvas_h = rows * panel_h + (rows - 1) * PAD_BETWEEN_PANELS + LEGEND_HEIGHT_PX
canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

idx = 0
for r in range(rows):
    for c in range(cols):
        x = c * (panel_w + PAD_BETWEEN_PANELS)
        y = r * (panel_h + PAD_BETWEEN_PANELS)
        canvas.paste(images[idx], (x, y))
        idx += 1

from PIL import Image, ImageOps

# 1) Build the legend FIRST and measure it
def handle(model, method):
    return Line2D([0], [0],
                  marker='o', linestyle='-',
                  color=color_for(model, method),
                  lw=2.2, markersize=6.5,
                  label=f"{model} {method}")

handles = [
    handle("DiagChebySD", "Analytic"),
    handle("BundleChebySD", "Analytic"),
    handle("GeneralChebySD", "Analytic"),
    handle("DiagChebySD", "Iterative"),
    handle("BundleChebySD", "Iterative"),
    handle("GeneralChebySD", "Iterative"),
]

# bigger legend figure + fonts
leg_fig = plt.figure(figsize=(12, 1.8), dpi=180)
ax = leg_fig.add_subplot(111)
ax.axis('off')
ax.legend(handles=handles, ncol=3, loc='center', frameon=True,
          title="Model × λ_max method", fontsize=10, title_fontsize=11)
leg_fig.tight_layout(pad=0.2)
legend_png = OUT_DIR / "legend_tmp.png"
leg_fig.savefig(legend_png, transparent=True, bbox_inches="tight", pad_inches=0.15)
plt.close(leg_fig)

legend_img = Image.open(legend_png).convert("RGBA")
LEGEND_TOP_MARGIN = 18
LEGEND_BOTTOM_MARGIN = 18
legend_height = legend_img.size[1] + LEGEND_TOP_MARGIN + LEGEND_BOTTOM_MARGIN

# 2) Load panels (already created above)
images = [Image.open(p).convert("RGB") for p in panel_paths]
while len(images) < 9:
    blank = Image.new("RGB", (int(PANEL_W_IN * DPI), int(PANEL_H_IN * DPI)), color=(255, 255, 255))
    images.append(blank)

# 3) Create canvas with height based on legend size
cols, rows = 3, 3
panel_w, panel_h = images[0].size
canvas_w = cols * panel_w + (cols - 1) * PAD_BETWEEN_PANELS
canvas_h = rows * panel_h + (rows - 1) * PAD_BETWEEN_PANELS + legend_height
canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

# 4) Paste the 9 panels
idx = 0
for r in range(rows):
    for c in range(cols):
        x = c * (panel_w + PAD_BETWEEN_PANELS)
        y = r * (panel_h + PAD_BETWEEN_PANELS)
        canvas.paste(images[idx], (x, y))
        idx += 1

# 5) Paste the legend centered in the new bottom band
leg_x = (canvas_w - legend_img.size[0]) // 2
leg_y = rows * (panel_h + PAD_BETWEEN_PANELS) + LEGEND_TOP_MARGIN
canvas.paste(legend_img, (leg_x, leg_y), legend_img)

final_path = OUT_DIR / "ChebK_Sweep_3x3_offset_families.png"
canvas.save(final_path)
print(f"[saved] {final_path.resolve()}")