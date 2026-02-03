import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ===================== User controls =====================

# File mapping (adjust paths as needed)
LEFT_LABEL    = "RiSNN"  # used in column title
RIGHT_LABEL   = "Diff"   # used in column title
LEFT_CSV      = "visualizations/csv_files/Heterophily/Het_RiSNN.csv"
RIGHT_CSV     = "visualizations/csv_files/Heterophily/Het_Diff.csv"
CHEBY_LEFT    = "visualizations//csv_files/Heterophily/Het_ChebyRiSNN.csv"
CHEBY_RIGHT   = "visualizations//csv_files/Heterophily/Het_ChebyDiff.csv"

# Which 4 rows (num_classes) and which het levels to display in each subplot.
# If left empty, they’re inferred from the data and truncated to 4 rows and all hets.
FORCE_CLASS_ROWS: List[int]   = []                  # e.g. [2,3,4,5]
FORCE_HETS: List[float]       = []                  # e.g. [0.0, 0.25, 0.5, 0.75, 1.0]
# Recommended (prevents weird ordering if CSVs mix separators):
# FORCE_HETS = [0.0, 0.25, 0.5, 0.75, 1.0]

# Optional preferred model order for legends; others appended alphabetically.
MODEL_ORDER: List[str] = [
    "MLP", "GCN", "GraphSAGE", "VanillaSheaf", "DiagSheaf",
    "JdSNN-noW", "JdSNN-WO", "JdSNN-W",
    "RiSNN-NoT", "RiSNN-T",
    # Chebyshev variants (two naming styles supported)
    "Diag-ChebySD", "Bundle-ChebySD", "General-ChebySD",
    "DiagSheafChebyshev", "BundleSheafChebyshev", "GeneralSheafChebyshev",
    # Your three extra models (if you want to emphasise them too)
    "EquiDiag", "EquiBundle", "EquiGeneral",
]

# Distinct markers for special series (Chebyshev + your 3 extras)
SPECIAL_MODELS: Dict[str, str] = {
    # Chebyshev (both naming styles)
    "Diag-ChebySD": "s", "Bundle-ChebySD": "D", "General-ChebySD": "^",
    "DiagSheafChebyshev": "s", "BundleSheafChebyshev": "D", "GeneralSheafChebyshev": "^",
    # Your three extra lines (optional)
    "EquiDiag": "s", "EquiBundle": "D", "EquiGeneral": "^",
}

# Figure style (don’t set explicit colors; matplotlib default is used)
FIGSIZE: Tuple[float, float] = (12, 16)  # 4 rows × 2 columns
Y_MIN, Y_MAX = 0, 100
Y_LABEL = "Test acc. (Upper is better)"
TITLE = "(Synthetic) Heterophily Experiment"

# Output
OUT_PNG = "visualizations/csv_files/Heterophily/heterophily_4x2.png"
OUT_PDF = "visualizations/csv_files/Heterophily/heterophily_4x2.pdf"

# =========================================================


def _to_float(x: Optional[str]) -> float:
    """Robust float parser: handles European decimals '0,25', strips %, thin spaces."""
    if x is None:
        return float("nan")
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    s = s.replace("%", "").replace("\u202f", "").replace(" ", "")
    if ("," in s) and ("." not in s):
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception as e:
        raise ValueError(f"Cannot parse float from '{x}'") from e


def _to_int_from_any(x: Optional[str]) -> int:
    """Accept '2', '2.0', '2,0' etc."""
    return int(round(_to_float(x)))


def _load_csv(path: str) -> pd.DataFrame:
    """
    Loads a CSV with columns:
      num_classes, het, model, test_acc, [std]
    - Handles both '.' and ',' decimal separators.
    - Fills missing 'std' with 0.0.
    """
    df = pd.read_csv(path, dtype=str).copy()

    need = {"num_classes", "het", "model", "test_acc"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    df["num_classes"] = df["num_classes"].apply(_to_int_from_any)
    df["het"]         = df["het"].apply(_to_float)
    df["model"]       = df["model"].astype(str).str.strip()
    df["test_acc"]    = df["test_acc"].apply(_to_float)

    if "std" in df.columns:
        df["std"] = df["std"].apply(_to_float)
    else:
        df["std"] = 0.0

    # Drop rows that fail parsing (NaNs in key columns)
    df = df.dropna(subset=["het", "test_acc"]).reset_index(drop=True)
    return df


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    # Keep the last occurrence for duplicate (num_classes, het, model)
    return (df
            .sort_values(["num_classes", "het", "model"])
            .drop_duplicates(subset=["num_classes", "het", "model"], keep="last")
            .reset_index(drop=True))


def _pick_rows(df: pd.DataFrame) -> List[int]:
    if FORCE_CLASS_ROWS:
        return FORCE_CLASS_ROWS
    rows = sorted(df["num_classes"].unique().tolist())
    if len(rows) < 4:
        print(f"[warn] Only {len(rows)} distinct num_classes found: {rows}")
    return rows[:4]


def _pick_hets(df: pd.DataFrame) -> List[float]:
    if FORCE_HETS:
        return FORCE_HETS
    hets = sorted(df["het"].unique().tolist())
    return hets  # use all distinct hets present


def _ordered_models(models: List[str]) -> List[str]:
    pref = [m for m in MODEL_ORDER if m in models]
    rest = sorted([m for m in models if m not in MODEL_ORDER])
    return pref + rest


def _draw_subplot(ax,
                  panel: pd.DataFrame,
                  hets: List[float]) -> Dict[str, object]:
    """
    Draw one curve per model (y = acc vs x = het) with error bars.
    Returns a dict {model_name: handle} for legend aggregation.
    """
    handles_by_model: Dict[str, object] = {}

    if panel.empty:
        ax.axis("off")
        return handles_by_model

    models = sorted(panel["model"].unique().tolist())
    order = _ordered_models(models)

    for m in order:
        sub = panel[panel["model"] == m].copy()
        if sub.empty:
            continue
        # align to hets list (drop missing hets gracefully)
        sub = sub.set_index("het").reindex(hets).reset_index()
        xs = sub["het"].values
        ys = sub["test_acc"].values
        es = sub["std"].values

        # Choose marker (special marker for Chebyshev / Equi*)
        marker = SPECIAL_MODELS.get(m, "o")

        # Plot with error bars; no explicit color
        h = ax.errorbar(xs, ys, yerr=es, fmt=f"{marker}-", linewidth=1, capsize=2, markersize=4)

        # store the first handle we create for each model
        if m not in handles_by_model:
            handles_by_model[m] = h

    ax.set_xlim(min(hets), max(hets))
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks(hets)
    ax.set_xlabel("heterophily (het)", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis='both', labelsize=8)

    return handles_by_model


def make_4x2(
    base_left: pd.DataFrame,
    cheby_left: pd.DataFrame,
    base_right: pd.DataFrame,
    cheby_right: pd.DataFrame,
):
    # Merge base and cheby per column, then dedup
    left_df  = _dedup(pd.concat([base_left,  cheby_left ], ignore_index=True))
    right_df = _dedup(pd.concat([base_right, cheby_right], ignore_index=True))

    # Determine rows and hets from union to keep axes consistent across columns
    rows = _pick_rows(pd.concat([left_df, right_df], ignore_index=True))
    hets = _pick_hets(pd.concat([left_df, right_df], ignore_index=True))

    # Fixed 4×2 canvas (blank rows if you supplied <4 class rows)
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=FIGSIZE, squeeze=False)

    # Column headers
    axes[0, 0].set_title(LEFT_LABEL, fontsize=11, pad=8)
    axes[0, 1].set_title(RIGHT_LABEL, fontsize=11, pad=8)

    # Row labels (num_classes)
    for i, nc in enumerate(rows):
        axes[i, 0].text(-0.18, 0.5, f"num classes={nc}",
                        transform=axes[i, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=10)

    # Collect one handle per model across all subplots
    legend_handles: Dict[str, object] = {}

    # Draw each available row for both columns
    for i, nc in enumerate(rows):
        left_panel  = left_df[left_df["num_classes"] == nc]
        right_panel = right_df[right_df["num_classes"] == nc]

        # Left subplot (RiSNN column)
        axl = axes[i, 0]
        hL = _draw_subplot(axl, left_panel, hets)
        for k, v in hL.items():
            legend_handles.setdefault(k, v)

        axl.set_ylabel(Y_LABEL, fontsize=9)

        # Right subplot (Diff column)
        axr = axes[i, 1]
        hR = _draw_subplot(axr, right_panel, hets)
        for k, v in hR.items():
            legend_handles.setdefault(k, v)

        axr.set_ylabel(Y_LABEL, fontsize=9)

    # Build a single, global legend (bottom, centered)
    if legend_handles:
        ordered_labels = _ordered_models(list(legend_handles.keys()))
        ordered_handles = [legend_handles[m] for m in ordered_labels]
        # place below both columns; adjust bottom space in tight_layout accordingly
        fig.legend(ordered_handles, ordered_labels,
                   loc="lower center", ncol=min(len(ordered_labels), 6),
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle(TITLE, fontsize=12, y=0.995)
    # Leave extra room at bottom for the legend row
    fig.tight_layout(rect=[0.04, 0.08, 0.995, 0.97])

    Path(OUT_PNG).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    print(f"Saved {OUT_PNG} and {OUT_PDF}")


def main():
    base_left   = _load_csv(LEFT_CSV)
    base_right  = _load_csv(RIGHT_CSV)
    cheby_left  = _load_csv(CHEBY_LEFT)
    cheby_right = _load_csv(CHEBY_RIGHT)

    make_4x2(base_left, cheby_left, base_right, cheby_right)


if __name__ == "__main__":
    main()
