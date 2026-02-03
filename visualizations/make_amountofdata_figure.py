import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Iterable, Optional

# ===================== User controls =====================

LEFT_GROUP_LABEL  = "JdSNN"   # Diff family
RIGHT_GROUP_LABEL = "RiSNN"

# CSVs
LEFT_BASE_CSV    = "visualizations/csv_files/AmountData/Data_Diff.csv"
LEFT_CHEBY_CSV   = "visualizations/csv_files/AmountData/Data_ChebyDiff.csv"
RIGHT_BASE_CSV   = "visualizations/csv_files/AmountData/Data_RiSNN.csv"
RIGHT_CHEBY_CSV  = "visualizations/csv_files/AmountData/Data_ChebyRiSNN.csv"

# Use 3 rows per family (-> 6×3 grid)
LEFT_NODES:  List[int] = [100, 500, 1000]
RIGHT_NODES: List[int] = [100, 500, 1000]
DEGREES:     List[int] = [2, 6, 10]   # 3 columns

# Model ordering
MODEL_ORDER_LEFT: List[str] = [
    "MLP", "VanillaSheaf", "DiagSheaf",
    "JointSheaf NoInit", "JointSheaf Init", "JointSheaf Params",
    "DiagSheafChebyshev", "BundleSheafChebyshev", "GeneralSheafChebyshev",
]
MODEL_ORDER_RIGHT: List[str] = [
    "MLP", "VanillaSheaf", "DiagSheaf",
    "RotInvSheaf NoTime", "RotInvSheaf Time",
    "DiagSheafChebyshev", "BundleSheafChebyshev", "GeneralSheafChebyshev",
]

# Pretty renames (ticks/legend only)
RENAME_MODELS: Dict[str, str] = {
    "JointSheaf NoInit": "JdSNN-NoInit",
    "JointSheaf Init":   "JdSNN-Init",
    "JointSheaf Params": "JdSNN-Params",
    "RotInvSheaf NoTime": "RiSNN-NoT",
    "RotInvSheaf Time":   "RiSNN-T",
    "DiagSheafChebyshev":    "Diag-ChebySD",
    "BundleSheafChebyshev":  "Bundle-ChebySD",
    "GeneralSheafChebyshev": "General-ChebySD",
}

# Special markers for PolySD; others use 'o'
SPECIAL_MODELS: Dict[str, str] = {
    "DiagSheafChebyshev": "s",
    "BundleSheafChebyshev": "D",
    "GeneralSheafChebyshev": "^",
}

# Axes / layout
Y_MIN, Y_MAX = 0, 100
Y_LABEL  = "Test acc. (mean ± std)"
SUPTITLE = "(Synthetic) Data Scalability Ablation Study"

# >>> Make panels taller + bigger spacing to avoid overlaps
PANEL_W = 3.5     # inches per col
PANEL_H = 3.1     # inches per row  (↑ was ~2.5)
WSPACE  = 0.70    # horizontal spacing between panels
HSPACE  = 1.20    # vertical spacing between panels
ROW_HEADER_Y = 1.14  # where the row label sits above each row (in axes coords)

# Output
OUT_PNG = "visualizations/csv_files/AmountData/amount_of_data_6x3_wideY.png"
OUT_PDF = "visualizations/csv_files/AmountData/amount_of_data_6x3_wideY.pdf"

# =========================================================


def _to_float(x: Optional[str]) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip().replace("%", "").replace("\u202f", "").replace(" ", "")
    if ("," in s) and ("." not in s):
        s = s.replace(",", ".")
    return float(s) if s not in ("", "nan", "None") else float("nan")


def _load_any(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)

    # degree harmonization
    if "degree" not in df.columns and "node_degree" in df.columns:
        df.rename(columns={"node_degree": "degree"}, inplace=True)

    # test acc/std harmonization
    if "test_acc" not in df.columns and "Test acc" in df.columns:
        df["test_acc"] = df["Test acc"]
    if "test_acc_std" not in df.columns:
        if "std" in df.columns:
            df["test_acc_std"] = df["std"]
        elif "acc_std" in df.columns:
            df["test_acc_std"] = df["acc_std"]
        else:
            df["test_acc_std"] = 0.0

    need = {"model", "num_nodes", "test_acc", "test_acc_std"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    if "degree" not in df.columns:
        df["degree"] = pd.NA

    def _parse_degree(v):
        s = "" if v is None else str(v).strip()
        if s == "" or s.lower() in ("nan", "none"):
            return float("nan")
        return int(round(_to_float(s)))

    # types
    df["num_nodes"]    = df["num_nodes"].apply(lambda v: int(round(_to_float(v))))
    df["degree"]       = df["degree"].apply(_parse_degree)
    df["test_acc"]     = df["test_acc"].apply(_to_float)
    df["test_acc_std"] = df["test_acc_std"].apply(_to_float)
    df["model"]        = df["model"].astype(str).str.strip()

    df = df.dropna(subset=["test_acc"]).reset_index(drop=True)

    df = (df.sort_values(["num_nodes", "degree", "model"])
            .drop_duplicates(subset=["num_nodes", "degree", "model"], keep="last")
            .reset_index(drop=True))
    return df


def _ordered(models: Iterable[str], preferred: List[str]) -> List[str]:
    models = list(models)
    pref = [m for m in preferred if m in models]
    rest = sorted([m for m in models if m not in preferred])
    return pref + rest


def _build_color_map(model_names: List[str]) -> Dict[str, str]:
    palette = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
    )
    mapping, k = {}, 0
    for m in model_names:
        mapping[m] = palette[k % len(palette)]
        k += 1
    return mapping


def _panel(ax, df: pd.DataFrame, node_count: int, degree: int,
           model_order: List[str], color_map: Dict[str, str]) -> Dict[str, object]:
    # Prefer exact (num_nodes, degree)
    sub = df[(df["num_nodes"] == node_count) & (df["degree"] == degree)].copy()
    # Fallback for CSVs lacking degree
    if sub.empty:
        sub = df[(df["num_nodes"] == node_count) & (df["degree"].isna())].copy()

    if sub.empty:
        ax.axis("off")
        return {}

    models_present = sorted(sub["model"].unique().tolist())
    models = _ordered(models_present, model_order)

    handles = {}
    for i, m in enumerate(models):
        r = sub[sub["model"] == m]
        if r.empty:
            continue
        r = r.iloc[-1]
        y, e = r["test_acc"], r["test_acc_std"]
        marker = SPECIAL_MODELS.get(m, "o")
        label_txt = RENAME_MODELS.get(m, m)

        h = ax.errorbar(
            [i], [y], yerr=[e],
            fmt=marker, linestyle="none",
            capsize=3, markersize=5,
            color=color_map.get(m, None),
            label=label_txt,
        )
        handles.setdefault(m, h)

    # Larger vertical real estate; shared y stays 0–100
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_ylabel(Y_LABEL, fontsize=11)
    xtxt = [RENAME_MODELS.get(m, m) for m in models]
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(xtxt, rotation=60, ha="right", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_xlabel("model", fontsize=11)
    return handles


def make_amount_of_data(
    left_df: pd.DataFrame, right_df: pd.DataFrame,
    left_nodes: List[int], right_nodes: List[int], degrees: List[int],
    left_order: List[str], right_order: List[str],
    left_label: str, right_label: str,
    suptitle: str, out_png: str, out_pdf: str
):
    all_models = sorted(set(left_df["model"]).union(set(right_df["model"])))
    legend_order = _ordered(all_models, left_order + [m for m in right_order if m not in left_order])
    color_map = _build_color_map(legend_order)

    n_cols = len(degrees)                         # 3
    n_rows = len(left_nodes) + len(right_nodes)   # 6

    fig_w = n_cols * PANEL_W + 1.4
    fig_h = n_rows * PANEL_H + 3.2   # extra bottom room for legend
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(fig_w, fig_h),
        sharey=True,
        gridspec_kw={"wspace": WSPACE, "hspace": HSPACE},
    )
    axes = axes.reshape(n_rows, n_cols)

    legend_handles: Dict[str, object] = {}

    # ----- JdSNN (top three rows) -----
    for r, nodes in enumerate(left_nodes):
        for c, deg in enumerate(degrees):
            h = _panel(axes[r, c], left_df, nodes, deg, left_order, color_map)
            for k, v in h.items():
                legend_handles.setdefault(k, v)
        # Row header above the row (no overlap thanks to HSPACE & ROW_HEADER_Y)
        axes[r, 0].text(-0.10, ROW_HEADER_Y, f"{left_label}, num_nodes = {nodes}",
                        transform=axes[r, 0].transAxes, fontsize=12,
                        fontweight="bold", ha="left", va="bottom")

    # ----- RiSNN (bottom three rows) -----
    offset = len(left_nodes)
    for i, nodes in enumerate(right_nodes):
        r = offset + i
        for c, deg in enumerate(degrees):
            h = _panel(axes[r, c], right_df, nodes, deg, right_order, color_map)
            for k, v in h.items():
                legend_handles.setdefault(k, v)
        axes[r, 0].text(-0.10, ROW_HEADER_Y, f"{right_label}, num_nodes = {nodes}",
                        transform=axes[r, 0].transAxes, fontsize=12,
                        fontweight="bold", ha="left", va="bottom")

    # column headers (degrees) on the top row only
    for j, deg in enumerate(degrees):
        axes[0, j].set_title(f"{deg}", fontsize=12, pad=10)

    # global legend at bottom (no overlap)
    ordered_labels  = [RENAME_MODELS.get(m, m) for m in legend_order if m in legend_handles]
    ordered_handles = [legend_handles[m] for m in legend_order if m in legend_handles]
    fig.legend(ordered_handles, ordered_labels,
               loc="lower center", ncol=min(len(ordered_labels), 7),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle(suptitle, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0.05, 0.09, 0.995, 0.963])

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[saved] {out_png}\n[saved] {out_pdf}")


def main():
    left_base   = _load_any(LEFT_BASE_CSV)
    left_cheby  = _load_any(LEFT_CHEBY_CSV)
    right_base  = _load_any(RIGHT_BASE_CSV)
    right_cheby = _load_any(RIGHT_CHEBY_CSV)

    left_df  = pd.concat([left_base,  left_cheby ], ignore_index=True)
    right_df = pd.concat([right_base, right_cheby], ignore_index=True)

    make_amount_of_data(
        left_df=left_df, right_df=right_df,
        left_nodes=LEFT_NODES, right_nodes=RIGHT_NODES, degrees=DEGREES,
        left_order=MODEL_ORDER_LEFT, right_order=MODEL_ORDER_RIGHT,
        left_label=LEFT_GROUP_LABEL, right_label=RIGHT_GROUP_LABEL,
        suptitle=SUPTITLE, out_png=OUT_PNG, out_pdf=OUT_PDF,
    )


if __name__ == "__main__":
    main()
