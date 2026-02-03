import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Iterable, Optional

# ===================== User controls =====================

# Group labels (rows 1–2 = left group/JdSNN; rows 3–4 = right group/RiSNN)
LEFT_GROUP_LABEL  = "JdSNN"   # a.k.a. Diff family
RIGHT_GROUP_LABEL = "RiSNN"

# CSVs
LEFT_BASE_CSV    = "visualizations/csv_files/Noise/Noise_Diff.csv"
LEFT_CHEBY_CSV   = "visualizations/csv_files/Noise/Noise_ChebyDiff.csv"
RIGHT_BASE_CSV   = "visualizations/csv_files/Noise/Noise_RiSNN.csv"
RIGHT_CHEBY_CSV  = "visualizations/csv_files/Noise/Noise_ChebyRiSNN.csv"

# EXACTLY 6 noise levels per group (2 rows × 3 cols). If empty, we infer and take the first 6 sorted.
FORCE_NOISE_LEFT:  List[float] = []  # e.g. [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
FORCE_NOISE_RIGHT: List[float] = []  # e.g. [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]

# Model ordering (legend & x-tick order; others appended alphabetically).
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

# Pretty labels (legend & x-ticks only)
RENAME_MODELS: Dict[str, str] = {
    "JointSheaf NoInit": "JdSNN-NoInit",
    "JointSheaf Init":   "JdSNN-Init",
    "JointSheaf Params": "JdSNN-Params",
    "RotInvSheaf NoTime": "RiSNN-NoT",
    "RotInvSheaf Time":   "RiSNN-T",
    "DiagSheafChebyshev":    "DiagChebySD",
    "BundleSheafChebyshev":  "BundleChebySD",
    "GeneralSheafChebyshev": "GeneralChebySD",
}

# Markers for special models (others use 'o')
SPECIAL_MODELS: Dict[str, str] = {
    "DiagSheafChebyshev": "s",
    "BundleSheafChebyshev": "D",
    "GeneralSheafChebyshev": "^",
}

# Axes / layout
Y_MIN, Y_MAX = 0, 100
Y_LABEL = "Test acc. (mean ± std)"
SUPTITLE = "(Synthetic) Noise Effect Ablation Study"

# Output
OUT_PNG = "visualizations/csv_files/Noise/effect_of_noise_4x3.png"
OUT_PDF = "visualizations/csv_files/Noise/effect_of_noise_4x3.pdf"

# =========================================================


def _to_float(x: Optional[str]) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    s = s.replace("%", "").replace("\u202f", "").replace(" ", "")
    if ("," in s) and ("." not in s):
        s = s.replace(",", ".")
    return float(s)


def _load_noise_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    cols = set(df.columns)

    if "feat_noise" not in cols and "noise" in cols:
        df.rename(columns={"noise": "feat_noise"}, inplace=True)
    if "test_acc" not in cols and "acc" in cols:
        df.rename(columns={"acc": "test_acc"}, inplace=True)

    need = {"model", "feat_noise", "test_acc"}
    if not need.issubset(set(df.columns)):
        missing = need - set(df.columns)
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    if "test_acc_std" not in df.columns:
        if "std" in df.columns:
            df["test_acc_std"] = df["std"]
        elif "acc_std" in df.columns:
            df["test_acc_std"] = df["acc_std"]
        else:
            df["test_acc_std"] = 0.0

    df["feat_noise"]   = df["feat_noise"].apply(_to_float)
    df["test_acc"]     = df["test_acc"].apply(_to_float)
    df["test_acc_std"] = df["test_acc_std"].apply(_to_float)
    df["model"]        = df["model"].astype(str).str.strip()

    df = df.dropna(subset=["feat_noise", "test_acc"]).reset_index(drop=True)
    df = (df.sort_values(["model", "feat_noise"])
            .drop_duplicates(subset=["model", "feat_noise"], keep="last")
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


def _panel(ax, df: pd.DataFrame, noise: float,
           model_order: List[str], color_map: Dict[str, str]) -> Dict[str, object]:
    sub = df[df["feat_noise"] == noise].copy()
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

        h = ax.errorbar([i], [y], yerr=[e],
                        fmt=marker, capsize=3, markersize=5,
                        linestyle="none",
                        color=color_map.get(m),
                        label=label_txt)
        handles.setdefault(m, h)

    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_ylabel(Y_LABEL, fontsize=9)
    xtxt = [RENAME_MODELS.get(m, m) for m in models]
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(xtxt, rotation=60, ha="right", fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_xlabel("Models", fontsize=9)
    ax.set_title(f"{noise}", fontsize=10, pad=6)
    return handles


def _select_six(noises: List[float], forced: List[float]) -> List[float]:
    if forced:
        return forced[:6]
    noises = sorted(noises)
    return noises[:6]  # truncate if more than 6 exist


def make_4x3(left_df: pd.DataFrame, right_df: pd.DataFrame,
             left_order: List[str], right_order: List[str],
             left_label: str, right_label: str,
             suptitle: str, out_png: str, out_pdf: str,
             force_noise_left: List[float], force_noise_right: List[float]):

    left_noises  = _select_six(left_df["feat_noise"].unique().tolist(),  force_noise_left)
    right_noises = _select_six(right_df["feat_noise"].unique().tolist(), force_noise_right)

    # consistent colors across BOTH groups
    all_models = sorted(set(left_df["model"]).union(set(right_df["model"])))
    legend_order = _ordered(all_models, left_order + [m for m in right_order if m not in left_order])
    color_map = _build_color_map(legend_order)

    # >>> Bigger/taller figure + shared y to avoid visual "shrink"
    FIG_WIDTH, FIG_HEIGHT = 12.5, 16.5   # was ~12×12; increase height for paper readability
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(FIG_WIDTH, FIG_HEIGHT), sharey=True)
    axes = axes.reshape(4, 3)

    # map the 6 panels of each group across 2 rows × 3 columns
    left_axes  = [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1], axes[1,2]]
    right_axes = [axes[2,0], axes[2,1], axes[2,2], axes[3,0], axes[3,1], axes[3,2]]

    legend_handles: Dict[str, object] = {}

    # LEFT GROUP (JdSNN/Diff)
    for ax, noise in zip(left_axes, left_noises):
        h = _panel(ax, left_df, noise, left_order, color_map)
        for k, v in h.items():
            legend_handles.setdefault(k, v)
        ax.set_ylim(Y_MIN, Y_MAX)  # enforce full scale

    # group label
    axes[0,0].text(-0.12, 1.15, left_label, transform=axes[0,0].transAxes,
                   fontsize=11, fontweight="bold", ha="left", va="bottom")

    # RIGHT GROUP (RiSNN)
    for ax, noise in zip(right_axes, right_noises):
        h = _panel(ax, right_df, noise, right_order, color_map)
        for k, v in h.items():
            legend_handles.setdefault(k, v)
        ax.set_ylim(Y_MIN, Y_MAX)  # enforce full scale

    axes[2,0].text(-0.12, 1.15, right_label, transform=axes[2,0].transAxes,
                   fontsize=11, fontweight="bold", ha="left", va="bottom")

    # global legend at bottom
    ordered_labels  = [RENAME_MODELS.get(m, m) for m in legend_order if m in legend_handles]
    ordered_handles = [legend_handles[m] for m in legend_order if m in legend_handles]
    fig.legend(ordered_handles, ordered_labels,
               loc="lower center", ncol=min(len(ordered_labels), 6),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle(suptitle, fontsize=12, y=0.995)
    # extra bottom room for legend; a bit more top/bottom breathing space now that it’s taller
    fig.tight_layout(rect=[0.04, 0.08, 0.995, 0.965])

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print(f"[saved] {out_png}\n[saved] {out_pdf}")


def main():
    # load & combine per group
    left_base   = _load_noise_csv(LEFT_BASE_CSV)
    left_cheby  = _load_noise_csv(LEFT_CHEBY_CSV)
    right_base  = _load_noise_csv(RIGHT_BASE_CSV)
    right_cheby = _load_noise_csv(RIGHT_CHEBY_CSV)

    left_df  = pd.concat([left_base,  left_cheby ], ignore_index=True)
    right_df = pd.concat([right_base, right_cheby], ignore_index=True)

    make_4x3(
        left_df=left_df, right_df=right_df,
        left_order=MODEL_ORDER_LEFT, right_order=MODEL_ORDER_RIGHT,
        left_label=LEFT_GROUP_LABEL, right_label=RIGHT_GROUP_LABEL,
        suptitle=SUPTITLE, out_png=OUT_PNG, out_pdf=OUT_PDF,
        force_noise_left=FORCE_NOISE_LEFT, force_noise_right=FORCE_NOISE_RIGHT,
    )


if __name__ == "__main__":
    main()
