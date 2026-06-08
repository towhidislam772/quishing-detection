"""
Step 19 - Combine fig5 (model AUC comparison) and fig6 (clean vs robust
adversarial detection) into a single 2-panel figure for the paper.
Writes:
  figures/fig5_6_combined.{png,pdf}
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import load_config, get_logger

log = get_logger("fig5_6")

VARIANTS = ["split", "nested", "pdf"]
MODELS   = ["XGB", "CNN", "Hybrid"]
COLORS   = {"XGB": "#7BAFD4", "CNN": "#F2A359", "Hybrid": "#3A6B35"}

def main():
    cfg = load_config()
    results = cfg["paths"]["results_dir"]
    figdir = Path(__file__).resolve().parent.parent / "figures"
    figdir.mkdir(exist_ok=True)

    cmp_df = pd.read_csv(results / "model_comparison.csv")
    cvr    = pd.read_csv(results / "clean_vs_robust.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6),
                             gridspec_kw={"width_ratios": [1.0, 1.6]})

    # ---------- Panel (a): model AUC comparison ----------
    ax = axes[0]
    bar_colors = ["#7BAFD4", "#F2A359", "#3A6B35"]
    bars = ax.bar(cmp_df["model"], cmp_df["test_auc"],
                  color=bar_colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, cmp_df["test_auc"]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylim(0.5, 1.03)
    ax.set_ylabel("Test AUC")
    ax.set_title("(a) Model Comparison (clean test, $n=40{,}230$)")
    ax.tick_params(axis="x", rotation=10)
    ax.grid(True, axis="y", alpha=0.3)

    # ---------- Panel (b): clean vs robust adversarial detection ----------
    ax = axes[1]
    n_var, n_mod = len(VARIANTS), len(MODELS)
    group_width = 0.85
    pair_width  = group_width / n_mod          # width occupied by (clean+robust) per model
    bar_width   = pair_width * 0.45            # each individual bar
    gap         = pair_width - 2*bar_width     # space between clean & robust within a model
    x_centers   = np.arange(n_var)

    for j, model in enumerate(MODELS):
        sub = cvr[cvr["model"] == model].set_index("variant")
        clean_vals  = [sub.loc[v, "clean"]  for v in VARIANTS]
        robust_vals = [sub.loc[v, "robust"] for v in VARIANTS]
        # Position: center group at x_centers[i]; offset by model index j
        offset = (j - (n_mod-1)/2) * pair_width
        x_clean  = x_centers + offset - (bar_width/2 + gap/2)
        x_robust = x_centers + offset + (bar_width/2 + gap/2)

        ax.bar(x_clean,  clean_vals,  width=bar_width, color=COLORS[model],
               alpha=0.45, edgecolor="black", linewidth=0.5,
               label=f"{model} (clean)" if j == 0 else None)
        ax.bar(x_robust, robust_vals, width=bar_width, color=COLORS[model],
               alpha=1.0, edgecolor="black", linewidth=0.5,
               label=f"{model} (robust)" if j == 0 else None)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([v.capitalize() for v in VARIANTS])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Detection rate (threshold 0.5)")
    ax.set_title("(b) Adversarial Detection: Clean vs Robust ($n=50{,}000$ / variant)")
    ax.grid(True, axis="y", alpha=0.3)

    # Custom legend: light = clean, dark = robust, color = model
    legend_elems = []
    for m in MODELS:
        legend_elems.append(plt.Rectangle((0,0),1,1, color=COLORS[m], alpha=0.45,
                                          label=f"{m} clean"))
        legend_elems.append(plt.Rectangle((0,0),1,1, color=COLORS[m], alpha=1.0,
                                          label=f"{m} robust"))
    ax.legend(handles=legend_elems, loc="lower right", fontsize=7.5,
              ncol=3, frameon=True)

    fig.tight_layout()
    out_png = figdir / "fig5_6_combined.png"
    out_pdf = figdir / "fig5_6_combined.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out_png}")
    log.info(f"Saved {out_pdf}")

if __name__ == "__main__":
    main()
