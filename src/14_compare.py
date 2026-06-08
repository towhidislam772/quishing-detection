"""
Step 14 — Generate the clean-vs-robust comparison table + figure for §5 of
the paper. Combines results/adversarial_results.csv (clean models from step 9)
with results/adversarial_results_robust.csv (robust models from step 13).

Outputs:
  results/clean_vs_robust.csv   - paper-ready table
  figures/fig6_robustness.png   - grouped bar chart
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import load_config, get_logger

log = get_logger("compare")


def main():
    cfg     = load_config()
    results = Path(cfg["paths"]["results_dir"])
    figures = Path(cfg["paths"]["figures_dir"])
    figures.mkdir(parents=True, exist_ok=True)

    clean_csv  = results / "adversarial_results.csv"
    robust_csv = results / "adversarial_results_robust.csv"
    if not clean_csv.exists() or not robust_csv.exists():
        raise FileNotFoundError("Run steps 9 and 13 before step 14.")

    clean  = pd.read_csv(clean_csv)
    robust = pd.read_csv(robust_csv)

    # Normalise model names so clean+robust line up by variant
    clean["regime"]  = "clean"
    robust["regime"] = "robust"
    robust["model"]  = robust["model"].str.replace("_robust", "", regex=False)

    combo = pd.concat([clean, robust], ignore_index=True)
    pivot = combo.pivot_table(
        index=["variant", "model"],
        columns="regime",
        values="detection_rate@0.5",
        aggfunc="first",
    ).reset_index()
    pivot["delta"] = (pivot["robust"] - pivot["clean"]).round(4)
    pivot = pivot[["variant", "model", "clean", "robust", "delta"]]
    out_csv = results / "clean_vs_robust.csv"
    pivot.to_csv(out_csv, index=False)
    log.info(f"Comparison table:\n{pivot.to_string(index=False)}")
    log.info(f"Saved {out_csv}")

    # Grouped bar chart: rows=variants, groups=models, side-by-side clean/robust
    variants = ["split", "nested", "pdf"]
    models   = ["XGB", "CNN", "Hybrid"]
    width    = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, var in zip(axes, variants):
        sub = pivot[pivot["variant"] == var].set_index("model").reindex(models)
        x = np.arange(len(models))
        ax.bar(x - width/2, sub["clean"].values,  width, label="clean",
               color="#888888")
        ax.bar(x + width/2, sub["robust"].values, width, label="robust",
               color="#3a7bd5")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title(var)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Detection rate @ 0.5")
        for i, (c, r) in enumerate(zip(sub["clean"], sub["robust"])):
            if pd.notna(c): ax.text(i - width/2, c + 0.02, f"{c:.2f}", ha="center", fontsize=8)
            if pd.notna(r): ax.text(i + width/2, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
    axes[0].legend(loc="upper left")
    plt.tight_layout()
    out_png = figures / "fig6_robustness.png"
    out_pdf = figures / "fig6_robustness.pdf"
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()
    log.info(f"Saved {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
