"""
Step 18 - Precision-Recall curve sweep across thresholds.
Uses already-saved test-set predictions for XGB, CNN, Hybrid.
Highlights tau=0.5 (default) and tau=0.9 (high-precision production).
Writes:
  figures/fig_pr_sweep.{png,pdf}
  results/pr_at_tau.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             precision_score, recall_score, f1_score)
from utils import load_config, get_logger

log = get_logger("pr_sweep")

TAUS = [0.5, 0.9]

def metrics_at_tau(y, p, tau):
    yhat = (p >= tau).astype(int)
    return (precision_score(y, yhat, zero_division=0),
            recall_score(y, yhat, zero_division=0),
            f1_score(y, yhat, zero_division=0))

def main():
    cfg = load_config()
    data = cfg["paths"]["data_dir"]
    results = cfg["paths"]["results_dir"]
    figdir = Path(__file__).resolve().parent.parent / "figures"
    figdir.mkdir(exist_ok=True)

    y_te = pd.read_parquet(data / "url_features_test.parquet")["label"].values
    p_xgb = np.load(results / "xgb_test_prob.npy")
    p_cnn = np.load(results / "cnn_test_prob.npy")
    p_hyb = np.load(results / "hybrid_test_prob.npy")

    models = [
        ("XGB (URL)",  p_xgb, "#7BAFD4"),
        ("CNN (QR)",   p_cnn, "#F2A359"),
        ("Hybrid",     p_hyb, "#3A6B35"),
    ]

    fig, ax = plt.subplots(figsize=(7, 5.2))
    rows = []

    for name, p, color in models:
        prec, rec, thr = precision_recall_curve(y_te, p)
        ap = average_precision_score(y_te, p)
        ax.plot(rec, prec, color=color, lw=2.0, label=f"{name} (AP={ap:.4f})")

        for tau in TAUS:
            pr, rc, f1 = metrics_at_tau(y_te, p, tau)
            ax.scatter([rc], [pr], color=color, marker="o" if tau == 0.5 else "s",
                       s=80, edgecolor="black", linewidth=0.8, zorder=5)
            ax.annotate(rf"$\tau$={tau}",
                        (rc, pr), xytext=(6, -10), textcoords="offset points",
                        fontsize=8)
            rows.append({"model": name, "tau": tau,
                         "precision": pr, "recall": rc, "f1": f1})
            log.info(f"{name:<12s} tau={tau}  P={pr:.4f}  R={rc:.4f}  F1={f1:.4f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.005)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(r"Precision-Recall curves with $\tau=0.5$ ($\bullet$) and $\tau=0.9$ ($\blacksquare$)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    out_png = figdir / "fig_pr_sweep.png"
    out_pdf = figdir / "fig_pr_sweep.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(results / "pr_at_tau.csv", index=False)
    log.info(f"Saved {out_png}, {out_pdf}, and pr_at_tau.csv")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
