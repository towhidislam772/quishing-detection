"""
Step 7 — Produce the publication figures: ROC, confusion matrix, feature
importance, ablation, model comparison. Saves PNG + PDF into figures/.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             precision_recall_fscore_support)
from utils import load_config, get_logger

log = get_logger("evaluate")

def savefig(fig, base):
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"),         bbox_inches="tight")
    plt.close(fig)

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

    # --- ROC ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in [("XGB (URL)", p_xgb), ("CNN (QR)", p_cnn), ("Hybrid", p_hyb)]:
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.4f})")
    ax.plot([0,1],[0,1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves"); ax.legend()
    savefig(fig, figdir / "fig1_roc_curve")

    # --- Confusion Matrix (hybrid) ---
    cm = confusion_matrix(y_te, (p_hyb > 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign","Malicious"],
                yticklabels=["Benign","Malicious"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Hybrid Confusion Matrix")
    savefig(fig, figdir / "fig2_confusion_matrix")

    # --- Feature Importance ---
    fi_path = results / "xgb_feature_importance.csv"
    if fi_path.exists():
        fi = pd.read_csv(fi_path, index_col=0).iloc[:, 0].sort_values(ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(fi.index, fi.values, color="steelblue")
        ax.set_xlabel("Importance"); ax.set_title("Top URL Features (XGBoost)")
        savefig(fig, figdir / "fig3_feature_importance")

    # --- Ablation ---
    rows = []
    for name, p in [("URL only (XGB)", p_xgb),
                    ("Image only (CNN)", p_cnn),
                    ("Hybrid", p_hyb)]:
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_te, (p > 0.5).astype(int), average="binary"
        )
        fpr, tpr, _ = roc_curve(y_te, p)
        rows.append({"variant":name, "precision":pr, "recall":rc,
                     "f1":f1, "auc":auc(fpr,tpr)})
    ab = pd.DataFrame(rows)
    ab.to_csv(results / "ablation.csv", index=False)
    fig, ax = plt.subplots(figsize=(7,4))
    ab.set_index("variant")[["precision","recall","f1","auc"]].plot.bar(ax=ax)
    ax.set_ylim(0,1.02); ax.set_ylabel("Score"); ax.set_title("Ablation: components vs. hybrid")
    plt.xticks(rotation=15)
    savefig(fig, figdir / "fig4_ablation")

    # --- Model comparison summary ---
    summary = pd.read_csv(results / "model_comparison.csv")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(summary["model"], summary["test_auc"], color=["#7BAFD4","#F2A359","#3A6B35"])
    for i,v in enumerate(summary["test_auc"]): ax.text(i, v+0.005, f"{v:.4f}", ha="center")
    ax.set_ylim(0.5,1.02); ax.set_ylabel("Test AUC"); ax.set_title("Model Comparison")
    savefig(fig, figdir / "fig5_model_comparison")

    log.info(f"All figures saved to {figdir}")

if __name__ == "__main__":
    main()
