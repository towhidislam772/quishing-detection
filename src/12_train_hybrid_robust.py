"""
Step 12 — Train the robust hybrid meta-classifier.

Uses:
  - the existing XGB val/test probabilities (URL channel is unchanged)
  - the NEW robust CNN val/test probabilities from step 11
The meta-learner re-learns how much to trust each channel given the robust
CNN. Saved as hybrid_meta_robust.joblib so step 13 can load it.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)
from xgboost import XGBClassifier
from utils import load_config, get_logger

log = get_logger("hybrid_robust")


def main():
    cfg     = load_config()
    data    = cfg["paths"]["data_dir"]
    models_ = cfg["paths"]["models_dir"]
    results = cfg["paths"]["results_dir"]

    y_va = pd.read_parquet(data / "url_features_val.parquet")["label"].values
    y_te = pd.read_parquet(data / "url_features_test.parquet")["label"].values

    xgb_va = np.load(results / "xgb_val_prob.npy")
    xgb_te = np.load(results / "xgb_test_prob.npy")
    cnn_va = np.load(results / "cnn_robust_val_prob.npy")
    cnn_te = np.load(results / "cnn_robust_test_prob.npy")

    Xv = np.column_stack([xgb_va, cnn_va])
    Xt = np.column_stack([xgb_te, cnn_te])

    if cfg["hybrid"]["meta"] == "xgb_small":
        meta = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                             eval_metric="auc", tree_method="hist",
                             random_state=cfg["dataset"]["random_state"])
    else:
        meta = LogisticRegression(max_iter=1000)

    meta.fit(Xv, y_va)
    p_te = meta.predict_proba(Xt)[:, 1]

    auc = roc_auc_score(y_te, p_te)
    log.info(f"Robust Hybrid Test AUC = {auc:.4f}")
    log.info(f"  XGB-only Test AUC          = {roc_auc_score(y_te, xgb_te):.4f}")
    log.info(f"  Robust CNN Test AUC        = {roc_auc_score(y_te, cnn_te):.4f}")
    log.info("Robust hybrid classification report:\n" +
             classification_report(y_te, (p_te > 0.5).astype(int), digits=4))
    cm = confusion_matrix(y_te, (p_te > 0.5).astype(int))
    log.info(f"Confusion matrix:\n{cm}")

    joblib.dump(meta, models_ / "hybrid_meta_robust.joblib")
    np.save(results / "hybrid_robust_test_prob.npy", p_te)

    fpr, tpr, _ = roc_curve(y_te, p_te)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        results / "roc_points_robust.csv", index=False)

    summary = pd.DataFrame({
        "model":    ["XGB (URL)", "CNN robust", "Hybrid robust"],
        "test_auc": [roc_auc_score(y_te, xgb_te),
                     roc_auc_score(y_te, cnn_te),
                     auc],
    })
    summary.to_csv(results / "model_comparison_robust.csv", index=False)
    log.info("Robust hybrid stacking complete.")


if __name__ == "__main__":
    main()
