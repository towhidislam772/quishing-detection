"""
Step 4 — Train XGBoost classifier on URL lexical features.
Saves the model + the OOF + val + test predicted probabilities (needed by hybrid).
"""
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from utils import load_config, get_logger

log = get_logger("xgb")

FEATURE_DROP = {"label", "qr_path", "url"}

def load_split(data, split):
    df = pd.read_parquet(data / f"url_features_{split}.parquet")
    X = df.drop(columns=[c for c in FEATURE_DROP if c in df.columns])
    y = df["label"].values
    return df, X, y

def main():
    cfg = load_config()
    data = cfg["paths"]["data_dir"]
    models = cfg["paths"]["models_dir"]
    results = cfg["paths"]["results_dir"]

    _, X_tr, y_tr = load_split(data, "train")
    _, X_va, y_va = load_split(data, "val")
    df_te, X_te, y_te = load_split(data, "test")

    log.info(f"Train={len(X_tr):,}  Val={len(X_va):,}  Test={len(X_te):,}  Features={X_tr.shape[1]}")

    xp = cfg["xgb"]
    clf = XGBClassifier(
        n_estimators=xp["n_estimators"],
        max_depth=xp["max_depth"],
        learning_rate=xp["learning_rate"],
        subsample=xp["subsample"],
        colsample_bytree=xp["colsample_bytree"],
        n_jobs=xp["n_jobs"],
        eval_metric="auc",
        tree_method="hist",
        random_state=cfg["dataset"]["random_state"],
    )
    log.info("Training XGBoost...")
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    p_va = clf.predict_proba(X_va)[:, 1]
    p_te = clf.predict_proba(X_te)[:, 1]
    log.info(f"Val AUC  = {roc_auc_score(y_va, p_va):.4f}")
    log.info(f"Test AUC = {roc_auc_score(y_te, p_te):.4f}")
    log.info("Test classification report:\n" +
             classification_report(y_te, (p_te > 0.5).astype(int), digits=4))

    joblib.dump(clf, models / "xgb_url.joblib")
    np.save(results / "xgb_val_prob.npy",  p_va)
    np.save(results / "xgb_test_prob.npy", p_te)

    # feature importance
    fi = pd.Series(clf.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    fi.to_csv(results / "xgb_feature_importance.csv")
    log.info("XGBoost training complete.")

if __name__ == "__main__":
    main()
