"""
16_extra_analyses.py
====================
Post-hoc analyses requested in paper review.

1. Bootstrap 95% CIs for clean-test AUC (per branch, hybrid) and for
   adversarial detection rates (Wilson interval, small-N exact).
2. Feature ablation on the 28 URL features: re-fit XGBoost using
   top-5 / top-10 / top-15 / all-28 features (importance-ranked).

Writes:
    results/bootstrap_cis.csv
    results/wilson_cis_adversarial.csv
    results/feature_ablation.csv
"""

from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
MODELS = ROOT / "models"

RNG = np.random.default_rng(42)
N_BOOT = 2000


# ----------------------------------------------------------------------
# 1. Bootstrap CIs on clean test
# ----------------------------------------------------------------------
def bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray, n=N_BOOT, ci=0.95):
    n_samples = len(y_true)
    aucs = np.empty(n)
    for i in range(n):
        idx = RNG.integers(0, n_samples, n_samples)
        # skip degenerate resamples (one class)
        if len(np.unique(y_true[idx])) < 2:
            aucs[i] = np.nan
            continue
        aucs[i] = roc_auc_score(y_true[idx], y_score[idx])
    aucs = aucs[~np.isnan(aucs)]
    lo = np.percentile(aucs, (1 - ci) / 2 * 100)
    hi = np.percentile(aucs, (1 + ci) / 2 * 100)
    return float(np.mean(aucs)), float(lo), float(hi)


def bootstrap_f1(y_true, y_pred, n=N_BOOT, ci=0.95):
    n_samples = len(y_true)
    f1s = np.empty(n)
    for i in range(n):
        idx = RNG.integers(0, n_samples, n_samples)
        if len(np.unique(y_true[idx])) < 2:
            f1s[i] = np.nan
            continue
        f1s[i] = f1_score(y_true[idx], y_pred[idx])
    f1s = f1s[~np.isnan(f1s)]
    lo = np.percentile(f1s, (1 - ci) / 2 * 100)
    hi = np.percentile(f1s, (1 + ci) / 2 * 100)
    return float(np.mean(f1s)), float(lo), float(hi)


def run_bootstrap():
    df = pd.read_csv(DATA / "dataset_test.csv")
    y = df["label"].to_numpy()
    rows = []
    for label, fname in [
        ("XGB (URL)", "xgb_test_prob.npy"),
        ("CNN (image)", "cnn_test_prob.npy"),
        ("Hybrid", "hybrid_test_prob.npy"),
    ]:
        p = np.load(RESULTS / fname)
        m, lo, hi = bootstrap_auc(y, p)
        yp = (p >= 0.5).astype(int)
        f1_m, f1_lo, f1_hi = bootstrap_f1(y, yp)
        rows.append(dict(
            model=label,
            auc=roc_auc_score(y, p),
            auc_ci_lo=lo, auc_ci_hi=hi,
            f1=f1_score(y, yp),
            f1_ci_lo=f1_lo, f1_ci_hi=f1_hi,
        ))
    pd.DataFrame(rows).to_csv(RESULTS / "bootstrap_cis.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


# ----------------------------------------------------------------------
# 2. Wilson CIs for adversarial detection rates
# ----------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - halfw), min(1.0, centre + halfw)


def run_wilson():
    rows = []
    for which in ["adversarial_results.csv", "adversarial_results_robust.csv"]:
        df = pd.read_csv(RESULTS / which)
        regime = "clean" if "robust" not in which else "robust"
        for _, r in df.iterrows():
            n = int(r.get("n", 2000))
            rate = float(r["detection_rate@0.5"])
            k = int(round(rate * n))
            lo, hi = wilson(k, n)
            rows.append(dict(
                regime=regime,
                variant=r["variant"],
                model=r["model"].replace("_robust", ""),
                detection=rate,
                n=n,
                ci_lo=lo,
                ci_hi=hi,
            ))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "wilson_cis_adversarial.csv", index=False)
    print(df.to_string(index=False))


# ----------------------------------------------------------------------
# 3. Feature ablation on URL branch
# ----------------------------------------------------------------------
def run_feature_ablation():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    feat_imp = pd.read_csv(RESULTS / "xgb_feature_importance.csv")
    # CSV columns are ['Unnamed: 0', '0'] -> rename
    feat_imp.columns = ["feature", "importance"]
    feat_imp = feat_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    ranked = feat_imp["feature"].tolist()

    Xtr = pd.read_parquet(DATA / "url_features_train.parquet")
    Xva = pd.read_parquet(DATA / "url_features_val.parquet")
    Xte = pd.read_parquet(DATA / "url_features_test.parquet")
    ytr, yva, yte = Xtr.pop("label"), Xva.pop("label"), Xte.pop("label")

    rows = []
    for k in [5, 10, 15, 20, 28]:
        cols = ranked[:k]
        clf = xgb.XGBClassifier(
            n_estimators=cfg["xgb"]["n_estimators"],
            max_depth=cfg["xgb"]["max_depth"],
            learning_rate=cfg["xgb"]["learning_rate"],
            subsample=cfg["xgb"]["subsample"],
            colsample_bytree=cfg["xgb"]["colsample_bytree"],
            n_jobs=cfg["xgb"]["n_jobs"],
            eval_metric="auc",
            tree_method="hist",
            random_state=42,
        )
        clf.fit(Xtr[cols], ytr, eval_set=[(Xva[cols], yva)], verbose=False)
        p = clf.predict_proba(Xte[cols])[:, 1]
        yp = (p >= 0.5).astype(int)
        rows.append(dict(
            n_features=k,
            auc=roc_auc_score(yte, p),
            precision=precision_score(yte, yp),
            recall=recall_score(yte, yp),
            f1=f1_score(yte, yp),
        ))
        print(f"k={k:2d}  AUC={rows[-1]['auc']:.5f}  F1={rows[-1]['f1']:.5f}")

    pd.DataFrame(rows).to_csv(RESULTS / "feature_ablation.csv", index=False)


if __name__ == "__main__":
    print("=== bootstrap CIs ===")
    run_bootstrap()
    print("\n=== Wilson CIs (adversarial) ===")
    run_wilson()
    print("\n=== feature ablation ===")
    run_feature_ablation()
    print("\nDone.")
