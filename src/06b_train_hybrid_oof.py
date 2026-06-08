"""
Step 6b - Hybrid meta-learner trained on OUT-OF-FOLD base predictions.

Replaces 06_train_hybrid.py and fixes the data-leakage problem: the old
script fit the LR meta on the validation set, but XGB & CNN had already
early-stopped on that same val set, so the meta was optimistic by
construction.

The new flow:
  1. K-fold the TRAIN set.
  2. For each fold, fit XGB + CNN on the other K-1 folds, predict on the
     held-out fold => out-of-fold (OOF) probabilities covering the whole
     train set, none of which the base learners ever saw.
  3. Fit the LR meta on those OOF train probs (leakage-free).
  4. Refit XGB + CNN on the FULL train set (early-stopping on val is OK
     for the base learners; only the meta needed leakage-free signal).
  5. Use those full-train models to score val + test.
  6. Run meta on (xgb_test_prob, cnn_test_prob) => reported hybrid probs.

Inputs : data/url_features_{train,val,test}.parquet
Outputs:
  results/xgb_oof_train_prob.npy
  results/cnn_oof_train_prob.npy
  results/xgb_val_prob.npy, xgb_test_prob.npy   (overwritten)
  results/cnn_val_prob.npy, cnn_test_prob.npy   (overwritten)
  results/hybrid_val_prob.npy                   (new)
  results/hybrid_test_prob.npy                  (overwritten)
  results/model_comparison.csv                  (overwritten)
  models/xgb_url.joblib, cnn_qr.keras, cnn_qr.tflite (overwritten)
  models/hybrid_meta_oof.joblib                 (new)
"""
from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from keras import layers, models, callbacks, metrics as keras_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

from utils import load_config, get_logger

log = get_logger("hybrid_oof")
AUTOTUNE = tf.data.AUTOTUNE
K = 5                       # number of folds for OOF
FEATURE_DROP = {"label", "qr_path", "url"}


# ---------------------------------------------------------------------------
# XGB helpers
# ---------------------------------------------------------------------------
def _new_xgb(cfg):
    xp = cfg["xgb"]
    return XGBClassifier(
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


# ---------------------------------------------------------------------------
# CNN helpers (mirrors 05_train_cnn.py but parameterised)
# ---------------------------------------------------------------------------
def _make_ds(paths, labels, size, batch, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(min(len(paths), 50000), seed=42,
                        reshuffle_each_iteration=True)

    def _load(path, y):
        raw = tf.io.read_file(path)
        img = tf.io.decode_png(raw, channels=1)
        img = tf.image.resize(img, [size, size], method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        return img, y

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return ds


def _build_cnn(size, chans):
    inp = layers.Input(shape=(size, size, chans))
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=[keras_metrics.AUC(name="auc"),
                       keras_metrics.BinaryAccuracy(name="acc")])
    return m


def _fit_cnn_fold(p_tr, y_tr, p_val_fold, y_val_fold, cfg):
    """Train a CNN with early stopping on the held-out fold."""
    size = cfg["image"]["size"]
    chans = cfg["image"]["channels"]
    batch = cfg["cnn"]["batch_size"]
    epochs = cfg["cnn"]["epochs"]
    pat = cfg["cnn"]["early_stop_patience"]

    train_ds = _make_ds(p_tr, y_tr, size, batch, shuffle=True)
    val_ds = _make_ds(p_val_fold, y_val_fold, size, batch, shuffle=False)
    model = _build_cnn(size, chans)
    cb = [
        callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                patience=pat, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=2),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs,
              callbacks=cb, verbose=0)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = load_config()
    data = cfg["paths"]["data_dir"]
    models_ = cfg["paths"]["models_dir"]
    results = cfg["paths"]["results_dir"]
    rs = cfg["dataset"]["random_state"]

    # ------- Load splits (URL features + image paths in same parquet) -----
    df_tr = pd.read_parquet(data / "url_features_train.parquet")
    df_va = pd.read_parquet(data / "url_features_val.parquet")
    df_te = pd.read_parquet(data / "url_features_test.parquet")

    y_tr = df_tr["label"].to_numpy()
    y_va = df_va["label"].to_numpy()
    y_te = df_te["label"].to_numpy()

    feat_cols = [c for c in df_tr.columns if c not in FEATURE_DROP]
    X_tr = df_tr[feat_cols].to_numpy()
    X_va = df_va[feat_cols].to_numpy()
    X_te = df_te[feat_cols].to_numpy()

    img_tr = df_tr["qr_path"].to_numpy()
    img_va = df_va["qr_path"].to_numpy()
    img_te = df_te["qr_path"].to_numpy()

    log.info(f"Train={len(y_tr):,}  Val={len(y_va):,}  Test={len(y_te):,}  "
             f"Features={len(feat_cols)}")

    # ------- K-fold OOF -----------------------------------------------------
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=rs)
    xgb_oof = np.zeros(len(y_tr), dtype=np.float32)
    cnn_oof = np.zeros(len(y_tr), dtype=np.float32)

    for fold, (idx_in, idx_out) in enumerate(skf.split(X_tr, y_tr), 1):
        log.info(f"=== fold {fold}/{K}  "
                 f"(train={len(idx_in):,}  oof={len(idx_out):,}) ===")

        # ---- XGB fold ----
        xgb_fold = _new_xgb(cfg)
        xgb_fold.fit(X_tr[idx_in], y_tr[idx_in],
                     eval_set=[(X_tr[idx_out], y_tr[idx_out])],
                     verbose=False)
        xgb_oof[idx_out] = xgb_fold.predict_proba(X_tr[idx_out])[:, 1]
        log.info(f"  fold {fold} XGB OOF AUC = "
                 f"{roc_auc_score(y_tr[idx_out], xgb_oof[idx_out]):.4f}")

        # ---- CNN fold ----
        cnn_fold = _fit_cnn_fold(img_tr[idx_in], y_tr[idx_in].astype("float32"),
                                 img_tr[idx_out], y_tr[idx_out].astype("float32"),
                                 cfg)
        oof_ds = _make_ds(img_tr[idx_out], y_tr[idx_out].astype("float32"),
                          cfg["image"]["size"], cfg["cnn"]["batch_size"],
                          shuffle=False)
        cnn_oof[idx_out] = cnn_fold.predict(oof_ds, verbose=0).ravel()
        log.info(f"  fold {fold} CNN OOF AUC = "
                 f"{roc_auc_score(y_tr[idx_out], cnn_oof[idx_out]):.4f}")

        # release fold model memory
        del cnn_fold, xgb_fold
        tf.keras.backend.clear_session()

    np.save(results / "xgb_oof_train_prob.npy", xgb_oof)
    np.save(results / "cnn_oof_train_prob.npy", cnn_oof)
    log.info(f"Overall OOF AUC  XGB={roc_auc_score(y_tr, xgb_oof):.4f}  "
             f"CNN={roc_auc_score(y_tr, cnn_oof):.4f}")

    # ------- Fit meta on OOF (no leakage) ----------------------------------
    Z_tr = np.column_stack([xgb_oof, cnn_oof])
    meta = LogisticRegression(max_iter=1000)
    meta.fit(Z_tr, y_tr)
    joblib.dump(meta, models_ / "hybrid_meta_oof.joblib")
    log.info(f"Meta coefficients = {meta.coef_.ravel()}  intercept = "
             f"{meta.intercept_.ravel()}")

    # ------- Refit base learners on FULL train -----------------------------
    log.info("Refitting XGB on full train ...")
    xgb_full = _new_xgb(cfg)
    xgb_full.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    joblib.dump(xgb_full, models_ / "xgb_url.joblib")

    xgb_val = xgb_full.predict_proba(X_va)[:, 1]
    xgb_test = xgb_full.predict_proba(X_te)[:, 1]
    np.save(results / "xgb_val_prob.npy", xgb_val)
    np.save(results / "xgb_test_prob.npy", xgb_test)
    log.info(f"XGB full   Val AUC = {roc_auc_score(y_va, xgb_val):.4f}  "
             f"Test AUC = {roc_auc_score(y_te, xgb_test):.4f}")

    log.info("Refitting CNN on full train ...")
    size = cfg["image"]["size"]
    batch = cfg["cnn"]["batch_size"]
    train_ds = _make_ds(img_tr, y_tr.astype("float32"), size, batch,
                        shuffle=True)
    val_ds = _make_ds(img_va, y_va.astype("float32"), size, batch,
                      shuffle=False)
    test_ds = _make_ds(img_te, y_te.astype("float32"), size, batch,
                       shuffle=False)

    cnn_full = _build_cnn(size, cfg["image"]["channels"])
    cb = [
        callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                patience=cfg["cnn"]["early_stop_patience"],
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=2),
    ]
    cnn_full.fit(train_ds, validation_data=val_ds,
                 epochs=cfg["cnn"]["epochs"], callbacks=cb, verbose=2)
    cnn_full.save(models_ / "cnn_qr.keras")

    cnn_val = cnn_full.predict(val_ds, verbose=0).ravel()
    cnn_test = cnn_full.predict(test_ds, verbose=0).ravel()
    np.save(results / "cnn_val_prob.npy", cnn_val)
    np.save(results / "cnn_test_prob.npy", cnn_test)
    log.info(f"CNN full   Val AUC = {roc_auc_score(y_va, cnn_val):.4f}  "
             f"Test AUC = {roc_auc_score(y_te, cnn_test):.4f}")

    # Re-export TFLite of the leakage-free CNN
    try:
        @tf.function
        def run_model(x):
            return cnn_full(x)
        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec([None, size, size, cfg["image"]["channels"]],
                          tf.float32))
        conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        (models_ / "cnn_qr.tflite").write_bytes(conv.convert())
        log.info(f"TFLite saved: "
                 f"{(models_ / 'cnn_qr.tflite').stat().st_size / 1024:.1f} KB")
    except Exception as e:
        log.warning(f"TFLite conversion failed: {e}")

    # ------- Apply meta to val + test --------------------------------------
    Z_va = np.column_stack([xgb_val, cnn_val])
    Z_te = np.column_stack([xgb_test, cnn_test])
    p_va = meta.predict_proba(Z_va)[:, 1]
    p_te = meta.predict_proba(Z_te)[:, 1]
    np.save(results / "hybrid_val_prob.npy", p_va)
    np.save(results / "hybrid_test_prob.npy", p_te)

    log.info(f"HYBRID  Val  AUC = {roc_auc_score(y_va, p_va):.4f}")
    log.info(f"HYBRID  Test AUC = {roc_auc_score(y_te, p_te):.4f}")
    log.info("Hybrid classification report (test):\n" +
             classification_report(y_te, (p_te > 0.5).astype(int), digits=5))
    log.info("Confusion matrix:\n" +
             str(confusion_matrix(y_te, (p_te > 0.5).astype(int))))

    pd.DataFrame({
        "model":    ["XGB (URL)", "CNN (QR image)", "Hybrid (OOF meta)"],
        "test_auc": [roc_auc_score(y_te, xgb_test),
                     roc_auc_score(y_te, cnn_test),
                     roc_auc_score(y_te, p_te)],
    }).to_csv(results / "model_comparison.csv", index=False)
    log.info("OOF hybrid stacking complete.")


if __name__ == "__main__":
    main()
