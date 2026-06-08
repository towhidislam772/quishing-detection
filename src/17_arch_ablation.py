"""
Step 17 - CNN architecture ablation.

Trains alternative CNN backbones on a balanced 20K-sample subsample of
the training split, validates on the full validation split, evaluates on
the full clean test split. The paper architecture (B) is loaded from the
already-trained model so that its number is comparable to the main paper
table (no re-training noise).

Variants:
  A - Single conv block (16 -> 32, GAP -> Dense), trained here
  B - Paper architecture (16 -> 32 -> 64, GAP -> Dense), loaded from
      models/cnn_qr.keras (i.e. the main paper's trained model)
  C - MobileNetV2 (alpha=0.35, no pretrained weights), trained here

Writes:
  results/arch_ablation.csv
  models/cnn_archA.keras, cnn_archC.keras (B is reused from cnn_qr.keras)
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models as kmodels, callbacks, metrics as keras_metrics
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score)
from utils import load_config, get_logger

log = get_logger("arch_ablation")
AUTOTUNE = tf.data.AUTOTUNE

# Subsample size per class for training the new architectures.
# Validation and test are always the full splits.
TRAIN_SUBSAMPLE_PER_CLASS = 10_000
SEED = 42


def make_ds(paths, labels, size, batch, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(min(len(paths), 50000), seed=SEED,
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


def build_arch_A(size, channels):
    inp = layers.Input(shape=(size, size, channels))
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return kmodels.Model(inp, out, name="arch_A_small")


def build_arch_B(size, channels):
    inp = layers.Input(shape=(size, size, channels))
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return kmodels.Model(inp, out, name="arch_B_paper")


def build_arch_C(size, channels):
    inp = layers.Input(shape=(size, size, channels))
    x = layers.Concatenate()([inp, inp, inp]) if channels == 1 else inp
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(size, size, 3),
        alpha=0.35,
        include_top=False,
        weights=None,
        pooling="avg",
    )
    x = backbone(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return kmodels.Model(inp, out, name="arch_C_mobilenet")


def compile_model(m):
    m.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=[keras_metrics.AUC(name="auc"),
                       keras_metrics.BinaryAccuracy(name="acc")])
    return m


def measure_latency_ms(model, size, channels, n_warmup=10, n_iter=100):
    x = np.random.rand(1, size, size, channels).astype("float32")
    for _ in range(n_warmup):
        _ = model(x, training=False)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = model(x, training=False)
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000.0


def load_split_df(data, split):
    return pd.read_parquet(data / f"url_features_{split}.parquet")


def subsample_balanced(df, per_class, seed):
    parts = []
    for cls in (0, 1):
        sub = df[df["label"] == cls]
        n = min(per_class, len(sub))
        parts.append(sub.sample(n=n, random_state=seed))
    out = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)
    return out.reset_index(drop=True)


def evaluate_model(model, ds, y_true):
    p = model.predict(ds, verbose=0).ravel()
    yhat = (p > 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_true, p),
        "precision": precision_score(y_true, yhat, zero_division=0),
        "recall": recall_score(y_true, yhat, zero_division=0),
        "f1": f1_score(y_true, yhat, zero_division=0),
    }


def main():
    cfg = load_config()
    data = cfg["paths"]["data_dir"]
    models_dir = cfg["paths"]["models_dir"]
    results = cfg["paths"]["results_dir"]

    size = cfg["image"]["size"]
    chans = cfg["image"]["channels"]
    batch = cfg["cnn"]["batch_size"]
    epochs = cfg["cnn"]["epochs"]
    pat = cfg["cnn"]["early_stop_patience"]

    df_tr = load_split_df(data, "train")
    df_va = load_split_df(data, "val")
    df_te = load_split_df(data, "test")

    df_tr_sub = subsample_balanced(df_tr, TRAIN_SUBSAMPLE_PER_CLASS, SEED)
    log.info(
        f"Train subsample={len(df_tr_sub):,}  "
        f"Val={len(df_va):,}  Test={len(df_te):,}"
    )

    p_tr = df_tr_sub["qr_path"].values
    y_tr = df_tr_sub["label"].values.astype("float32")
    p_va = df_va["qr_path"].values
    y_va = df_va["label"].values.astype("float32")
    p_te = df_te["qr_path"].values
    y_te = df_te["label"].values.astype("float32")

    train_ds = make_ds(p_tr, y_tr, size, batch, shuffle=True)
    val_ds   = make_ds(p_va, y_va, size, batch, shuffle=False)
    test_ds  = make_ds(p_te, y_te, size, batch, shuffle=False)

    rows = []

    # ---------- Arch A: train from scratch on subsample ----------
    log.info("=== Architecture A_small (subsample-trained) ===")
    modelA = compile_model(build_arch_A(size, chans))
    log.info(f"Params: {modelA.count_params():,}")
    cb = [
        callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                patience=pat, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss",
                                    factor=0.5, patience=2),
    ]
    t0 = time.perf_counter()
    modelA.fit(train_ds, validation_data=val_ds, epochs=epochs,
               callbacks=cb, verbose=2)
    t_trainA = time.perf_counter() - t0
    metA = evaluate_model(modelA, test_ds, y_te)
    latA = measure_latency_ms(modelA, size, chans)
    modelA.save(models_dir / "cnn_archA.keras")
    rows.append({
        "arch": "A_small_subsample",
        "params": modelA.count_params(),
        "auc": metA["auc"], "precision": metA["precision"],
        "recall": metA["recall"], "f1": metA["f1"],
        "latency_ms_per_sample": latA,
        "train_seconds": t_trainA,
    })
    log.info(
        f"[A] AUC={metA['auc']:.4f}  F1={metA['f1']:.4f}  "
        f"latency={latA:.2f}ms"
    )

    # ---------- Arch B: load the existing paper model ----------
    log.info("=== Architecture B_paper (loaded from models/cnn_qr.keras) ===")
    modelB = tf.keras.models.load_model(models_dir / "cnn_qr.keras")
    metB = evaluate_model(modelB, test_ds, y_te)
    latB = measure_latency_ms(modelB, size, chans)
    rows.append({
        "arch": "B_paper_full",
        "params": modelB.count_params(),
        "auc": metB["auc"], "precision": metB["precision"],
        "recall": metB["recall"], "f1": metB["f1"],
        "latency_ms_per_sample": latB,
        "train_seconds": float("nan"),
    })
    log.info(
        f"[B] AUC={metB['auc']:.4f}  F1={metB['f1']:.4f}  "
        f"latency={latB:.2f}ms"
    )

    # ---------- Arch C: train from scratch on subsample ----------
    log.info("=== Architecture C_mobilenet (subsample-trained) ===")
    modelC = compile_model(build_arch_C(size, chans))
    log.info(f"Params: {modelC.count_params():,}")
    cb = [
        callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                patience=pat, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss",
                                    factor=0.5, patience=2),
    ]
    t0 = time.perf_counter()
    modelC.fit(train_ds, validation_data=val_ds, epochs=epochs,
               callbacks=cb, verbose=2)
    t_trainC = time.perf_counter() - t0
    metC = evaluate_model(modelC, test_ds, y_te)
    latC = measure_latency_ms(modelC, size, chans)
    modelC.save(models_dir / "cnn_archC.keras")
    rows.append({
        "arch": "C_mobilenet_subsample",
        "params": modelC.count_params(),
        "auc": metC["auc"], "precision": metC["precision"],
        "recall": metC["recall"], "f1": metC["f1"],
        "latency_ms_per_sample": latC,
        "train_seconds": t_trainC,
    })
    log.info(
        f"[C] AUC={metC['auc']:.4f}  F1={metC['f1']:.4f}  "
        f"latency={latC:.2f}ms"
    )

    df = pd.DataFrame(rows)
    out = results / "arch_ablation.csv"
    df.to_csv(out, index=False)
    log.info(f"Wrote {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
