"""
Step 11 — Train an adversarially robust CNN.

Same architecture as step 5, but trains on the augmented CSV produced by step 10
(original train + 30k split/nested/PDF samples mixed in). Validation set stays
clean so val-AUC measures generalisation, not memorisation of augmentations.

Outputs:
  models/cnn_qr_robust.keras
  models/cnn_qr_robust.tflite
  results/cnn_robust_val_prob.npy
  results/cnn_robust_test_prob.npy
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, callbacks, metrics as keras_metrics
from sklearn.metrics import roc_auc_score, classification_report
from utils import load_config, get_logger

log = get_logger("cnn_robust")
AUTOTUNE = tf.data.AUTOTUNE


def make_ds(paths, labels, size, batch, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(min(len(paths), 50000), seed=42, reshuffle_each_iteration=True)

    def _load(path, y):
        raw = tf.io.read_file(path)
        img = tf.io.decode_png(raw, channels=1)
        img = tf.image.resize(img, [size, size], method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        return img, y

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)


def build_model(size, channels):
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
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=[keras_metrics.AUC(name="auc"),
                       keras_metrics.BinaryAccuracy(name="acc")])
    return m


def main():
    cfg     = load_config()
    data    = cfg["paths"]["data_dir"]
    models_ = cfg["paths"]["models_dir"]
    results = cfg["paths"]["results_dir"]

    size    = cfg["image"]["size"]
    chans   = cfg["image"]["channels"]
    batch   = cfg["cnn"]["batch_size"]
    epochs  = cfg["cnn"]["epochs"]
    pat     = cfg["cnn"]["early_stop_patience"]

    # Augmented train (original + adversarial), clean val/test from URL features
    train_csv = data / "dataset_train_augmented.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Run step 10 first: {train_csv} missing")
    tr = pd.read_csv(train_csv)
    va = pd.read_parquet(data / "url_features_val.parquet")
    te = pd.read_parquet(data / "url_features_test.parquet")

    log.info(f"Train (with aug)={len(tr):,}  Val={len(va):,}  Test={len(te):,}")
    log.info(f"  train source breakdown:\n{tr['source'].value_counts().to_string()}")

    train_ds = make_ds(tr["qr_path"].values, tr["label"].values.astype("float32"),
                       size, batch, shuffle=True)
    val_ds   = make_ds(va["qr_path"].values, va["label"].values.astype("float32"),
                       size, batch, shuffle=False)
    test_ds  = make_ds(te["qr_path"].values, te["label"].values.astype("float32"),
                       size, batch, shuffle=False)

    model = build_model(size, chans)
    model.summary(print_fn=log.info)

    cb = [
        callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=pat,
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cb, verbose=2)

    y_va = va["label"].values.astype("float32")
    y_te = te["label"].values.astype("float32")
    p_va_pred = model.predict(val_ds,  verbose=0).ravel()
    p_te_pred = model.predict(test_ds, verbose=0).ravel()
    log.info(f"Val AUC (clean)  = {roc_auc_score(y_va, p_va_pred):.4f}")
    log.info(f"Test AUC (clean) = {roc_auc_score(y_te, p_te_pred):.4f}")
    log.info("Test classification report (clean):\n" +
             classification_report(y_te, (p_te_pred > 0.5).astype(int), digits=4))

    model.save(models_ / "cnn_qr_robust.keras")
    np.save(results / "cnn_robust_val_prob.npy",  p_va_pred)
    np.save(results / "cnn_robust_test_prob.npy", p_te_pred)

    # TFLite export via Concrete Function (same approach as step 5)
    try:
        log.info("Converting robust model to TFLite via Concrete Function...")

        @tf.function
        def run_model(x):
            return model(x)

        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec([None, size, size, chans], tf.float32)
        )
        conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        tfl = conv.convert()

        out_tfl = models_ / "cnn_qr_robust.tflite"
        out_tfl.write_bytes(tfl)
        log.info(f"TFLite saved: {out_tfl.stat().st_size/1024:.1f} KB")
    except Exception as e:
        log.warning(f"TFLite conversion failed: {e}")

    log.info("Robust CNN training complete.")


if __name__ == "__main__":
    main()
