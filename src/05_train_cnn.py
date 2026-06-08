"""
Step 5 — Train a small CNN on the QR code images (structural channel).
Keeps the model well under 5MB so it fits the on-device budget.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, callbacks, metrics as keras_metrics
from sklearn.metrics import roc_auc_score, classification_report
from utils import load_config, get_logger

log = get_logger("cnn")
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
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return ds

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

def load_split_paths(data, split):
    df = pd.read_parquet(data / f"url_features_{split}.parquet")
    return df["qr_path"].values, df["label"].values.astype("float32")

def main():
    cfg = load_config()
    data    = cfg["paths"]["data_dir"]
    models_ = cfg["paths"]["models_dir"]
    results = cfg["paths"]["results_dir"]

    size    = cfg["image"]["size"]
    chans   = cfg["image"]["channels"]
    batch   = cfg["cnn"]["batch_size"]
    epochs  = cfg["cnn"]["epochs"]
    pat     = cfg["cnn"]["early_stop_patience"]

    p_tr, y_tr = load_split_paths(data, "train")
    p_va, y_va = load_split_paths(data, "val")
    p_te, y_te = load_split_paths(data, "test")

    log.info(f"Train images={len(p_tr):,}  Val={len(p_va):,}  Test={len(p_te):,}")
    train_ds = make_ds(p_tr, y_tr, size, batch, shuffle=True)
    val_ds   = make_ds(p_va, y_va, size, batch, shuffle=False)
    test_ds  = make_ds(p_te, y_te, size, batch, shuffle=False)

    model = build_model(size, chans)
    model.summary(print_fn=log.info)

    cb = [
        callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=pat, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cb, verbose=2)

    p_va_pred = model.predict(val_ds,  verbose=0).ravel()
    p_te_pred = model.predict(test_ds, verbose=0).ravel()
    log.info(f"Val AUC  = {roc_auc_score(y_va, p_va_pred):.4f}")
    log.info(f"Test AUC = {roc_auc_score(y_te, p_te_pred):.4f}")
    log.info("Test classification report:\n" +
             classification_report(y_te, (p_te_pred > 0.5).astype(int), digits=4))

    model.save(models_ / "cnn_qr.keras")
    np.save(results / "cnn_val_prob.npy",  p_va_pred)
    np.save(results / "cnn_test_prob.npy", p_te_pred)

    # TFLite export for mobile deployment via Concrete Function tracing
    try:
        log.info("Converting Keras model to TFLite format via Concrete Function...")
        
        # Wrap the model execution inside a traced tf.function
        @tf.function
        def run_model(x):
            return model(x)

        # Get the concrete function graph signature
        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec([None, size, size, chans], tf.float32)
        )
        
        # Convert using the low-level functional graph representation
        conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        tfl = conv.convert()
        
        (models_ / "cnn_qr.tflite").write_bytes(tfl)
        log.info(f"TFLite saved: {(models_ / 'cnn_qr.tflite').stat().st_size/1024:.1f} KB")
    except Exception as e:
        log.warning(f"TFLite conversion failed: {e}")

    log.info("CNN training complete.")

if __name__ == "__main__":
    main()