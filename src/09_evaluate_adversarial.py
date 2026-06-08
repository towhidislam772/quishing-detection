"""
Step 9 — Evaluate the trained hybrid pipeline on the adversarial test sets
produced by step 8.

For each adversarial QR we:
  1. Decode it with pyzbar to extract whatever URL a phone scanner would read
     (may be the original malicious URL, the benign cover URL, or nothing).
  2. Run URL features through XGB (empty-string features when decoding fails).
  3. Run the image through the CNN.
  4. Stack with the trained hybrid meta-learner.
  5. Report decode rate + detection rate@0.5 for each model and variant.

Saves results/adversarial_results.csv for the paper's §5 table.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import importlib.util
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from PIL import Image
from pyzbar.pyzbar import decode as zbar_decode
from utils import load_config, get_logger

log = get_logger("adv_eval")

# --- Reuse the exact feature extractor from step 3 ---------------------------
HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("url_features_mod", HERE / "03_url_features.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
extract = _mod.extract            # 27-feature URL extractor (returns dict)
# ----------------------------------------------------------------------------


def decode_url(path: str):
    """Return decoded URL string, or None if the QR could not be read."""
    try:
        img = Image.open(path)
        results = zbar_decode(img)
        if results:
            return results[0].data.decode("utf-8", errors="ignore")
    except Exception:
        pass
    return None


def cnn_probs(model, paths, size, chans, batch=64):
    """Vectorised CNN inference over a list of image paths."""
    out = np.empty(len(paths), dtype=np.float32)
    for start in range(0, len(paths), batch):
        chunk = paths[start:start + batch]
        arr = np.empty((len(chunk), size, size, chans), dtype=np.float32)
        for j, p in enumerate(chunk):
            im = Image.open(p).convert("L").resize((size, size))
            arr[j, ..., 0] = np.asarray(im, dtype=np.float32) / 255.0
        out[start:start + len(chunk)] = model.predict(arr, verbose=0).ravel()
    return out


def main():
    cfg     = load_config()
    data    = cfg["paths"]["data_dir"]
    models_ = cfg["paths"]["models_dir"]
    results = cfg["paths"]["results_dir"]
    size    = cfg["image"]["size"]
    chans   = cfg["image"]["channels"]

    log.info("Loading models...")
    xgb_model  = joblib.load(models_ / "xgb_url.joblib")
    cnn_model  = tf.keras.models.load_model(models_ / "cnn_qr.keras")
    meta_model = joblib.load(models_ / "hybrid_meta.joblib")
    feature_cols = list(extract("http://example.com").keys())  # canonical order

    rows = []
    for variant in ("split", "nested", "pdf"):
        path = data / f"adversarial_{variant}.parquet"
        if not path.exists():
            log.warning(f"Skip {variant}: {path} not found (run step 8 first).")
            continue

        df = pd.read_parquet(path)
        n = len(df)
        log.info(f"=== {variant.upper()} (n={n:,}) ===")

        # 1) Decode QRs
        decoded_urls = [decode_url(p) for p in df["qr_path"].tolist()]
        decode_mask  = np.array([u is not None for u in decoded_urls])
        decode_rate  = decode_mask.mean()
        log.info(f"  decoded {int(decode_mask.sum()):,}/{n:,} ({decode_rate:.2%})")

               # 2) URL features (empty string for failed decodes -> all-zero features)
        feats = [extract(u if u else "") for u in decoded_urls]
        X_url = pd.DataFrame(feats)[feature_cols].values
        p_xgb = xgb_model.predict_proba(X_url)[:, 1]
        p_xgb[~decode_mask] = 0.5   # §5.3 neutral default for undecodable QRs

        # 3) CNN scores
        p_cnn = cnn_probs(cnn_model, df["qr_path"].tolist(), size, chans)

                # 4) Hybrid meta
        p_hyb = meta_model.predict_proba(np.column_stack([p_xgb, p_cnn]))[:, 1]

        y_true = df["label"].values  # all 1s (malicious ground truth)
        for name, probs in [("XGB", p_xgb), ("CNN", p_cnn), ("Hybrid", p_hyb)]:
            tpr = float((probs > 0.5).mean())
            log.info(f"  {name:6s}  detection@0.5 = {tpr:.4f}   mean_prob = {probs.mean():.4f}")
            rows.append({
                "variant":             variant,
                "model":               name,
                "n":                   n,
                "decode_rate":         round(float(decode_rate), 4),
                "detection_rate@0.5":  round(tpr, 4),
                "mean_prob":           round(float(probs.mean()), 4),
            })

    out_csv = results / "adversarial_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    log.info(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
