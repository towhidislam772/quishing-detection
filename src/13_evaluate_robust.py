"""
Step 13 — Evaluate the ROBUST hybrid pipeline on the adversarial test sets
(reusing the parquets from step 8).

Key differences from step 9:
  - loads the robust CNN (cnn_qr_robust.keras) and robust meta (hybrid_meta_robust.joblib)
  - fixes the "empty URL -> XGB's biased default" artefact by routing failed
    decodes to a neutral probability of 0.5 (the right behaviour for the
    hybrid: trust only the CNN in that case)
  - writes results/adversarial_results_robust.csv for the paper's §5 table

Run after steps 10 -> 11 -> 12 have completed.
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

log = get_logger("eval_robust")

# Reuse the step-3 URL feature extractor
HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("url_features_mod", HERE / "03_url_features.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
extract = _mod.extract


def decode_url(path: str):
    try:
        img = Image.open(path)
        results = zbar_decode(img)
        if results:
            return results[0].data.decode("utf-8", errors="ignore")
    except Exception:
        pass
    return None


def cnn_probs(model, paths, size, chans, batch=64):
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

    log.info("Loading robust models...")
    xgb_model  = joblib.load(models_ / "xgb_url.joblib")
    cnn_model  = tf.keras.models.load_model(models_ / "cnn_qr_robust.keras")
    meta_model = joblib.load(models_ / "hybrid_meta_robust.joblib")
    feature_cols = list(extract("http://example.com").keys())

    rows = []
    for variant in ("split", "nested", "pdf"):
        path = data / f"adversarial_{variant}.parquet"
        if not path.exists():
            log.warning(f"Skip {variant}: {path} missing (run step 8 first)")
            continue

        df = pd.read_parquet(path)
        n = len(df)
        log.info(f"=== {variant.upper()} (n={n:,}) ===")

        # 1) Decode QRs
        decoded_urls = [decode_url(p) for p in df["qr_path"].tolist()]
        decode_mask  = np.array([u is not None for u in decoded_urls])
        decode_rate  = decode_mask.mean()
        log.info(f"  decoded {int(decode_mask.sum()):,}/{n:,} ({decode_rate:.2%})")

        # 2) URL -> XGB, with NEUTRAL 0.5 when decode failed (no biased default)
        p_xgb = np.full(n, 0.5, dtype=np.float32)
        decoded_idx = np.where(decode_mask)[0]
        if decoded_idx.size:
            feats = [extract(decoded_urls[i]) for i in decoded_idx]
            X_url = pd.DataFrame(feats)[feature_cols].values
            p_xgb[decoded_idx] = xgb_model.predict_proba(X_url)[:, 1]

        # 3) Robust CNN
        p_cnn = cnn_probs(cnn_model, df["qr_path"].tolist(), size, chans)

        # 4) Robust hybrid meta
        p_hyb = meta_model.predict_proba(np.column_stack([p_cnn, p_xgb]))[:, 1]

        for name, probs in [("XGB", p_xgb), ("CNN_robust", p_cnn), ("Hybrid_robust", p_hyb)]:
            tpr = float((probs > 0.5).mean())
            log.info(f"  {name:14s}  detection@0.5 = {tpr:.4f}   mean_prob = {probs.mean():.4f}")
            rows.append({
                "variant":            variant,
                "model":              name,
                "n":                  n,
                "decode_rate":        round(float(decode_rate), 4),
                "detection_rate@0.5": round(tpr, 4),
                "mean_prob":          round(float(probs.mean()), 4),
            })

    out_csv = results / "adversarial_results_robust.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    log.info(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
