"""
Physical-photo hybrid evaluation.
Runs CNN (image) + XGBoost (URL) + logistic-meta on the 30 phone photos
of QR codes displayed on a PC screen. Reports per-condition detection rate.
"""
import os, sys, json
import numpy as np
import cv2
import joblib
import pandas as pd
from pyzbar.pyzbar import decode as zbar_decode

# Make src/ importable so we can reuse the exact 28-feature extractor
sys.path.insert(0, "src")
from importlib import import_module
url_feat_mod = import_module("03_url_features") if False else None
# importlib can't load files starting with a digit, so inline-load instead
import importlib.util
spec = importlib.util.spec_from_file_location("url_features", "src/03_url_features.py")
url_features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(url_features)

# ----- load models -----
import tensorflow as tf
cnn = tf.keras.models.load_model("models/cnn_qr.keras", compile=False)
xgb = joblib.load("models/xgb_url.joblib")
meta = joblib.load("models/hybrid_meta.joblib")
print("Loaded CNN, XGB, meta.\n")

# ----- decode helpers (same as eval_physical.py) -----
cv_det = cv2.QRCodeDetector()

def try_decode(img):
    h, w = img.shape[:2]
    if max(h, w) > 1200:
        s = 1200 / max(h, w)
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants = [gray, cv2.GaussianBlur(gray, (3, 3), 0),
                cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)]
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    variants += [otsu,
                 cv2.adaptiveThreshold(gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5),
                 255 - otsu]
    for v in variants:
        try:
            data, _, _ = cv_det.detectAndDecode(v)
            if data: return data
        except Exception: pass
    for v in variants:
        try:
            r = zbar_decode(v)
            if r: return r[0].data.decode(errors="ignore")
        except Exception: pass
    return ""

# ----- CNN preprocess -----
def cnn_predict(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32) / 255.0
    x = x[None, ..., None]                # (1, 64, 64, 1)
    return float(cnn.predict(x, verbose=0)[0, 0])

# ----- XGB feature columns (same order as training) -----
FEAT_COLS = list(url_features.extract("http://x").keys())   # 28 features in order

def xgb_predict(url):
    if not url:
        return 0.5         # neutral default for empty URL (paper's fix)
    feats = url_features.extract(url)
    row = pd.DataFrame([[feats[c] for c in FEAT_COLS]], columns=FEAT_COLS)
    return float(xgb.predict_proba(row)[0, 1])

# ----- main loop -----
PHOTO_DIR = "physical_test/photos"
rows = []
for fname in sorted(os.listdir(PHOTO_DIR)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")): continue
    img = cv2.imread(os.path.join(PHOTO_DIR, fname))
    if img is None: continue

    url   = try_decode(img)
    p_img = cnn_predict(img)
    p_url = xgb_predict(url)
    p_hyb = float(meta.predict_proba([[p_url, p_img]])[0, 1])

    cond = ("normal" if "_normal" in fname else
            "angle"  if "_angle"  in fname else
            "dim"    if "_dim"    in fname else "?")
    label = 1 if fname.startswith("malicious") else 0

    rows.append(dict(file=fname, cond=cond, label=label,
                     decoded=bool(url), url=url,
                     p_url=p_url, p_img=p_img, p_hyb=p_hyb))
    print(f"{fname:26s} cond={cond:<7} label={label} "
          f"dec={int(bool(url))} p_url={p_url:.3f} "
          f"p_img={p_img:.3f} p_hyb={p_hyb:.3f}")

df = pd.DataFrame(rows)

# ----- per-condition summary -----
print("\n=========== HYBRID SUMMARY (tau=0.5) ===========")
print(f"{'cond':<8}{'N':>4}{'dec':>6}{'CNN det':>10}{'XGB det':>10}{'Hybrid det':>13}")
for c in ["normal", "angle", "dim", "ALL"]:
    sub = df if c == "ALL" else df[df["cond"] == c]
    if len(sub) == 0: continue
    n = len(sub)
    dec = sub["decoded"].sum()
    # detection rate = fraction with prob > 0.5 matching the true label
    cnn_det = ((sub["p_img"] > 0.5).astype(int) == sub["label"]).mean()
    xgb_det = ((sub["p_url"] > 0.5).astype(int) == sub["label"]).mean()
    hyb_det = ((sub["p_hyb"] > 0.5).astype(int) == sub["label"]).mean()
    print(f"{c:<8}{n:>4}{dec:>6}{cnn_det:>10.3f}{xgb_det:>10.3f}{hyb_det:>13.3f}")

# also: malicious-only detection rate (recall on phishing class)
print("\n=========== MALICIOUS-ONLY recall (tau=0.5) ===========")
print(f"{'cond':<8}{'N_mal':>6}{'CNN':>8}{'XGB':>8}{'Hybrid':>10}")
for c in ["normal", "angle", "dim", "ALL"]:
    sub = df if c == "ALL" else df[df["cond"] == c]
    mal = sub[sub["label"] == 1]
    if len(mal) == 0: continue
    cnn_r = (mal["p_img"] > 0.5).mean()
    xgb_r = (mal["p_url"] > 0.5).mean()
    hyb_r = (mal["p_hyb"] > 0.5).mean()
    print(f"{c:<8}{len(mal):>6}{cnn_r:>8.3f}{xgb_r:>8.3f}{hyb_r:>10.3f}")

df.to_csv("physical_test/results_hybrid.csv", index=False)
print("\nSaved physical_test/results_hybrid.csv")
