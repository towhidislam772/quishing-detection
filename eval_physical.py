import os, cv2, json
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode

PHOTO_DIR = "physical_test/photos"
cv_det = cv2.QRCodeDetector()


def try_decode(img):
    """Try several preprocessing + decoder combos. Return first non-empty URL or ''. """
    h, w = img.shape[:2]

    # Resize down if huge (phones produce 3000+ px which often hurts zbar)
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Variant set: original, blurred, sharpened, Otsu, adaptive, inverted
    variants = []
    variants.append(("gray", gray))
    variants.append(("gauss", cv2.GaussianBlur(gray, (3, 3), 0)))
    # contrast stretch
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    variants.append(("norm", norm))
    # Otsu binarisation
    _, otsu = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu))
    # adaptive
    adap = cv2.adaptiveThreshold(gray, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 5)
    variants.append(("adap", adap))
    # inverted Otsu (in case screen photo is reversed by glare)
    variants.append(("otsu_inv", 255 - otsu))

    # 1) Try OpenCV QRCodeDetector on each variant
    for name, v in variants:
        try:
            data, pts, _ = cv_det.detectAndDecode(v)
            if data:
                return data, f"cv-{name}"
        except Exception:
            pass

    # 2) Try pyzbar on each variant
    for name, v in variants:
        try:
            res = zbar_decode(v)
            if res:
                return res[0].data.decode(errors="ignore"), f"zbar-{name}"
        except Exception:
            pass

    return "", ""


results = []
for fname in sorted(os.listdir(PHOTO_DIR)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(PHOTO_DIR, fname)
    img = cv2.imread(path)
    if img is None:
        print(f"WARN: could not read {fname}")
        continue

    url, method = try_decode(img)

    if   "_normal" in fname: cond = "normal"
    elif "_angle"  in fname: cond = "angle"
    elif "_dim"    in fname: cond = "dim"
    else:                    cond = "unknown"

    true_label = 1 if fname.startswith("malicious") else 0

    results.append({
        "file": fname, "condition": cond, "true_label": true_label,
        "decoded": bool(url), "url": url, "method": method,
    })
    print(f"{fname:28s} cond={cond:<7} dec={bool(url):<5} "
          f"via={method:<14} url={url[:55]}")

print("\n========== SUMMARY ==========")
total = len(results)
ok    = sum(r["decoded"] for r in results)
print(f"Total photos       : {total}")
print(f"Decoded successful : {ok}  ({100*ok/total:.1f}%)")

for cond in ["normal", "angle", "dim"]:
    sub  = [r for r in results if r["condition"] == cond]
    sub_ok = sum(r["decoded"] for r in sub)
    if sub:
        print(f"  {cond:7s}: {sub_ok}/{len(sub)} decoded "
              f"({100*sub_ok/len(sub):.1f}%)")

with open("physical_test/results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved physical_test/results.json")