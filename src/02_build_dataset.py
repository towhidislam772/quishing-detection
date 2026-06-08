"""
Step 2 — Read both CSVs, label them, fix the qr_path column to point at the
actual extracted PNGs, optionally down-sample, and split into train/val/test.
Writes data/dataset_{train,val,test}.csv.
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import load_config, get_logger

log = get_logger("build_dataset")

BENIGN_CSV   = "QR_All_benign/all_generated_urls_20251015_161937.csv"
MAL_CSV      = "QR_All_Malicious/all_generated_urls_20251015_184324.csv"
BENIGN_QRDIR = "QR_All_benign/qrs"
MAL_QRDIR    = "QR_All_Malicious/qrs"

def fix_paths(df, data_dir, qr_subdir):
    """Replace the CSV's original Output\\... path with the real extracted path."""
    base = (data_dir / qr_subdir).resolve()
    df = df.copy()
    df["qr_path"] = df["qr_path"].apply(
        lambda p: str(base / Path(p).name)
    )
    return df

def main():
    cfg = load_config()
    data = cfg["paths"]["data_dir"]

    log.info("Loading CSVs...")
    df_b = pd.read_csv(data / BENIGN_CSV)
    df_m = pd.read_csv(data / MAL_CSV)
    log.info(f"Benign rows={len(df_b):,}  Malicious rows={len(df_m):,}")

    df_b = fix_paths(df_b, data, BENIGN_QRDIR);  df_b["label"] = 0
    df_m = fix_paths(df_m, data, MAL_QRDIR);     df_m["label"] = 1

    # Verify a few QR files exist; if not the user hasn't extracted yet.
    missing = sum(1 for p in df_b["qr_path"].head(5) if not Path(p).exists())
    if missing:
        log.warning(
            "Some QR PNGs missing on disk. Did you run 01_extract_data.py?"
        )

    n = cfg["dataset"]["sample_per_class"]
    if n:
        log.info(f"Sampling {n:,} per class for a fast pilot run.")
        rs = cfg["dataset"]["random_state"]
        df_b = df_b.sample(min(n, len(df_b)), random_state=rs)
        df_m = df_m.sample(min(n, len(df_m)), random_state=rs)

    df = pd.concat([df_b, df_m], ignore_index=True).sample(
        frac=1, random_state=cfg["dataset"]["random_state"]
    ).reset_index(drop=True)
    log.info(f"Combined dataset: {len(df):,} rows  (benign={(df.label==0).sum():,}, malicious={(df.label==1).sum():,})")

    ts = cfg["dataset"]["test_size"]
    vs = cfg["dataset"]["val_size"]
    rs = cfg["dataset"]["random_state"]

    train, test = train_test_split(df, test_size=ts, stratify=df["label"], random_state=rs)
    train, val  = train_test_split(train, test_size=vs/(1-ts), stratify=train["label"], random_state=rs)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        out = data / f"dataset_{name}.csv"
        part.to_csv(out, index=False)
        log.info(f"  {name}: {len(part):,} rows -> {out}")

    log.info("Dataset build complete.")

if __name__ == "__main__":
    main()
