"""
Step 10 — Build an adversarially augmented training set.

We sample N malicious + N benign QRs from dataset_train.csv and apply the
three evasion transforms (split / nested / PDF round-trip). Malicious sources
keep label=1 (payload still present, just obscured); benign sources keep
label=0 (transformation does not create malice). The CNN must learn that
the transformation itself is uninformative.

Outputs:
  data/adversarial/train_aug/{variant}/...png
  data/dataset_train_augmented.csv   (original train + augmented samples)
"""
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import load_config, get_logger

# Reuse the same transformation functions from step 8 via importlib
import importlib.util
from pathlib import Path
HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("adv_make", HERE / "08_make_adversarial.py")
_adv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_adv)

log = get_logger("aug_train")

N_PER_CLASS_PER_VARIANT = 5000   # 5k mal + 5k ben × 3 variants = 30k augmented


def build_variant(variant: str, fn, mal_df, ben_df, out_dir, rs):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(N_PER_CLASS_PER_VARIANT, len(mal_df), len(ben_df))
    mal_sample = mal_df.sample(n=n, random_state=rs).reset_index(drop=True)
    ben_sample = ben_df.sample(n=n, random_state=rs + 1).reset_index(drop=True)
    rows = []
    for label_name, df in (("mal", mal_sample), ("ben", ben_sample)):
        # For variants that consume a benign cover (split / nested), pick a cover
        # from the OPPOSITE class so the mix is meaningful.
        pool = ben_sample if label_name == "mal" else mal_sample
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"{variant}-{label_name}"):
            try:
                src_img = Image.open(row["qr_path"])
                if variant == "pdf":
                    img = fn(src_img)
                else:
                    cover = Image.open(pool.iloc[i]["qr_path"])
                    img = fn(src_img, cover)
                out_path = out_dir / f"{variant}_{label_name}_{i:06d}.png"
                img.save(out_path)
                rows.append({
                    "qr_path": str(out_path),
                    "url": row["url"],
                    "label": 1 if label_name == "mal" else 0,
                    "source": f"aug_{variant}_{label_name}",
                })
            except Exception as e:
                log.warning(f"{variant}-{label_name} #{i} failed: {e}")
    return rows


def main():
    cfg     = load_config()
    data    = cfg["paths"]["data_dir"]
    rs      = cfg["dataset"]["random_state"]

    log.info("Loading train split metadata...")
    train = pd.read_csv(data / "dataset_train.csv")
    mal   = train[train["label"] == 1].reset_index(drop=True)
    ben   = train[train["label"] == 0].reset_index(drop=True)
    log.info(f"Train pool: {len(mal):,} malicious / {len(ben):,} benign")

    aug_root = data / "adversarial" / "train_aug"
    all_rows = []
    for variant, fn in _adv.VARIANT_FNS.items():
        log.info(f"Building variant: {variant}")
        rows = build_variant(variant, fn, mal, ben, aug_root / variant, rs)
        all_rows.extend(rows)
        log.info(f"  {variant}: {len(rows):,} samples written")

    aug_df = pd.DataFrame(all_rows)
    # Combine with the original train split (tag original rows with source="orig")
    orig = train.copy()
    orig["source"] = "orig"
    combined = pd.concat([orig, aug_df], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=rs).reset_index(drop=True)
    out_csv = data / "dataset_train_augmented.csv"
    combined.to_csv(out_csv, index=False)
    log.info(f"Augmented train set: {len(combined):,} rows -> {out_csv}")
    log.info(f"  original={len(orig):,}  augmented={len(aug_df):,}")
    log.info(f"  augmented class balance: mal={int((aug_df['label']==1).sum()):,}  "
             f"ben={int((aug_df['label']==0).sum()):,}")


if __name__ == "__main__":
    main()
