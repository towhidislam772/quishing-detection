"""
Step 8 — Generate adversarial QR variants for evasion-robustness testing.

Three variants inspired by Barracuda 2025 quishing reports:
  - split:  malicious QR's left half tiled with a benign QR's right half
  - nested: a small malicious QR placed in the centre "logo slot" of a benign QR
  - pdf:    a malicious QR rendered through a one-page PDF round-trip

Each variant samples N malicious QRs from the held-out test split (so we never
touch training data) and writes PNGs + a metadata parquet that step 9 reads.

Parallelised with multiprocessing.Pool — per-sample work is CPU-bound PIL /
reportlab / pdfium calls, so n_workers ~= cpu_count gives near-linear speedup.
"""
import argparse
import io
import os
from multiprocessing import Pool, freeze_support
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader
import pypdfium2 as pdfium
from utils import load_config, get_logger

log = get_logger("adv_make")

DEFAULT_N_PER_VARIANT = 10000   # bumped from 2000 for tighter Wilson CIs
OUT_SIZE              = 256     # output image side length in pixels


def split_qr(mal_img: Image.Image, ben_img: Image.Image, out_size: int = OUT_SIZE) -> Image.Image:
    """Left half of malicious QR tiled with right half of benign QR."""
    m = mal_img.convert("L").resize((out_size, out_size))
    b = ben_img.convert("L").resize((out_size, out_size))
    canvas = Image.new("L", (out_size, out_size), 255)
    canvas.paste(m.crop((0, 0, out_size // 2, out_size)), (0, 0))
    canvas.paste(b.crop((out_size // 2, 0, out_size, out_size)), (out_size // 2, 0))
    return canvas


def nested_qr(mal_img: Image.Image, ben_img: Image.Image,
              out_size: int = OUT_SIZE, inner_frac: float = 0.25) -> Image.Image:
    """Small malicious QR pasted into the centre of a larger benign QR."""
    b = ben_img.convert("L").resize((out_size, out_size))
    inner = int(out_size * inner_frac)
    m = mal_img.convert("L").resize((inner, inner))
    pad = (out_size - inner) // 2
    b.paste(m, (pad, pad))
    return b


def pdf_roundtrip(mal_img: Image.Image, dpi: int = 150) -> Image.Image:
    """Embed QR in a single-page PDF then re-render back to a grayscale PNG."""
    pdf_buf = io.BytesIO()
    c = pdf_canvas.Canvas(pdf_buf)
    c.drawImage(ImageReader(mal_img.convert("RGB")), 100, 500, width=200, height=200)
    c.save()
    pdf_buf.seek(0)
    pdf = pdfium.PdfDocument(pdf_buf.read())
    page = pdf[0]
    rendered = page.render(scale=dpi / 72).to_pil().convert("L")
    return rendered


VARIANT_FNS = {"split": split_qr, "nested": nested_qr, "pdf": pdf_roundtrip}


def _worker(task):
    """Generate one adversarial sample. Runs in a worker process.

    Tasks carry only path strings (cheap to pickle); images are opened
    inside the worker.
    """
    variant, i, mal_path, ben_path, orig_url, out_dir = task
    try:
        mal_img = Image.open(mal_path)
        fn = VARIANT_FNS[variant]
        if variant == "pdf":
            img = fn(mal_img)
        else:
            ben_img = Image.open(ben_path)
            img = fn(mal_img, ben_img)
        out_path = Path(out_dir) / f"{variant}_{i:06d}.png"
        img.save(out_path)
        return {
            "ok": True,
            "qr_path": str(out_path),
            "orig_url": orig_url,
            "label": 1,           # ground truth: malicious
            "variant": variant,
        }
    except Exception as e:
        return {"ok": False, "error": f"{variant} #{i}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_N_PER_VARIANT,
                    help="adversarial samples per variant (default %(default)s)")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1),
                    help="parallel worker processes (default: cpu_count - 1)")
    ap.add_argument("--chunksize", type=int, default=64,
                    help="tasks dispatched per worker batch (default %(default)s)")
    args = ap.parse_args()
    N_PER_VARIANT = args.n

    cfg     = load_config()
    data    = cfg["paths"]["data_dir"]
    rs      = cfg["dataset"]["random_state"]

    log.info("Loading test split metadata...")
    test = pd.read_csv(data / "dataset_test.csv")
    mal  = test[test["label"] == 1].reset_index(drop=True)
    ben  = test[test["label"] == 0].reset_index(drop=True)
    log.info(f"Test pool: {len(mal):,} malicious / {len(ben):,} benign")

    n = min(N_PER_VARIANT, len(mal), len(ben))
    if n < N_PER_VARIANT:
        log.warning(f"Requested n={N_PER_VARIANT:,} exceeds test pool; "
                    f"capping to n={n:,}.")
    mal_sample = mal.sample(n=n, random_state=rs).reset_index(drop=True)
    ben_pool   = ben.sample(n=n, random_state=rs).reset_index(drop=True)
    log.info(f"Building {n:,} samples per variant.")
    log.info(f"Pool: {args.workers} worker processes, chunksize={args.chunksize}.")

    for variant in VARIANT_FNS:
        out_dir = data / "adversarial" / variant
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build cheap task list once (just strings + ints — no images yet)
        tasks = []
        for i in range(n):
            mal_path = mal_sample.iloc[i]["qr_path"]
            ben_path = None if variant == "pdf" else ben_pool.iloc[i]["qr_path"]
            orig_url = mal_sample.iloc[i]["url"]
            tasks.append((variant, i, mal_path, ben_path, orig_url, str(out_dir)))

        rows = []
        with Pool(processes=args.workers) as pool:
            for result in tqdm(pool.imap_unordered(_worker, tasks,
                                                   chunksize=args.chunksize),
                               total=len(tasks), desc=variant):
                if result["ok"]:
                    rows.append({k: v for k, v in result.items() if k != "ok"})
                else:
                    log.warning(result["error"])

        df = pd.DataFrame(rows)
        df.to_parquet(data / f"adversarial_{variant}.parquet", index=False)
        log.info(f"  {variant}: {len(df):,} rows -> "
                 f"{data / f'adversarial_{variant}.parquet'}")

    log.info("Adversarial generation complete.")


if __name__ == "__main__":
    freeze_support()    # required for Windows when frozen; harmless otherwise
    main()
