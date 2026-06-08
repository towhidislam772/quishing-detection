"""
Step 1 — Extract the two large ZIPs into data/.
Skips files that already exist so it's safe to re-run.
"""
import zipfile
from pathlib import Path
from tqdm import tqdm
from utils import load_config, get_logger

log = get_logger("extract")

def extract_zip(zip_path: Path, out_dir: Path):
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing zip: {zip_path}")
    log.info(f"Extracting {zip_path.name} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as z:
        members = z.namelist()
        for m in tqdm(members, desc=zip_path.stem, unit="file"):
            target = out_dir / m
            if target.exists() and target.is_file() and target.stat().st_size > 0:
                continue
            z.extract(m, out_dir)
    log.info(f"Done: {zip_path.name}")

def main():
    cfg = load_config()
    raw = cfg["paths"]["raw_dir"]
    data = cfg["paths"]["data_dir"]
    extract_zip(raw / cfg["paths"]["benign_zip"], data)
    extract_zip(raw / cfg["paths"]["malicious_zip"], data)
    log.info("Extraction complete.")

if __name__ == "__main__":
    main()
