"""Shared helpers: config loading, path handling, logging."""
from pathlib import Path
import yaml
import logging

ROOT = Path(__file__).resolve().parent.parent

def load_config():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve paths relative to project root
    cfg["paths"]["raw_dir"] = (ROOT / cfg["paths"]["raw_dir"]).resolve()
    for k in ("data_dir", "models_dir", "results_dir"):
        p = (ROOT / cfg["paths"][k]).resolve()
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][k] = p
    return cfg

def get_logger(name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)
