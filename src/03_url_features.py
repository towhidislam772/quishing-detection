"""
Step 3 — Extract lexical / structural URL features (no network calls, so this is
fast and reproducible). Output: data/url_features_{train,val,test}.parquet
"""
import re
import math
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import urlparse
import tldextract
from tqdm import tqdm
from utils import load_config, get_logger

log = get_logger("url_features")

SUSPICIOUS_WORDS = [
    "login","verify","update","secure","account","bank","confirm","signin",
    "wallet","webscr","ebay","paypal","free","bonus","gift","click","prize",
    "password","admin","cmd","cgi-bin","invoice","support","auth",
]
SHORTENERS = {"bit.ly","tinyurl.com","goo.gl","t.co","ow.ly","is.gd","buff.ly",
              "adf.ly","cutt.ly","shorturl.at","rebrand.ly","tiny.cc"}

def shannon_entropy(s: str) -> float:
    if not s: return 0.0
    p = pd.Series(list(s)).value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum())

def extract(url: str) -> dict:
    u = str(url) if pd.notna(url) else ""
    try:
        parsed = urlparse(u if "://" in u else "http://" + u)
        host = parsed.netloc or ""
        path = parsed.path or ""
        query = parsed.query or ""
    except Exception:
        host, path, query = "", "", ""
    ext = tldextract.extract(u)
    tld = ext.suffix or ""
    domain = ext.domain or ""
    sub = ext.subdomain or ""
    full_host = ".".join(x for x in (sub, domain, tld) if x)

    feats = {
        "url_len": len(u),
        "host_len": len(host),
        "path_len": len(path),
        "query_len": len(query),
        "num_dots": u.count("."),
        "num_hyphens": u.count("-"),
        "num_at": u.count("@"),
        "num_qmark": u.count("?"),
        "num_pct": u.count("%"),
        "num_amp": u.count("&"),
        "num_eq": u.count("="),
        "num_slash": u.count("/"),
        "num_digits": sum(c.isdigit() for c in u),
        "num_letters": sum(c.isalpha() for c in u),
        "digit_ratio": (sum(c.isdigit() for c in u) / max(1, len(u))),
        "has_https": int(u.lower().startswith("https://")),
        "has_http": int(u.lower().startswith("http://")),
        "has_ip": int(bool(re.match(r"^https?://\d{1,3}(\.\d{1,3}){3}", u))),
        "has_port": int(":" in host and not host.endswith(":")),
        "subdomain_count": len(sub.split(".")) if sub else 0,
        "tld_len": len(tld),
        "host_entropy": shannon_entropy(full_host),
        "url_entropy": shannon_entropy(u),
        "has_punycode": int("xn--" in host.lower()),
        "is_shortener": int(full_host.lower() in SHORTENERS),
        "suspicious_word_count": sum(w in u.lower() for w in SUSPICIOUS_WORDS),
        "double_slash_in_path": int("//" in path),
        "hex_in_url": int(bool(re.search(r"%[0-9a-fA-F]{2}", u))),
    }
    return feats

def process_split(split: str, data: Path):
    src = data / f"dataset_{split}.csv"
    df = pd.read_csv(src)
    log.info(f"{split}: extracting URL features for {len(df):,} rows")
    feats = [extract(u) for u in tqdm(df["url"].tolist(), desc=split)]
    feat_df = pd.DataFrame(feats)
    feat_df["label"] = df["label"].values
    feat_df["qr_path"] = df["qr_path"].values
    feat_df["url"] = df["url"].values
    out = data / f"url_features_{split}.parquet"
    feat_df.to_parquet(out, index=False)
    log.info(f"  saved {out}")

def main():
    cfg = load_config()
    data = cfg["paths"]["data_dir"]
    for split in ("train", "val", "test"):
        process_split(split, data)
    log.info("URL feature extraction complete.")

if __name__ == "__main__":
    main()
