"""Fetch a GMP-vs-listing performance table YOURSELF and normalise it to the raw schema.

This repo deliberately ships no third-party data. Run this script against a source you are
permitted to use (check the site's terms of use / robots.txt), or against a page you saved locally.

    python -m ipo_model.fetch --url "<page with an HTML table>" --out data/raw/ipos.csv
    python -m ipo_model.fetch --html saved_page.html          --out data/raw/ipos.csv

The parser picks the first HTML table that has issue-price, GMP and listing-price-like columns and
maps them by header keywords. Anything it cannot map is dropped. Output columns:
    name, type, listing_date, size_cr, subscription_x, gmp, issue_price, listing_price
"""
from __future__ import annotations

import argparse
import re
from io import StringIO
from pathlib import Path

import pandas as pd

# header keyword -> canonical column (first match wins, checked in this order)
COLUMN_RULES = [
    ("listing_price", ("listing price", "listed price")),
    ("issue_price", ("ipo price", "issue price", "price band", "price")),
    ("gmp", ("gmp",)),
    ("subscription_x", ("subscription", "subscribed", "sub")),
    ("size_cr", ("size", "issue size")),
    ("listing_date", ("listing date", "date")),
    ("name", ("ipo", "company", "name")),
]


def _num(x):
    if pd.isna(x):
        return None
    m = re.search(r"-?\d[\d,]*\.?\d*", str(x).replace(",", ""))
    return float(m.group(0)) if m else None


def _map_columns(cols: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for col in cols:
        low = str(col).lower()
        for canon, keys in COLUMN_RULES:
            if canon in mapping.values():
                continue
            if any(k in low for k in keys):
                mapping[col] = canon
                break
    return mapping


def normalise(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for t in tables:
        m = _map_columns([str(c) for c in t.columns])
        if {"issue_price", "gmp", "listing_price"} <= set(m.values()):
            df = t.rename(columns=m)[list(m.values())].copy()
            for c in ("issue_price", "gmp", "listing_price", "subscription_x", "size_cr"):
                if c in df.columns:
                    df[c] = df[c].map(_num)
            if "name" in df.columns:
                nm = df["name"].astype(str)
                df["type"] = nm.str.contains(r"\bSME\b", regex=True).map({True: "SME", False: "Mainboard"})
                df["name"] = nm.str.replace(r"\b(SME|IPO)\b", "", regex=True).str.strip()
            return df
    raise ValueError("no table with issue price + GMP + listing price columns found")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch + normalise a GMP performance table (bring your own source)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url")
    src.add_argument("--html", help="path to a locally saved HTML page")
    ap.add_argument("--out", default="data/raw/ipos.csv")
    a = ap.parse_args()
    if a.url:
        import urllib.request

        req = urllib.request.Request(a.url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
        html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
    else:
        html = Path(a.html).read_text(encoding="utf-8", errors="ignore")
    df = normalise(pd.read_html(StringIO(html)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows -> {a.out}  (raw data stays local; data/raw/ is git-ignored)")


if __name__ == "__main__":
    main()
