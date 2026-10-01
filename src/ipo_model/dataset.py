"""Turn a raw IPO table into model features.

Required raw columns: issue_price, gmp, listing_price. Optional: type, subscription_x, size_cr.
Features:  gmp_pct          = gmp / issue_price * 100   (grey-market premium as % of issue price)
Targets:   listing_gain_pct = (listing_price - issue_price) / issue_price * 100
           positive         = listing_gain_pct > 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("issue_price", "gmp", "listing_price")


def build(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"raw data missing columns: {missing}")
    df = raw.copy()
    for c in ("issue_price", "gmp", "listing_price", "subscription_x", "size_cr"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["gmp"] = df["gmp"].fillna(0.0)
    df = df[(df["issue_price"] > 0) & df["listing_price"].notna()].copy()
    df["gmp_pct"] = (df["gmp"] / df["issue_price"] * 100).round(2)
    df["listing_gain_pct"] = ((df["listing_price"] - df["issue_price"]) / df["issue_price"] * 100).round(2)
    df["positive"] = (df["listing_gain_pct"] > 0).astype(int)
    if "subscription_x" in df.columns:
        df["log_sub"] = np.log1p(df["subscription_x"].clip(lower=0).fillna(0))
    if "size_cr" in df.columns:
        df["log_size"] = np.log1p(df["size_cr"].clip(lower=0).fillna(0))
    if "type" in df.columns:
        df["is_sme"] = (df["type"].astype(str).str.upper() == "SME").astype(int)
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build model features from a raw IPO CSV")
    ap.add_argument("--raw", default="data/sample/ipo_synthetic.csv")
    ap.add_argument("--out", default="data/processed/dataset.csv")
    a = ap.parse_args()
    df = build(pd.read_csv(a.raw))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} usable rows | positive-listing rate {df.positive.mean():.0%} | "
          f"corr(GMP%, gain%) {df.gmp_pct.corr(df.listing_gain_pct):.3f} -> {a.out}")


if __name__ == "__main__":
    main()
