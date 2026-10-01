"""Generate a clearly-labelled SYNTHETIC IPO sample so the pipeline runs out of the box.

None of these rows are real IPOs. The generator is tuned so that its *aggregate* shape loosely
resembles small Indian IPO samples (many flat-GMP SME issues, a long right tail of hot issues,
listing gain roughly linear in GMP% with fat-tailed noise). The true data-generating process is
written below so readers can check whether the model recovers it.

    gmp_pct        ~ 40%: exactly 0   (flat grey market, common for small SME issues)
                     45%: Normal(6, 9)
                     15%: 10 + Exponential(mean 25)          (hot issues)
    listing_gain   = -2.5 + 1.05 * gmp_pct + 7 * StudentT(df=3)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRUE_INTERCEPT = -2.5
TRUE_SLOPE = 1.05


def make_sample(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    comp = rng.choice(3, size=n, p=[0.40, 0.45, 0.15])
    gmp_pct = np.where(
        comp == 0,
        0.0,
        np.where(comp == 1, rng.normal(6, 9, n), 10 + rng.exponential(25, n)),
    )
    gmp_pct = np.clip(gmp_pct, -30, 100)
    noise = 7 * rng.standard_t(df=3, size=n)
    gain_pct = np.clip(TRUE_INTERCEPT + TRUE_SLOPE * gmp_pct + noise, -60, 150)

    issue_type = rng.choice(["SME", "Mainboard"], size=n, p=[0.7, 0.3])
    issue_price = np.where(
        issue_type == "SME", rng.integers(40, 300, n), rng.integers(100, 1200, n)
    ).astype(float)
    gmp = np.round(issue_price * gmp_pct / 100.0, 0)
    listing_price = np.round(issue_price * (1 + gain_pct / 100.0), 2)
    # subscription loosely tracks GMP (hot issues are oversubscribed) -- an extra, noisy feature
    subscription = np.round(np.exp(rng.normal(1.2 + 0.05 * gmp_pct, 1.2, n)), 2)
    size_cr = np.round(np.where(issue_type == "SME", rng.lognormal(3.6, 0.7, n), rng.lognormal(6.0, 1.0, n)), 2)
    dates = pd.date_range("2024-01-01", periods=n, freq="5D")

    return pd.DataFrame(
        {
            "ipo_id": [f"SYN-{i:03d}" for i in range(1, n + 1)],
            "type": issue_type,
            "listing_date": dates.strftime("%Y-%m-%d"),
            "size_cr": size_cr,
            "subscription_x": subscription,
            "gmp": gmp,
            "issue_price": issue_price,
            "listing_price": listing_price,
            "is_synthetic": True,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/sample/ipo_synthetic.csv")
    a = ap.parse_args()
    df = make_sample(a.n, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} SYNTHETIC rows -> {a.out}")


if __name__ == "__main__":
    main()
