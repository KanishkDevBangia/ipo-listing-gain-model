# Model card: GMP-based IPO listing-gain model

**Author:** Kanishk Dev Bangia · **Version:** 1.0 · **License:** MIT
**Status:** educational research artefact. Not investment advice; not a trading signal.

## What it is
Two small models that take one input, `gmp_pct` (grey-market premium ÷ issue price × 100), and return:
1. **P(positive listing)**: standardised logistic regression; its out-of-fold calibration is checked (reliability bins + Brier score).
2. **Expected listing-day gain %**: ordinary least-squares linear regression.

## Training data
- **Original study (not shipped):** 83 listed Indian IPOs (59 SME, 23 mainboard, 1 REIT), last pre-listing GMP vs
  listing price, compiled from a public GMP-performance table in May 2026. The raw rows are third-party data, so
  they are **not** in this repo. `src/ipo_model/fetch.py` lets you assemble your own table from a source you are
  permitted to use.
- **Shipped sample:** `data/sample/ipo_synthetic.csv`, 120 **synthetic** rows from a documented generator
  (`src/ipo_model/synthetic.py`). It exists so the code runs; it is not evidence about real markets.

## Evaluation
| Metric | Original 83-IPO dataset (from the original study) | Synthetic sample (this repo, `make demo`) |
|---|---|---|
| 5-fold CV AUC, GMP-only logistic | **0.842** | 0.795 |
| CV accuracy vs majority baseline | 0.76 vs 0.51 | 0.76 vs 0.54 |
| corr(GMP%, listing gain%) | 0.87 | 0.885 |
| P(positive) at 0% GMP | 38% | 41% |
| GMP% for even odds | ≈ +5% | ≈ +3.2% |

Seven model variants were compared on the original data (linear, ridge, Huber, random forest, gradient
boosting, calibrated logistic, 3-class logistic). The single-feature GMP model matched or beat every
multi-feature model. With ~80 rows, extra features (subscription, issue size, SME flag) added variance, not
signal. The synthetic run reproduces the same ordering; see `results/RESULTS.md`.

## Intended use
- Teaching and research: how much information an informal pre-listing price carries, and how to measure
  that honestly (out-of-fold AUC, calibration, baselines).
- **Not intended** for: deciding whether to apply for, buy or sell any security; publishing per-IPO
  probabilities; any automated trading.

## Limitations and risks
- **Small sample, SME-heavy, one market regime.** The 83 IPOs cover a few months of a strong primary market.
  Coefficients will drift with market conditions.
- **GMP is unofficial.** It is an unregulated, thinly traded grey-market quote. It can be stale, manipulated,
  or reported differently across sources, and any of these breaks the model.
- **Timing leakage risk.** The feature must be the last GMP observed *before* listing. A GMP scraped after the
  listing price is known would leak the label.
- **Selection bias.** Only issues that listed and had a reported GMP are included. Withdrawn or unreported
  issues are missing.
- **Fat tails.** Rare blockbuster or crash listings sit far from the regression line. Expected-gain figures
  have wide error bars (on the synthetic sample, CV MAE is about 7 percentage points).
