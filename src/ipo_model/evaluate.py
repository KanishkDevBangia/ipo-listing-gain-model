"""End-to-end: build features -> compare models -> out-of-fold evaluation -> charts -> results/.

    python -m ipo_model.evaluate --raw data/sample/ipo_synthetic.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold

from . import dataset, model

BINS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0001)]


def calibration_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    rows = []
    for lo, hi in BINS:
        m = (p >= lo) & (p < hi)
        rows.append({"bin": f"{lo:.0%}-{min(hi, 1):.0%}", "n": int(m.sum()),
                     "mean_predicted": float(p[m].mean()) if m.any() else None,
                     "actual_positive_rate": float(y[m].mean()) if m.any() else None})
    return pd.DataFrame(rows)


def plot_calibration(y, p, out: Path, label: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    edges = np.linspace(0, 1, 6)
    idx = np.clip(np.digitize(p, edges) - 1, 0, 4)
    xs, ys, ns = [], [], []
    for b in range(5):
        m = idx == b
        if m.sum():
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect calibration")
    ax.plot(xs, ys, "o-", c="#1f5fa8", label="model (out-of-fold)")
    for x, yy, n in zip(xs, ys, ns):
        ax.annotate(f"n={n}", (x, yy), textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.set(xlabel="Predicted P(positive listing)", ylabel="Observed share positive",
           xlim=(0, 1), ylim=(0, 1.02), title=f"Calibration ({label})")
    ax.legend(loc="upper left", fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_scatter(df, reg, out: Path, label: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.6))
    c = np.where(df["positive"] == 1, "#2a9d55", "#c0392b")
    ax.scatter(df["gmp_pct"], df["listing_gain_pct"], c=c, s=22, alpha=0.75, edgecolor="none")
    g = np.linspace(df["gmp_pct"].min(), df["gmp_pct"].max(), 100).reshape(-1, 1)
    ax.plot(g, reg.predict(g), c="#1f5fa8", lw=2, label="linear fit (expected gain %)")
    ax.plot(g, g, c="grey", ls=":", lw=1, label="gain = GMP%")
    ax.axhline(0, c="black", lw=0.6)
    r = df["gmp_pct"].corr(df["listing_gain_pct"])
    ax.set(xlabel="GMP % of issue price", ylabel="Listing-day gain %",
           title=f"GMP% vs listing gain ({label}), r = {r:.2f}")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_roc(df, out: Path, label: str) -> None:
    X = df[model.GMP_ONLY].to_numpy(); y = df["positive"].to_numpy()
    skf = StratifiedKFold(5, shuffle=True, random_state=model.SEED)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    grid = np.linspace(0, 1, 101); tprs = []; aucs = []
    for k, (tr, te) in enumerate(skf.split(X, y), 1):
        p = model.logit().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        fpr, tpr, _ = roc_curve(y[te], p)
        aucs.append(auc(fpr, tpr)); tprs.append(np.interp(grid, fpr, tpr)); tprs[-1][0] = 0
        ax.plot(fpr, tpr, lw=1, alpha=0.45, label=f"fold {k} AUC {aucs[-1]:.2f}")
    mt = np.mean(tprs, axis=0); mt[-1] = 1
    ax.plot(grid, mt, c="#1f5fa8", lw=2.2, label=f"mean AUC {np.mean(aucs):.2f} ± {np.std(aucs):.2f}")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"5-fold CV ROC ({label})")
    ax.legend(fontsize=7, loc="lower right"); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full IPO listing-gain evaluation")
    ap.add_argument("--raw", default="data/sample/ipo_synthetic.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--label", default=None, help="label printed on charts (default: SYNTHETIC if is_synthetic column)")
    ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args()

    raw = pd.read_csv(a.raw)
    synthetic = bool(raw.get("is_synthetic", pd.Series([False])).astype(bool).all())
    label = a.label or ("SYNTHETIC sample" if synthetic else "user data")
    df = dataset.build(raw)
    out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)

    ctab, rtab = model.compare(df, n_repeats=a.repeats)
    oof = model.oof_predictions(df)
    head = model.headline(df, oof)
    final = model.fit_final(df)
    cal = calibration_table(df["positive"].to_numpy(), oof)

    plot_calibration(df["positive"].to_numpy(), oof, out / "figures" / "calibration.png", label)
    plot_scatter(df, final["reg"], out / "figures" / "gmp_vs_gain.png", label)
    plot_roc(df, out / "figures" / "cv_roc.png", label)

    examples = [{"gmp_pct": g, "p_positive": float(final["clf"].predict_proba([[g]])[0, 1]),
                 "expected_gain_pct": float(final["reg"].predict([[g]])[0])} for g in (-10, 0, 5, 10, 20, 40)]
    metrics = {"data_label": label, **head,
               "breakeven_gmp_pct": final["breakeven_gmp_pct"], "p_at_zero_gmp": final["p_at_zero_gmp"],
               "expected_gain_at_zero_gmp": final["expected_gain_at_zero_gmp"],
               "slope_gain_per_gmp_pt": final["slope_gain_per_gmp_pt"],
               "calibration": cal.to_dict("records"), "model_comparison_classification": ctab.to_dict("records"),
               "model_comparison_regression": rtab.to_dict("records"), "curve_examples": examples}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"clf": final["clf"], "reg": final["reg"], "feature": "gmp_pct", "data_label": label},
                "models/gmp_model.joblib")

    fmt = lambda t: t.to_markdown(index=False, floatfmt=".3f")
    md = [f"# Results ({label})", "",
          f"Generated by `python -m ipo_model.evaluate --raw {a.raw}`.", "",
          f"- rows: **{head['n']}**, positive-listing rate {head['positive_rate']:.0%}",
          f"- corr(GMP%, listing gain%): **{head['corr_gmp_gain']:.3f}**",
          f"- GMP-only logistic, 5-fold out-of-fold: **AUC {head['cv_auc']:.3f}**, "
          f"accuracy **{head['cv_accuracy']:.2f}** vs majority baseline {head['majority_baseline_accuracy']:.2f}, "
          f"Brier {head['brier']:.3f}",
          f"- P(positive | GMP 0%) = {final['p_at_zero_gmp']:.0%}, expected gain at 0% GMP "
          f"{final['expected_gain_at_zero_gmp']:+.1f}%; even odds at GMP ≈ {final['breakeven_gmp_pct']:+.1f}%",
          "", "## Classifier comparison (5-fold x %d repeats)" % a.repeats, "", fmt(ctab),
          "", "## Regressor comparison (expected gain %)", "", fmt(rtab),
          "", "## Calibration (out-of-fold)", "", fmt(cal), ""]
    (out / "RESULTS.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
