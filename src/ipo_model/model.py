"""Model zoo, cross-validated comparison, and the final GMP-only model."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import HuberRegressor, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
GMP_ONLY = ["gmp_pct"]
FULL = ["gmp_pct", "log_sub", "log_size", "is_sme"]


def logit() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])


def linreg() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("reg", LinearRegression())])


def classifiers() -> dict[str, tuple[list[str], object]]:
    return {
        "logistic · GMP only": (GMP_ONLY, logit()),
        "logistic · GMP + subscription": (["gmp_pct", "log_sub"], logit()),
        "logistic · all features": (FULL, logit()),
        "logistic + sigmoid calibration · GMP only": (
            GMP_ONLY, CalibratedClassifierCV(logit(), method="sigmoid", cv=3)),
        "random forest (depth 3) · all features": (
            FULL, RandomForestClassifier(n_estimators=400, max_depth=3, random_state=SEED)),
        "gradient boosting · all features": (
            FULL, GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=SEED)),
    }


def regressors() -> dict[str, tuple[list[str], object]]:
    return {
        "linear · GMP only": (GMP_ONLY, linreg()),
        "ridge · all features": (FULL, Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))])),
        "huber (robust) · all features": (FULL, Pipeline([("scale", StandardScaler()), ("reg", HuberRegressor())])),
    }


def _available(df: pd.DataFrame, feats: list[str]) -> bool:
    return all(f in df.columns for f in feats)


def compare(df: pd.DataFrame, n_repeats: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeated stratified 5-fold CV. Returns (classification table, regression table)."""
    y = df["positive"].to_numpy()
    yr = df["listing_gain_pct"].to_numpy()
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    crow = []
    for name, (feats, est) in classifiers().items():
        if not _available(df, feats):
            continue
        X = df[feats].to_numpy()
        aucs, accs = [], []
        for tr, te in rskf.split(X, y):
            m = _clone(est).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
            accs.append(accuracy_score(y[te], (p >= 0.5).astype(int)))
        crow.append({"model": name, "cv_auc_mean": np.mean(aucs), "cv_auc_sd": np.std(aucs),
                     "cv_acc_mean": np.mean(accs)})
    rrow = []
    for name, (feats, est) in regressors().items():
        if not _available(df, feats):
            continue
        X = df[feats].to_numpy()
        r2s, maes = [], []
        for rep in range(n_repeats):
            kf = KFold(5, shuffle=True, random_state=SEED + rep)
            pred = cross_val_predict(_clone(est), X, yr, cv=kf)
            r2s.append(r2_score(yr, pred))
            maes.append(mean_absolute_error(yr, pred))
        rrow.append({"model": name, "cv_r2_mean": np.mean(r2s), "cv_mae_pct": np.mean(maes)})
    ctab = pd.DataFrame(crow).sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)
    rtab = pd.DataFrame(rrow).sort_values("cv_mae_pct").reset_index(drop=True)
    return ctab, rtab


def _clone(est):
    from sklearn.base import clone
    return clone(est)


def oof_predictions(df: pd.DataFrame) -> np.ndarray:
    """Single 5-fold stratified out-of-fold P(positive) for the final GMP-only logistic."""
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return cross_val_predict(logit(), df[GMP_ONLY].to_numpy(), df["positive"].to_numpy(),
                             cv=skf, method="predict_proba")[:, 1]


def headline(df: pd.DataFrame, oof: np.ndarray) -> dict:
    y = df["positive"].to_numpy()
    base = max(y.mean(), 1 - y.mean())
    return {
        "n": int(len(df)),
        "positive_rate": float(y.mean()),
        "corr_gmp_gain": float(df["gmp_pct"].corr(df["listing_gain_pct"])),
        "cv_auc": float(roc_auc_score(y, oof)),
        "cv_accuracy": float(accuracy_score(y, (oof >= 0.5).astype(int))),
        "majority_baseline_accuracy": float(base),
        "brier": float(brier_score_loss(y, oof)),
    }


def fit_final(df: pd.DataFrame) -> dict:
    X = df[GMP_ONLY].to_numpy()
    clf = logit().fit(X, df["positive"].to_numpy())
    reg = linreg().fit(X, df["listing_gain_pct"].to_numpy())
    # GMP% at which P(positive) = 0.5: solve on a fine grid (model is monotone in GMP%)
    grid = np.linspace(-30, 60, 9001).reshape(-1, 1)
    p = clf.predict_proba(grid)[:, 1]
    even = float(grid[np.argmin(np.abs(p - 0.5)), 0])
    return {"clf": clf, "reg": reg, "breakeven_gmp_pct": even,
            "p_at_zero_gmp": float(clf.predict_proba([[0.0]])[0, 1]),
            "expected_gain_at_zero_gmp": float(reg.predict([[0.0]])[0]),
            "slope_gain_per_gmp_pt": float(reg.named_steps["reg"].coef_[0] / reg.named_steps["scale"].scale_[0])}
