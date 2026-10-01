import numpy as np
import pandas as pd
import pytest

from ipo_model import dataset, fetch, model, synthetic


@pytest.fixture(scope="module")
def df():
    return dataset.build(synthetic.make_sample(120, seed=7))


def test_synthetic_is_reproducible_and_labelled():
    a, b = synthetic.make_sample(50, 3), synthetic.make_sample(50, 3)
    pd.testing.assert_frame_equal(a, b)
    assert a["is_synthetic"].all()
    assert a["ipo_id"].str.startswith("SYN-").all()


def test_feature_math():
    raw = pd.DataFrame({"issue_price": [100, 200], "gmp": [10, None], "listing_price": [120, 190]})
    d = dataset.build(raw)
    assert d["gmp_pct"].tolist() == [10.0, 0.0]
    assert d["listing_gain_pct"].tolist() == [20.0, -5.0]
    assert d["positive"].tolist() == [1, 0]


def test_build_drops_unusable_rows():
    raw = pd.DataFrame({"issue_price": [0, 100], "gmp": [1, 1], "listing_price": [10, None]})
    assert len(dataset.build(raw)) == 0
    with pytest.raises(ValueError):
        dataset.build(pd.DataFrame({"gmp": [1]}))


def test_final_model_is_monotone_in_gmp(df):
    final = model.fit_final(df)
    g = np.linspace(-20, 60, 50).reshape(-1, 1)
    p = final["clf"].predict_proba(g)[:, 1]
    assert np.all(np.diff(p) > 0)
    assert np.all(np.diff(final["reg"].predict(g)) > 0)


def test_recovers_synthetic_slope(df):
    # true slope in the generator is 1.05; fat-tailed noise, so allow a loose band
    assert abs(model.fit_final(df)["slope_gain_per_gmp_pt"] - synthetic.TRUE_SLOPE) < 0.2


def test_oof_beats_baseline(df):
    h = model.headline(df, model.oof_predictions(df))
    assert h["cv_auc"] > 0.7
    assert h["cv_accuracy"] > h["majority_baseline_accuracy"]


def test_fetch_normalises_html_table():
    html = """<table><tr><th>IPO</th><th>Listing Date</th><th>IPO Size</th><th>Subscription</th>
    <th>GMP</th><th>IPO Price</th><th>Listing Price</th></tr>
    <tr><td>Example Co SME IPO</td><td>01-Jan-2025</td><td>25.5 Cr</td><td>12.3x</td>
    <td>&#8377;15</td><td>&#8377;100</td><td>&#8377;118.5</td></tr></table>"""
    from io import StringIO
    d = fetch.normalise(pd.read_html(StringIO(html)))
    row = d.iloc[0]
    assert (row["issue_price"], row["gmp"], row["listing_price"]) == (100, 15, 118.5)
    assert row["type"] == "SME" and row["name"] == "Example Co"
    assert row["subscription_x"] == 12.3 and row["size_cr"] == 25.5
