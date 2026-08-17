"""
Regression-metric tests.

Two defects motivated these:

* `_spearman` used the shortcut formula rho = 1 - 6*sum(d^2)/(n*(n^2-1)),
  which is valid only when there are NO tied values, and ranked with
  argsort(argsort(.)), which breaks ties by row position. Vinardo writes
  affinities to two decimals, so ties are guaranteed and the reported rho
  depended on CSV row order. Spearman's rho is the headline metric of this
  study, so it has to be the real one.

* No R^2 was reported anywhere. Correlation cannot distinguish a useful model
  from a useless one; only R^2 answers "does this beat predicting the mean?".
"""
from __future__ import annotations

import numpy as np
import pytest

from Graph_model.train.metrics import (
    FoldMetrics, _rankdata, _spearman, aggregate_folds, regression_metrics,
)


# ── Tie handling ─────────────────────────────────────────────────────────────

def test_rankdata_averages_ties():
    got = _rankdata(np.array([10.0, 20.0, 20.0, 30.0]))
    assert list(got) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_is_invariant_to_row_order():
    """
    The old estimator drifted when rows were permuted, because argsort broke
    ties by position. A metric that depends on CSV ordering is not a metric.
    """
    rng = np.random.default_rng(0)
    x = np.round(rng.normal(size=300), 1)          # heavy ties, as in the data
    y = np.round(x + rng.normal(scale=0.6, size=300), 1)

    p = rng.permutation(300)
    assert _spearman(x, y) == pytest.approx(_spearman(x[p], y[p]), abs=1e-12)


def test_spearman_matches_scipy_when_available():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    x = np.round(rng.normal(size=200), 1)
    y = np.round(x + rng.normal(scale=0.8, size=200), 1)

    assert _spearman(x, y) == pytest.approx(
        scipy_stats.spearmanr(x, y).statistic, abs=1e-10)


def test_spearman_is_exactly_one_for_a_monotone_map():
    x = np.arange(50, dtype=float)
    assert _spearman(x, np.exp(x / 10)) == pytest.approx(1.0)
    assert _spearman(x, -x) == pytest.approx(-1.0)


def test_spearman_is_nan_when_a_side_is_constant():
    x = np.arange(20, dtype=float)
    assert np.isnan(_spearman(x, np.ones(20)))


# ── R^2 ──────────────────────────────────────────────────────────────────────

def test_r2_is_zero_for_the_mean_predictor():
    rng = np.random.default_rng(2)
    t = rng.normal(size=200)
    m = regression_metrics(np.full(200, t.mean()), t)
    assert m["r2"] == pytest.approx(0.0, abs=1e-12)


def test_r2_is_one_for_perfect_prediction():
    t = np.linspace(-3, 3, 100)
    assert regression_metrics(t, t)["r2"] == pytest.approx(1.0)


def test_r2_goes_negative_when_worse_than_the_mean():
    """
    The case that matters here: a model can show a positive correlation and
    still be worse than a constant. Only R^2 says so.
    """
    rng = np.random.default_rng(3)
    t = rng.normal(size=200)
    preds = t * 5.0            # right ranking, badly mis-scaled
    m = regression_metrics(preds, t)
    assert m["pearson_r"] == pytest.approx(1.0)
    assert m["r2"] < 0, "R^2 must expose the scale error that r hides"


def test_r2_is_not_pearson_r_squared():
    """A biased predictor has r = 1 and R^2 well below 1."""
    t = np.linspace(0, 10, 100)
    m = regression_metrics(t + 4.0, t)          # constant offset
    assert m["pearson_r"] == pytest.approx(1.0)
    assert m["r2"] < 0.9


# ── Plumbing ─────────────────────────────────────────────────────────────────

def test_regression_metrics_exposes_every_expected_key():
    m = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.5, 2.8])
    for k in ("rmse", "mae", "r2", "pearson_r", "spearman_r", "n"):
        assert k in m, f"missing metric {k!r}"


def test_empty_input_returns_nan_not_a_crash():
    m = regression_metrics([], [])
    assert m["n"] == 0 and np.isnan(m["r2"]) and np.isnan(m["rmse"])


def test_nan_pairs_are_dropped_before_scoring():
    m = regression_metrics([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])
    assert m["n"] == 2 and m["rmse"] == pytest.approx(0.0)


def test_aggregate_folds_reports_r2():
    folds = [
        FoldMetrics(fold=i, held_out=f"lig{i}", n_test=10, rmse=1.0, mae=0.8,
                    pearson_r=0.3, spearman_r=0.2, r2=-0.5)
        for i in range(3)
    ]
    agg = aggregate_folds(folds)
    assert agg["r2_mean"] == pytest.approx(-0.5)
    assert agg["r2_std"] == pytest.approx(0.0)
    assert agg["n_folds"] == 3


def test_fold_metrics_r2_defaults_without_breaking_existing_callers():
    fm = FoldMetrics(fold=0, held_out="x", n_test=1, rmse=1.0, mae=1.0,
                     pearson_r=0.0, spearman_r=0.0)
    assert np.isnan(fm.r2)
    assert "r2" in fm.to_dict()
