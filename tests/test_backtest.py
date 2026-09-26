import numpy as np
import pandas as pd
import pytest
from scipy import stats

from varengine.backtest import (violations, kupiec, transition_counts,
                                christoffersen_independence, christoffersen_cc,
                                basel_zone, es_backtest, backtest, backtest_all)


def hit_sequence(n_ones, n_total):
    return np.r_[np.ones(n_ones), np.zeros(n_total - n_ones)]


# ---------------------------------------------------------------- Kupiec

def test_kupiec_is_zero_at_the_expected_rate():
    assert kupiec(hit_sequence(25, 2500), 0.99)["LR_uc"] == pytest.approx(0.0, abs=1e-9)


def test_kupiec_zero_violations_closed_form():
    """With x = 0 the statistic collapses to -2 T ln(1 - p)."""
    r = kupiec(hit_sequence(0, 250), 0.99)
    assert r["LR_uc"] == pytest.approx(-2 * 250 * np.log(0.99))


def test_kupiec_worked_example_from_the_notes():
    r = kupiec(hit_sequence(8, 250), 0.99)
    assert r["LR_uc"] == pytest.approx(7.7336, abs=1e-3)
    assert r["p_uc"] == pytest.approx(0.0054, abs=1e-3)


@pytest.mark.parametrize("x", [0, 1, 5, 25, 60, 120])
def test_likelihood_ratio_is_never_negative(x):
    assert kupiec(hit_sequence(x, 2500), 0.99)["LR_uc"] >= -1e-12


def test_kupiec_ignores_the_ordering():
    rng = np.random.default_rng(0)
    h = hit_sequence(30, 2000)
    shuffled = rng.permutation(h)
    assert kupiec(h)["LR_uc"] == pytest.approx(kupiec(shuffled)["LR_uc"])


def test_kupiec_rejects_a_clearly_wrong_rate():
    assert kupiec(hit_sequence(100, 1000), 0.99)["p_uc"] < 1e-6


# -------------------------------------------------- Christoffersen

def test_transition_counts_add_up():
    h = np.array([0, 1, 1, 0, 0, 1, 0])
    n = transition_counts(h)
    assert sum(n.values()) == len(h) - 1
    assert n[1, 1] == 1
    assert n[0, 1] == 2


def test_independence_detects_clustering_kupiec_cannot_see():
    """The example in Chapter 13: identical counts, opposite verdicts."""
    spread = np.zeros(1000); spread[::100] = 1
    clustered = np.zeros(1000); clustered[500:510] = 1

    assert kupiec(spread)["LR_uc"] == pytest.approx(kupiec(clustered)["LR_uc"])
    assert christoffersen_independence(spread)["p_ind"] > 0.5
    assert christoffersen_independence(clustered)["p_ind"] < 1e-10


def test_independence_is_zero_without_violations():
    r = christoffersen_independence(np.zeros(500))
    assert r["LR_ind"] == pytest.approx(0.0, abs=1e-12)
    assert r["p_ind"] == pytest.approx(1.0)


def test_conditional_coverage_is_the_sum_of_its_parts():
    rng = np.random.default_rng(1)
    h = (rng.random(2000) < 0.015).astype(int)
    r = christoffersen_cc(h, 0.99)
    assert r["LR_cc"] == pytest.approx(r["LR_uc"] + r["LR_ind"])
    assert r["p_cc"] == pytest.approx(stats.chi2.sf(r["LR_cc"], 2))


def test_tests_have_roughly_the_right_size_on_correct_models():
    """A correct model should be rejected about 5% of the time, not often."""
    rng = np.random.default_rng(4)
    rejects = sum(kupiec((rng.random(2000) < 0.01).astype(int))["p_uc"] < 0.05
                  for _ in range(600))
    assert 0.01 < rejects / 600 < 0.12


# ------------------------------------------------------------ Basel

@pytest.mark.parametrize("x, zone", [(0, "green"), (4, "green"), (5, "yellow"),
                                     (9, "yellow"), (10, "red"), (20, "red")])
def test_basel_zone_boundaries(x, zone):
    assert basel_zone(hit_sequence(x, 250), 0.99)["zone"] == zone


def test_basel_multiplier_increases_with_violations():
    mults = [basel_zone(hit_sequence(x, 250))["multiplier"] for x in range(0, 11)]
    assert mults == sorted(mults)
    assert mults[0] == 3.0 and mults[-1] == 4.0


def test_basel_uses_only_the_last_window():
    h = np.r_[np.ones(50), np.zeros(250)]          # all violations long ago
    assert basel_zone(h, 0.99, window=250)["violations"] == 0


# --------------------------------------------------------------- ES

@pytest.mark.parametrize("factor, sign", [(1.0, 0), (0.7, 1), (1.4, -1)])
def test_z2_sign_detects_mis_stated_es(factor, sign):
    alpha, n = 0.99, 40_000
    z = stats.norm.ppf(alpha)
    es = stats.norm.pdf(z) / (1 - alpha)

    loss = pd.Series(np.random.default_rng(0).standard_normal(n))
    forecasts = pd.DataFrame({"VaR": z, "ES": factor * es}, index=loss.index)
    z2 = es_backtest(loss, forecasts, alpha)["Z2"]

    if sign == 0:
        assert abs(z2) < 0.1
    else:
        assert np.sign(z2) == sign and abs(z2) > 0.1


def test_z2_flags_a_normal_model_against_fat_tailed_losses():
    alpha, nu = 0.99, 4
    z = stats.norm.ppf(alpha)
    loss = pd.Series(stats.t.rvs(nu, size=40_000, random_state=1)
                     * np.sqrt((nu - 2) / nu))
    forecasts = pd.DataFrame({"VaR": z, "ES": stats.norm.pdf(z) / (1 - alpha)},
                             index=loss.index)
    assert es_backtest(loss, forecasts, alpha)["Z2"] > 0.5


# ------------------------------------------------------- integration

def test_violations_align_on_shared_dates(losses):
    from varengine.historical import rolling_hs
    f = rolling_hs(losses, window=250)
    h = violations(losses, f["VaR"])
    assert h.index.equals(f.index)
    assert set(np.unique(h)) <= {0, 1}


def test_backtest_all_returns_numeric_columns(losses):
    from varengine.historical import rolling_hs, rolling_fhs
    from varengine.parametric import rolling_normal

    models = {"HS": rolling_hs(losses, window=250),
              "Normal": rolling_normal(losses, window=250),
              "FHS": rolling_fhs(losses, window=250)}
    out = backtest_all(losses, models, alpha=0.99)

    assert list(out.index) == list(models)
    assert out["violations"].dtype.kind == "i"
    assert out["p_cc"].dtype.kind == "f"
    assert out["p_cc"].between(0, 1).all()
    assert out["expected"].nunique() == 1          # same sample for every model
