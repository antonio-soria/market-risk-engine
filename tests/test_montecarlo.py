import numpy as np
import pytest

from varengine.montecarlo import draw_innovations, mc_var_es, rolling_mc
from varengine.parametric import (normal_var_es, t_var_es, t_scale_from_sigma,
                                  rolling_ewma)


def test_innovations_have_identity_covariance():
    z = draw_innovations(200_000, 3, seed=0)
    assert z.mean(axis=0) == pytest.approx(np.zeros(3), abs=0.01)
    assert np.cov(z, rowvar=False) == pytest.approx(np.eye(3), abs=0.02)


def test_t_innovations_also_have_unit_variance():
    """The (nu-2)/W scaling must leave the covariance alone."""
    z = draw_innovations(400_000, 3, dist="t", nu=6, seed=0)
    assert np.cov(z, rowvar=False) == pytest.approx(np.eye(3), abs=0.05)


def test_t_innovations_are_fatter_tailed():
    normal = draw_innovations(200_000, 1, seed=0)
    student = draw_innovations(200_000, 1, dist="t", nu=4, seed=0)
    assert np.quantile(student, 0.999) > np.quantile(normal, 0.999)


def test_unknown_distribution_is_rejected():
    with pytest.raises(ValueError):
        draw_innovations(10, 2, dist="cauchy")


def test_cholesky_reproduces_the_covariance(returns):
    sigma = returns.cov().to_numpy()
    c = np.linalg.cholesky(sigma)
    assert c @ c.T == pytest.approx(sigma)


def test_simulated_returns_have_the_target_covariance(returns, weights):
    sigma = returns.cov().to_numpy()
    z = draw_innovations(400_000, len(weights), seed=1)
    simulated = z @ np.linalg.cholesky(sigma).T
    assert np.cov(simulated, rowvar=False) == pytest.approx(sigma, rel=0.02)


def test_seed_makes_results_reproducible(returns, weights):
    sigma = returns.cov().to_numpy()
    a = mc_var_es(sigma, weights, Z=draw_innovations(20_000, 4, seed=7))
    b = mc_var_es(sigma, weights, Z=draw_innovations(20_000, 4, seed=7))
    assert a == pytest.approx(b)


def test_monte_carlo_reproduces_the_closed_form(returns, weights):
    """Proposition 8.5: for a linear portfolio, MC must match the formula."""
    sigma = returns.cov().to_numpy()
    vol = np.sqrt(np.asarray(weights) @ sigma @ np.asarray(weights))

    var_mc, es_mc = mc_var_es(sigma, weights, 0.99,
                              Z=draw_innovations(500_000, len(weights), seed=3))
    var_cf, es_cf = normal_var_es(vol, 0.99)
    assert var_mc == pytest.approx(var_cf, rel=0.02)
    assert es_mc == pytest.approx(es_cf, rel=0.02)


def test_monte_carlo_t_reproduces_the_closed_form(returns, weights):
    nu = 6
    sigma = returns.cov().to_numpy()
    vol = np.sqrt(np.asarray(weights) @ sigma @ np.asarray(weights))

    var_mc, es_mc = mc_var_es(sigma, weights, 0.99,
                              Z=draw_innovations(500_000, len(weights),
                                                 dist="t", nu=nu, seed=3))
    var_cf, es_cf = t_var_es(t_scale_from_sigma(vol, nu), nu, 0.99)
    assert var_mc == pytest.approx(var_cf, rel=0.03)
    assert es_mc == pytest.approx(es_cf, rel=0.05)


def test_simulation_error_shrinks_as_one_over_root_n(returns, weights):
    """Chapter 8: the standard error should fall roughly like 1/sqrt(N)."""
    sigma = returns.cov().to_numpy()
    spread = {}
    for n in (2_000, 32_000):
        q = [mc_var_es(sigma, weights, 0.99,
                       Z=draw_innovations(n, len(weights), seed=s))[0]
             for s in range(40)]
        spread[n] = np.std(q)
    assert spread[2_000] / spread[32_000] == pytest.approx(4.0, rel=0.5)


def test_rolling_mc_tracks_rolling_ewma(returns, weights, losses):
    """The engine's internal consistency check, on the whole series."""
    mc = rolling_mc(returns, weights, n_sims=40_000, start=250)
    par = rolling_ewma(losses)
    common = mc.index.intersection(par.index)
    ratio = mc.loc[common, "VaR"] / par.loc[common, "VaR"]
    assert ratio.mean() == pytest.approx(1.0, abs=0.02)
    assert ratio.max() < 1.05
