import numpy as np
import pytest
from scipy import stats

from varengine.parametric import (normal_var_es, t_var_es, t_scale_from_sigma,
                                  fit_t, rolling_normal, rolling_ewma,
                                  risk_contributions)
from varengine.volatility import ewma_variance, ewma_covariance


def test_normal_var_es_closed_form():
    """Proposition 3.4, at the values tabulated in the notes."""
    var, es = normal_var_es(1.0, 0.99)
    assert var == pytest.approx(2.3263, abs=1e-4)
    assert es == pytest.approx(2.6652, abs=1e-4)


def test_normal_scales_linearly_with_sigma():
    v1, e1 = normal_var_es(1.0, 0.99)
    v2, e2 = normal_var_es(3.0, 0.99)
    assert (v2, e2) == pytest.approx((3 * v1, 3 * e1))


def test_t_var_es_matches_simulation():
    """Proposition 6.3 against 2 million draws."""
    nu, scale = 4, t_scale_from_sigma(1.0, 4)
    var, es = t_var_es(scale, nu, 0.99)
    x = stats.t.rvs(nu, size=2_000_000, random_state=0) * scale
    assert var == pytest.approx(np.quantile(x, 0.99), rel=0.01)
    assert es == pytest.approx(x[x >= var].mean(), rel=0.01)


def test_t_converges_to_normal_as_nu_grows():
    nu = 5_000
    var_t, es_t = t_var_es(t_scale_from_sigma(1.0, nu), nu, 0.99)
    var_n, es_n = normal_var_es(1.0, 0.99)
    assert var_t == pytest.approx(var_n, rel=1e-3)
    assert es_t == pytest.approx(es_n, rel=1e-3)


def test_t_scale_is_not_the_standard_deviation():
    """The bug the helper exists to prevent: Var(t_nu) = nu / (nu - 2)."""
    nu, sigma = 4, 2.0
    scale = t_scale_from_sigma(sigma, nu)
    assert scale ** 2 * nu / (nu - 2) == pytest.approx(sigma ** 2)


def test_normal_and_t_var_cross_below_975():
    """Chapter 6: at equal volatility the t gives a LOWER VaR at 95%."""
    nu = 4
    lower = t_var_es(t_scale_from_sigma(1.0, nu), nu, 0.95)[0]
    upper = t_var_es(t_scale_from_sigma(1.0, nu), nu, 0.99)[0]
    assert lower < normal_var_es(1.0, 0.95)[0]
    assert upper > normal_var_es(1.0, 0.99)[0]


def test_fit_t_recovers_known_parameters():
    nu_true, scale_true = 6.0, 0.02
    x = stats.t.rvs(nu_true, size=200_000, random_state=1) * scale_true
    nu, scale = fit_t(x)
    assert nu == pytest.approx(nu_true, rel=0.15)
    assert scale == pytest.approx(scale_true, rel=0.02)


def test_fit_t_caps_nu_on_normal_data():
    x = np.random.default_rng(2).standard_normal(20_000)
    assert fit_t(x, nu_max=100.0)[0] <= 100.0


def test_risk_contributions_sum_to_one(returns, weights):
    pc = risk_contributions(returns.cov(), weights)
    assert pc.sum() == pytest.approx(1.0)


def test_risk_contributions_equal_under_symmetry():
    """Exercise 5.1: equal rows of Sigma give equal contributions."""
    k = 4
    sigma = 0.5 * np.ones((k, k)) + 0.5 * np.eye(k)
    pc = risk_contributions(sigma, np.full(k, 1 / k))
    assert pc == pytest.approx(np.full(k, 1 / k))


def test_risk_contribution_matches_numerical_derivative(returns, weights):
    """Euler: w_i * d(sigma_p)/d(w_i) / sigma_p."""
    sigma = returns.cov().to_numpy()
    w = np.asarray(weights)
    vol = lambda v: np.sqrt(v @ sigma @ v)

    eps = 1e-7
    bumped = w.copy(); bumped[0] += eps
    marginal = (vol(bumped) - vol(w)) / eps
    assert w[0] * marginal / vol(w) == pytest.approx(
        risk_contributions(sigma, w)[0], rel=1e-5)


def test_ewma_variance_follows_the_recursion(losses):
    lam = 0.94
    v = ewma_variance(losses, lam).dropna()
    lhs = v.iloc[1:].to_numpy()
    rhs = lam * v.iloc[:-1].to_numpy() + (1 - lam) * losses.loc[v.index[:-1]].to_numpy() ** 2
    assert lhs == pytest.approx(rhs)


def test_ewma_covariance_reproduces_univariate(returns, weights, losses):
    """Proposition 8.4: w' Sigma w equals the univariate EWMA variance."""
    S = ewma_covariance(returns)
    port = np.einsum("i,tij,j->t", weights, S, weights)
    uni = ewma_variance(losses).to_numpy()
    assert port[400:] == pytest.approx(uni[400:], rel=1e-9)


def test_ewma_is_more_variable_than_a_fixed_window(losses):
    fixed = rolling_normal(losses, window=250)["VaR"]
    ewma = rolling_ewma(losses)["VaR"]
    common = fixed.index.intersection(ewma.index)
    assert ewma[common].std() > fixed[common].std()
