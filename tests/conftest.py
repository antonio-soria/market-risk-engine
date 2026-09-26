"""Shared fixtures. Synthetic data keeps the tests fast and offline."""

import numpy as np
import pandas as pd
import pytest

from varengine.returns import portfolio_losses

N_DAYS, N_ASSETS = 1200, 4


@pytest.fixture(scope="session")
def returns():
    """Correlated, fat-tailed, volatility-clustered daily returns."""
    rng = np.random.default_rng(0)
    corr = 0.4 * np.ones((N_ASSETS, N_ASSETS)) + 0.6 * np.eye(N_ASSETS)
    chol = np.linalg.cholesky(corr)

    var_t, out = 1e-4, np.empty((N_DAYS, N_ASSETS))
    for t in range(N_DAYS):
        if t:
            var_t = 2e-6 + 0.08 * np.mean(out[t - 1] ** 2) + 0.90 * var_t
        shock = rng.standard_t(6, N_ASSETS) * np.sqrt(4 / 6)   # unit variance
        out[t] = np.sqrt(var_t) * (chol @ shock)

    return pd.DataFrame(out,
                        index=pd.bdate_range("2018-01-01", periods=N_DAYS),
                        columns=[f"A{i}" for i in range(N_ASSETS)])


@pytest.fixture(scope="session")
def weights():
    return np.full(N_ASSETS, 1 / N_ASSETS)


@pytest.fixture(scope="session")
def losses(returns, weights):
    return portfolio_losses(returns, weights)
