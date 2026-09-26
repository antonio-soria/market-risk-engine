"""Returns and portfolio losses."""

import numpy as np


def simple_returns(prices):
    """Simple returns R_t = P_t / P_{t-1} - 1.

    These aggregate exactly across assets, which is why the portfolio uses
    them: R_p = w'R holds with no approximation.
    """
    return prices.pct_change().dropna()


def log_returns(prices):
    """Log returns r_t = ln(P_t / P_{t-1}).

    These aggregate exactly over time, so they belong in time-series models
    rather than in cross-sectional portfolio arithmetic.
    """
    return np.log(prices / prices.shift(1)).dropna()


def portfolio_losses(returns, weights):
    """Portfolio loss series L_t = -w'R_t, with losses positive.

    Weights are constant, which assumes the portfolio is rebalanced to target
    at every close with no transaction costs.
    """
    w = np.asarray(weights, dtype=float)
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must sum to 1")
    return -(returns @ w).rename("loss")
