import numpy as np

def simple_returns(prices):
    return prices.pct_change().dropna()


def log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


def portfolio_losses(returns, weights):
    """L_t = -w'R_t, constant weights (daily rebalancing)."""
    w = np.asarray(weights, dtype=float)
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must sum to 1")
    return -(returns @ w).rename("loss")