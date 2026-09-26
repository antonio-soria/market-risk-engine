import numpy as np
import pandas as pd
import pytest

from varengine.returns import simple_returns, log_returns, portfolio_losses


def test_simple_returns_known_values():
    prices = pd.DataFrame({"X": [100.0, 110.0, 99.0]})
    assert simple_returns(prices)["X"].tolist() == pytest.approx([0.10, -0.10])


def test_log_returns_add_over_time():
    """Proposition 2.2: log returns over h days sum to the total log return."""
    prices = pd.DataFrame({"X": [100.0, 110.0, 99.0, 120.0]})
    r = log_returns(prices)["X"]
    assert r.sum() == pytest.approx(np.log(120 / 100))


def test_portfolio_return_is_weighted_sum(returns, weights):
    """Proposition 2.3: exact for simple returns."""
    loss = portfolio_losses(returns, weights)
    assert loss.to_numpy() == pytest.approx(-(returns.to_numpy() @ weights))


def test_weighted_log_returns_understate(returns, weights):
    """Proposition 2.4: Jensen's inequality, on every single day."""
    prices = 100 * (1 + returns).cumprod()
    exact = -portfolio_losses(returns, weights)
    approx = np.exp(log_returns(prices) @ weights) - 1
    gap = (exact - approx).dropna()
    assert (gap >= -1e-15).all()


def test_weights_must_sum_to_one(returns):
    with pytest.raises(ValueError):
        portfolio_losses(returns, [0.5, 0.2, 0.2, 0.2])


def test_losses_are_negated_returns(returns, weights):
    loss = portfolio_losses(returns, weights)
    assert loss.name == "loss"
    assert loss.mean() == pytest.approx(-(returns @ weights).mean())
