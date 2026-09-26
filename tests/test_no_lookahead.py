"""The most important test in the suite.

A forecast dated t may only use information available at the close of t-1.
If that is violated the backtest flatters the model and every result is void.

The check: corrupt the loss on one day, recompute, and require that no
forecast dated on or before that day changes.
"""

import numpy as np
import pytest

from varengine.historical import rolling_hs, rolling_fhs
from varengine.parametric import rolling_normal, rolling_t, rolling_ewma
from varengine.montecarlo import rolling_mc

SHOCK_POS = -60          # corrupt a day near the end, well after the burn-in
WINDOW = 250

LOSS_MODELS = {
    "rolling_hs": lambda x: rolling_hs(x, window=WINDOW),
    "rolling_fhs": lambda x: rolling_fhs(x, window=WINDOW),
    "rolling_normal": lambda x: rolling_normal(x, window=WINDOW),
    "rolling_t": lambda x: rolling_t(x, window=WINDOW, refit_every=25),
    "rolling_ewma": lambda x: rolling_ewma(x),
}


@pytest.mark.parametrize("name", list(LOSS_MODELS))
def test_loss_models_do_not_peek(name, losses):
    model = LOSS_MODELS[name]
    shock_date = losses.index[SHOCK_POS]

    base = model(losses)
    corrupted = losses.copy()
    corrupted.iloc[SHOCK_POS] *= 10          # a day ten times worse
    after = model(corrupted)

    past = base.index[base.index <= shock_date]
    assert not past.empty
    assert base.loc[past].to_numpy() == pytest.approx(after.loc[past].to_numpy())


def test_loss_models_do_react_afterwards(losses):
    """Sanity check on the test itself: the shock must change something."""
    shock_date = losses.index[SHOCK_POS]
    corrupted = losses.copy()
    corrupted.iloc[SHOCK_POS] *= 10

    base, after = rolling_hs(losses, window=WINDOW), rolling_hs(corrupted, window=WINDOW)
    future = base.index[base.index > shock_date]
    assert not np.allclose(base.loc[future, "VaR"], after.loc[future, "VaR"])


def test_monte_carlo_does_not_peek(returns, weights):
    shock_date = returns.index[SHOCK_POS]

    base = rolling_mc(returns, weights, n_sims=2_000, start=WINDOW)
    corrupted = returns.copy()
    corrupted.iloc[SHOCK_POS] *= 10
    after = rolling_mc(corrupted, weights, n_sims=2_000, start=WINDOW)

    past = base.index[base.index <= shock_date]
    assert base.loc[past].to_numpy() == pytest.approx(after.loc[past].to_numpy())
