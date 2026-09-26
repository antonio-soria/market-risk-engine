"""Historical simulation and filtered historical simulation."""

import numpy as np
import pandas as pd

from .volatility import ewma_variance


def hs_var(losses, alpha=0.99):
    """Empirical alpha-quantile: the ceil(n*alpha)-th order statistic.

    ``method="inverted_cdf"`` picks that order statistic exactly, rather than
    interpolating between neighbours as numpy does by default.
    """
    return np.quantile(np.asarray(losses), alpha, method="inverted_cdf")


def hs_es(losses, alpha=0.99):
    """Average of the losses at or beyond the historical-simulation VaR."""
    x = np.asarray(losses)
    return x[x >= hs_var(x, alpha)].mean()


def rolling_hs(losses, window=500, alpha=0.99):
    """One-day-ahead historical-simulation forecasts.

    The row dated t is computed from the ``window`` losses up to and including
    t-1, so it is a genuine forecast. Returns columns ``VaR`` and ``ES``.
    """
    roll = losses.rolling(window)
    var = roll.apply(hs_var, raw=True, kwargs={"alpha": alpha}).shift(1)
    es = roll.apply(hs_es, raw=True, kwargs={"alpha": alpha}).shift(1)
    return pd.DataFrame({"VaR": var, "ES": es}).dropna()


def rolling_fhs(losses, lam=0.94, window=500, alpha=0.99):
    """Filtered historical simulation.

    Standardise past losses by their EWMA volatility, take empirical quantiles
    of the standardised series, then rescale by tomorrow's volatility forecast.
    Keeps the empirical tail shape of historical simulation while reacting to
    changes in volatility.
    """
    sigma = np.sqrt(ewma_variance(losses, lam))        # sigma_{t|t-1}, dated t
    z = losses / sigma                                  # standardised losses
    roll = z.rolling(window)
    zq = roll.apply(hs_var, raw=True, kwargs={"alpha": alpha}).shift(1)
    ze = roll.apply(hs_es, raw=True, kwargs={"alpha": alpha}).shift(1)
    return pd.DataFrame({"VaR": sigma * zq, "ES": sigma * ze}).dropna()
