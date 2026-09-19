import numpy as np
import pandas as pd

from .volatility import ewma_variance


def hs_var(losses, alpha=0.99):
    return np.quantile(np.asarray(losses), alpha, method="inverted_cdf")

def hs_es(losses, alpha=0.99):
    x = np.asarray(losses)
    return x[x >= hs_var(x, alpha)].mean()

def rolling_hs(losses, window=500, alpha=0.99):
    roll = losses.rolling(window)
    var = roll.apply(hs_var, raw=True, kwargs={"alpha": alpha}).shift(1)
    es = roll.apply(hs_es, raw=True, kwargs={"alpha": alpha}).shift(1)
    return pd.DataFrame({"VaR": var, "ES": es}).dropna()


def rolling_fhs(losses, lam=0.94, window=500, alpha=0.99):
    """Filtered HS: HS on EWMA-standardised losses, rescaled by tomorrow's vol."""
    sigma = np.sqrt(ewma_variance(losses, lam))
    z = losses / sigma
    roll = z.rolling(window)
    zq = roll.apply(hs_var, raw=True, kwargs={"alpha": alpha}).shift(1)
    ze = roll.apply(hs_es, raw=True, kwargs={"alpha": alpha}).shift(1)
    return pd.DataFrame({"VaR": sigma * zq, "ES": sigma * ze}).dropna()
