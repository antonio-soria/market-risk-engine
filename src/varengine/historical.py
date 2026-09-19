import numpy as np
import pandas as pd


def hs_var(losses, alpha=0.99):
    """Empirical alpha-quantile: the ceil(n*alpha)-th order statistic."""
    return np.quantile(np.asarray(losses), alpha, method="inverted_cdf")


def hs_es(losses, alpha=0.99):
    """Average of losses at or beyond the HS VaR."""
    x = np.asarray(losses)
    return x[x >= hs_var(x, alpha)].mean()


def rolling_hs(losses, window=500, alpha=0.99):
    """One-day-ahead HS forecasts. The value dated t+1 uses data up to t."""
    roll = losses.rolling(window)
    var = roll.apply(hs_var, raw=True, kwargs={"alpha": alpha}).shift(1)
    es = roll.apply(hs_es, raw=True, kwargs={"alpha": alpha}).shift(1)
    return pd.DataFrame({"VaR": var, "ES": es}).dropna()