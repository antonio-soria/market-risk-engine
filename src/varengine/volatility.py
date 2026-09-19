import numpy as np
import pandas as pd


def ewma_variance(losses, lam=0.94):
    """sigma^2_{t|t-1}: EWMA variance forecast for day t, made at close t-1."""
    s = (losses**2).ewm(alpha=1 - lam, adjust=False).mean()   # s_t uses data up to t
    return s.shift(1).rename("ewma_var")


def ewma_covariance(returns, lam=0.94, seed_days=30):
    """Array S with S[t] = Sigma_{t|t-1}, the EWMA covariance forecast for day t."""
    R = returns.to_numpy()
    T, k = R.shape
    S = np.empty((T, k, k))
    S[0] = R[:seed_days].T @ R[:seed_days] / seed_days
    for t in range(1, T):
        r = R[t - 1][:, None]
        S[t] = lam * S[t - 1] + (1 - lam) * (r @ r.T)
    return S
