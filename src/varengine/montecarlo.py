import numpy as np
import pandas as pd

from .historical import hs_var, hs_es
from .volatility import ewma_covariance


def draw_innovations(n_sims, k, dist="normal", nu=None, seed=42):
    """n_sims x k draws with zero mean and identity covariance."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_sims, k))
    if dist == "t":
        W = rng.chisquare(nu, size=n_sims)
        Z *= np.sqrt((nu - 2) / W)[:, None]      # multivariate t, unit variance
    elif dist != "normal":
        raise ValueError("dist must be 'normal' or 't'")
    return Z


def mc_var_es(Sigma, w, alpha=0.99, Z=None, **kwargs):
    """Simulate R = C Z with C C' = Sigma, then read VaR/ES off L = -w'R."""
    if Z is None:
        Z = draw_innovations(n_sims=100_000, k=len(w), **kwargs)
    C = np.linalg.cholesky(Sigma)
    L = -(Z @ C.T) @ np.asarray(w)
    return hs_var(L, alpha), hs_es(L, alpha)


def rolling_mc(returns, w, lam=0.94, alpha=0.99, n_sims=50_000,
               dist="normal", nu=None, start=500, seed=42):
    """Monte Carlo VaR/ES with EWMA covariance and common random numbers."""
    Z = draw_innovations(n_sims, returns.shape[1], dist, nu, seed)
    S = ewma_covariance(returns, lam)
    rows = [mc_var_es(S[t], w, alpha, Z) for t in range(start, len(returns))]
    return pd.DataFrame(rows, index=returns.index[start:], columns=["VaR", "ES"])
