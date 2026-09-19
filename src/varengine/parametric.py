import numpy as np
import pandas as pd
from scipy import stats

from .volatility import ewma_variance


def normal_var_es(sigma, alpha=0.99, mu=0.0):
    """VaR and ES of a N(mu, sigma^2) loss (Proposition 3.4)."""
    z = stats.norm.ppf(alpha)
    var = mu + sigma * z
    es = mu + sigma * stats.norm.pdf(z) / (1 - alpha)
    return var, es


def t_var_es(scale, nu, alpha=0.99, loc=0.0):
    """VaR and ES of loc + scale * T, with T ~ Student-t(nu)."""
    q = stats.t.ppf(alpha, nu)
    var = loc + scale * q
    es = loc + scale * stats.t.pdf(q, nu) * (nu + q**2) / ((nu - 1) * (1 - alpha))
    return var, es


def t_scale_from_sigma(sigma, nu):
    """Scale of a t(nu) whose standard deviation is sigma."""
    return sigma * np.sqrt((nu - 2) / nu)


def fit_t(x, nu_max=100.0):
    """MLE of (nu, scale) for a zero-location Student-t.

    nu is capped at nu_max: beyond that the t is indistinguishable from a
    normal and the likelihood is flat, so the raw estimate can run off to 1e9.
    """
    nu, _, scale = stats.t.fit(np.asarray(x), floc=0)
    return min(nu, nu_max), scale


def rolling_normal(losses, window=500, alpha=0.99):
    """Zero-mean normal VaR/ES with sigma from a rolling window."""
    sigma = np.sqrt((losses**2).rolling(window).mean()).shift(1)
    var, es = normal_var_es(sigma, alpha)
    return pd.DataFrame({"VaR": var, "ES": es}).dropna()


def rolling_t(losses, window=500, alpha=0.99, refit_every=1):
    """Zero-location Student-t VaR/ES, (nu, scale) re-fitted by MLE."""
    x = losses.to_numpy()
    idx, rows, params = [], [], None
    for i, t in enumerate(range(window, len(x))):
        if params is None or i % refit_every == 0:
            params = fit_t(x[t - window:t])
        nu, scale = params
        idx.append(losses.index[t])
        rows.append((*t_var_es(scale, nu, alpha), nu))
    return pd.DataFrame(rows, index=idx, columns=["VaR", "ES", "nu"])


def risk_contributions(Sigma, w):
    """Euler percentage contributions to portfolio volatility (sum to 1)."""
    w = np.asarray(w, dtype=float)
    Sw = np.asarray(Sigma) @ w
    return w * Sw / (w @ Sw)


def rolling_ewma(losses, lam=0.94, alpha=0.99, dist="normal", nu=None):
    """EWMA volatility with normal or standardised-t innovations."""
    sigma = np.sqrt(ewma_variance(losses, lam))
    if dist == "normal":
        var, es = normal_var_es(sigma, alpha)
    elif dist == "t":
        var, es = t_var_es(t_scale_from_sigma(sigma, nu), nu, alpha)
    else:
        raise ValueError("dist must be 'normal' or 't'")
    return pd.DataFrame({"VaR": var, "ES": es}).dropna()
