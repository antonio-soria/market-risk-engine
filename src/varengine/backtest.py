"""Backtesting VaR and ES forecasts: Kupiec, Christoffersen, Basel."""

import numpy as np
import pandas as pd
from scipy import stats


def violations(losses, var):
    """Indicator series I_t = 1{L_t > VaR_t} on the dates both series share."""
    idx = losses.index.intersection(var.index)
    return (losses.loc[idx] > var.loc[idx]).astype(int)


def _xlogy(n, p):
    """n * log(p), with the convention 0 * log(0) = 0."""
    return 0.0 if n == 0 else n * np.log(p)


def kupiec(hits, alpha=0.99):
    """Kupiec (1995) proportion-of-failures test. H0: violation rate = 1 - alpha."""
    h = np.asarray(hits)
    T, x = len(h), int(h.sum())
    p = 1 - alpha
    pi_hat = x / T

    ll_null = _xlogy(T - x, 1 - p) + _xlogy(x, p)
    ll_alt = _xlogy(T - x, 1 - pi_hat) + _xlogy(x, pi_hat)
    lr = -2 * (ll_null - ll_alt)
    return {"T": T, "violations": x, "expected": p * T, "rate": pi_hat,
            "LR_uc": lr, "p_uc": stats.chi2.sf(lr, 1)}


def transition_counts(hits):
    """n_ij = number of days with I_{t-1} = i and I_t = j."""
    h = np.asarray(hits)
    prev, cur = h[:-1], h[1:]
    return {(i, j): int(np.sum((prev == i) & (cur == j)))
            for i in (0, 1) for j in (0, 1)}


def christoffersen_independence(hits):
    """Christoffersen (1998) independence test. H0: violations do not cluster."""
    n = transition_counts(hits)
    n00, n01, n10, n11 = n[0, 0], n[0, 1], n[1, 0], n[1, 1]

    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    ll_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll_alt = (_xlogy(n00, 1 - pi01) + _xlogy(n01, pi01)
              + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11))
    lr = -2 * (ll_null - ll_alt)
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "pi01": pi01, "pi11": pi11,
            "LR_ind": lr, "p_ind": stats.chi2.sf(lr, 1)}


def christoffersen_cc(hits, alpha=0.99):
    """Conditional coverage: LR_cc = LR_uc + LR_ind, chi-squared with 2 df."""
    uc = kupiec(hits, alpha)
    ind = christoffersen_independence(hits)
    lr = uc["LR_uc"] + ind["LR_ind"]
    return {**uc, **ind, "LR_cc": lr, "p_cc": stats.chi2.sf(lr, 2)}


def basel_zone(hits, alpha=0.99, window=250):
    """Basel traffic light on the most recent `window` days (99%, 250 days)."""
    x = int(np.asarray(hits)[-window:].sum())
    cum = stats.binom.cdf(x, window, 1 - alpha)
    if cum < 0.95:
        zone, mult = "green", 3.00
    elif cum < 0.9999:
        zone = "yellow"
        mult = {5: 3.40, 6: 3.50, 7: 3.65, 8: 3.75, 9: 3.85}.get(x, 3.85)
    else:
        zone, mult = "red", 4.00
    return {"window": window, "violations": x, "zone": zone, "multiplier": mult}


def es_backtest(losses, forecasts, alpha=0.99):
    """Acerbi-Szekely (2014) Z2 statistic. Z2 = 0 if ES is right, > 0 if too low."""
    idx = losses.index.intersection(forecasts.index)
    L = losses.loc[idx]
    var, es = forecasts["VaR"].loc[idx], forecasts["ES"].loc[idx]
    exceed = L > var
    z2 = (L[exceed] / es[exceed]).sum() / (len(idx) * (1 - alpha)) - 1
    return {"n_exceed": int(exceed.sum()), "Z2": z2}


def backtest(losses, forecasts, alpha=0.99, basel_window=250):
    """Full backtest of one model's forecasts."""
    hits = violations(losses, forecasts["VaR"])
    out = christoffersen_cc(hits, alpha)
    out.update({f"basel_{k}": v for k, v in
                basel_zone(hits, alpha, basel_window).items()})
    out.update(es_backtest(losses, forecasts, alpha))
    return out


def common_index(losses, models):
    """Dates for which every model has a forecast."""
    idx = losses.index
    for m in models.values():
        idx = idx.intersection(m.index)
    return idx


def backtest_all(losses, models, alpha=0.99, basel_window=250):
    """Backtest every model in a {name: DataFrame} dict on common dates."""
    idx = common_index(losses, models)
    rows = {name: backtest(losses.loc[idx], m.loc[idx], alpha, basel_window)
            for name, m in models.items()}
    cols = ["violations", "expected", "rate", "LR_uc", "p_uc",
            "LR_ind", "p_ind", "LR_cc", "p_cc", "basel_zone", "Z2"]
    out = pd.DataFrame(rows).T[cols]

    # The string zone column makes every column `object`, which silently
    # disables .round() and sorting. Restore proper numeric dtypes.
    numeric = [c for c in cols if c != "basel_zone"]
    out[numeric] = out[numeric].astype(float)
    out["violations"] = out["violations"].astype(int)
    return out
