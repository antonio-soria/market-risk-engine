"""A Value-at-Risk and Expected Shortfall engine for equity portfolios.

Typical use::

    from varengine import load_prices, simple_returns, portfolio_losses
    from varengine import rolling_fhs, backtest

    prices = load_prices(TICKERS, START, END)
    loss = portfolio_losses(simple_returns(prices), weights)
    result = backtest(loss, rolling_fhs(loss))

Every ``rolling_*`` function returns a DataFrame with columns ``VaR`` and
``ES``, indexed by the forecast date, built only from data available at the
previous close.
"""

from .config import TICKERS, START, END
from .data import load_prices
from .returns import simple_returns, log_returns, portfolio_losses
from .volatility import ewma_variance, ewma_covariance
from .historical import hs_var, hs_es, rolling_hs, rolling_fhs
from .parametric import (normal_var_es, t_var_es, t_scale_from_sigma, fit_t,
                         rolling_normal, rolling_t, rolling_ewma,
                         risk_contributions)
from .montecarlo import draw_innovations, mc_var_es, rolling_mc
from .backtest import (violations, kupiec, christoffersen_independence,
                       christoffersen_cc, basel_zone, es_backtest,
                       backtest, backtest_all, common_index)

__version__ = "1.0.0"

__all__ = [
    "TICKERS", "START", "END", "load_prices",
    "simple_returns", "log_returns", "portfolio_losses",
    "ewma_variance", "ewma_covariance",
    "hs_var", "hs_es", "rolling_hs", "rolling_fhs",
    "normal_var_es", "t_var_es", "t_scale_from_sigma", "fit_t",
    "rolling_normal", "rolling_t", "rolling_ewma", "risk_contributions",
    "draw_innovations", "mc_var_es", "rolling_mc",
    "violations", "kupiec", "christoffersen_independence",
    "christoffersen_cc", "basel_zone", "es_backtest",
    "backtest", "backtest_all", "common_index",
]
