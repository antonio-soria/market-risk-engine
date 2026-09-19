# Market Risk Engine

Given an equity portfolio, this engine calculates Value at Risk (VaR) and Expected Shortfall (ES) measures. It produces one-day risk forecasts with historical, parametric and Monte Carlo models, and backtests them against realised losses. For every trading day in the test period, the engine uses only information available at the previous close to forecast tomorrow's 99% VaR and ES. It then compares those forecasts with the realised losses. Everything is built in Python.


## Current Portfolio and Data

- **Assets:** six large Spanish stocks listed in Madrid: Santander (SAN), BBVA, Iberdrola (IBE), Inditex (ITX), Telefónica (TEF) and Repsol (REP) (currency risk is not considered since all assets are in EUR).
- **Weights:** equal (1/6 each).
- **Data:** daily adjusted close prices from Yahoo Finance via the `yfinance` Python library, from January 2015 to the latest completed trading day at download time. Prices are cached in `data/`.

**To change the portfolio composition or the date range, edit `src/varengine/config.py`. To update the data to the latest close, run `notebooks/00_load_prices.ipynb`, which downloads a fresh snapshot with `load_prices(..., refresh=True)`, then re-run the other notebooks. Otherwise the cache keeps the existing snapshot, so results stay reproducible.**

## Modelling conventions

The engine works with simple returns, because they allow for the return of a portfolio to be exactly the weighted sum of its assets' returns. Losses are defined as minus the portfolio return, so a loss is a positive number and all risk measures are read from the right tail.

Weights are held constant, which means assuming the portfolio is rebalanced back to its set weights at every close, with no transaction costs. The horizon is one trading day and the default confidence level is 99%. Parametric and Monte Carlo models assume a zero daily mean (which is reasonable because at a one-day horizon the mean is very small compared with volatility, and its estimation error is larger than the mean itself).

Window-based models use the last 500 trading days. Every one-day forecast is strictly out of sample: the VaR and ES for day t+1 use data up to the close of day t.

## Models

Eight models are implemented, organised along two dimensions: **how volatility is estimated**, and **what shape is assumed for the loss distribution**.

- The first three use a fixed 500-day window, in which every past day carries equal weight. **Historical simulation (HS)** reads VaR and ES directly off the empirical distribution of past losses. The **normal (variance–covariance)** model fits a zero-mean normal distribution, and the **Student-t** model fits a fat-tailed t distribution whose degrees of freedom are re-estimated by maximum likelihood on each window.

- The next three let volatility change over time using an exponentially weighted moving average (EWMA), which gives recent days more weight and reacts quickly to market stress. **EWMA-normal** combines it with a normal distribution and **EWMA-t** with a standardised Student-t. **Filtered historical simulation** standardises past losses by their EWMA volatility, takes their empirical quantiles, and rescales them by tomorrow's volatility forecast. 

- The last two are **Monte Carlo** models. They simulate 50,000 scenarios of tomorrow's asset returns from a multivariate normal or multivariate Student-t distribution, and revalue the portfolio in each scenario.

The engine also computes Euler risk contributions, which show how much of the portfolio's risk each position is responsible for (beyond its weight).

## Backtesting (WIP)


## Repository structure

```
market-risk-engine/
├── notebooks/
│   ├── 00_load_prices.ipynb             # download or refresh the price snapshot
│   ├── 01_returns_and_losses.ipynb      # data, returns, stylised facts
│   ├── 02_historical_simulation.ipynb   # historical simulation and first backtest
│   └── 03_parametric_and_mc.ipynb       # parametric, EWMA, filtered HS, Monte Carlo
├── src/varengine/
│   ├── config.py        # tickers and sample period
│   ├── data.py          # download and cache prices
│   ├── returns.py       # returns and portfolio losses
│   ├── historical.py    # historical simulation and filtered HS
│   ├── parametric.py    # normal, Student-t, EWMA models, risk contributions
│   ├── volatility.py    # EWMA variance and covariance
│   ├── montecarlo.py    # Monte Carlo simulation
│   └── backtest.py      # coming next
├── pyproject.toml
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/antonio-soria/market-risk-engine.git
cd market-risk-engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Then open the notebooks in order. The first run downloads prices and caches them in `data/`.

## Quick example

```python
import numpy as np

from varengine.config import TICKERS, START, END
from varengine.data import load_prices
from varengine.returns import simple_returns, portfolio_losses
from varengine.historical import rolling_hs, rolling_fhs
from varengine.parametric import rolling_ewma

prices = load_prices(TICKERS, START, END)     # add refresh=True to update the snapshot
w = np.full(len(TICKERS), 1 / len(TICKERS))
loss = portfolio_losses(simple_returns(prices), w)

hs = rolling_hs(loss, window=500, alpha=0.99)    # DataFrame with columns VaR and ES
ewma = rolling_ewma(loss, lam=0.94, alpha=0.99)
fhs = rolling_fhs(loss, lam=0.94, window=500, alpha=0.99)

violations = loss.loc[fhs.index] > fhs["VaR"]
print(f"Filtered HS violation rate: {violations.mean():.2%} (target: 1%)")
```

## References

Glasserman, P. (2003). *Monte Carlo methods in financial engineering*. Springer.

Hull, J. C. (2023). *Risk management and financial institutions* (6th ed.). Wiley.

McNeil, A. J., Frey, R., & Embrechts, P. (2005). *Quantitative risk management: Concepts, techniques and tools*. Princeton University Press.

## Author

Antonio F. Soria Bollini, MSc Economics of Global Risks, Toulouse School of Economics
