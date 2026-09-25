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

## Backtesting

A 99% VaR model makes two promises: losses will be greater than the forecast on about 1% of days, and those days will be scattered at random rather than clustered together. Backtesting checks both.

Each day is marked 1 if the loss beat the VaR forecast (a violation) and 0 if it did not. That gives a long string of zeros and ones, and the tests ask two questions about it:

- **Kupiec's test**: are there the right number of violations?
- **Christoffersen's test**: does a violation make another violation more likely the next day?

Put together they give the conditional coverage test, which checks both at once. All three are likelihood ratio tests.

The engine also reports the Basel traffic light, which is how regulators classify a model based on its violations over the last 250 days, and the Acerbi–Székely Z₂ statistic. Z₂ checks the ES instead of the VaR: 0 means the ES is right, and above 0 means the model understates how bad the bad days are.

## Results

Backtested on 2,494 trading days, December 2016 to September 2026, at 99% confidence. Expected violations: 24.9. Bold means the model is rejected at the 5% level.

| Model | Violations | Rate | p (rate) | p (clustering) | p (both) | Basel zone | Z₂ |
|---|---|---|---|---|---|---|---|
| HS | 28 | 1.12% | 0.546 | **0.0000** | **0.0000** | green | 0.32 |
| Normal | 39 | 1.56% | **0.0089** | **0.0000** | **0.0000** | green | 1.18 |
| Student-t | 29 | 1.16% | 0.426 | **0.0000** | **0.0000** | green | 0.47 |
| EWMA-normal | 55 | 2.21% | **0.0000** | 0.160 | **0.0000** | yellow | 1.55 |
| **EWMA-t** | 33 | 1.32% | 0.122 | 0.458 | **0.230** | green | 0.36 |
| **Filtered HS** | 35 | 1.40% | 0.056 | 0.517 | **0.131** | green | 0.39 |
| Monte Carlo normal | 55 | 2.21% | **0.0000** | 0.160 | **0.0000** | yellow | 1.58 |
| Monte Carlo t | 35 | 1.40% | 0.056 | 0.098 | **0.041** | green | 0.45 |


**1. The timing of the violations matters more than their frequency.** Historical Simulation gets the count almost right: 1.12% against a 1% target, and it passes Kupiec easily. It still fails, because of when those violations happen. The three fixed-window models put about half of all their violations into 2020, a year that is only 10% of the sample.

| Violations in 2020 | HS | Normal | Student-t | EWMA-N | EWMA-t | FHS |
|---|---|---|---|---|---|---|
| Count | 14 | 19 | 15 | 6 | 4 | 4 |
| Share of the model's violations | 50% | 49% | 52% | 11% | 12% | 11% |

A 500 day window gives every past day the same weight, so it takes months to notice that the market has changed. The model ends up being right on average and wrong exactly when it matters. The EWMA models spread their violations roughly in line with the number of days.

**2. Reacting faster does not help if the tails are still normal.** EWMA-normal and Monte Carlo normal have 55 violations against 24.9 expected, more than twice the target, and they are the only two models in the Basel yellow zone. Their Z₂ of about 1.55 says that on the days they were breached, the loss was well above the ES they had forecast.

EWMA-normal actually has more violations than the plain normal model (55 against 39). The reason is that the plain model's 500 day volatility is too high during calm periods, because it still remembers the last crisis. That accidental effect hides how thin the normal tails are. EWMA takes this effect away and the tail problem shows up in full.

**3. Both fixes are needed.** Only EWMA-t and Filtered Historical Simulation pass the joint test. EWMA-normal reacts quickly but has the wrong tail. Historical Simulation and the Student-t have a reasonable tail but react too slowly. Each one fixes half the problem and fails.

**4. The Monte Carlo models agree with the formulas, which is the point of having them.** Monte Carlo normal gives exactly the same violations as EWMA-normal. Monte Carlo t differs from EWMA-t by two violations, which is what you would expect from simulating 50,000 scenarios: the sampling error on a 99% quantile is about 1.2% of the VaR for a Student-t, against 0.7% for a normal.

### Running the same test at 95%

Lowering the confidence level to 95% raises the expected violations from 24.9 to 124.7. With five times as many violations, both tests become much better at spotting a bad model.

| Model | Violations | Rate | p (rate) | p (clustering) | p (both) |
|---|---|---|---|---|---|
| HS | 110 | 4.41% | 0.168 | **0.0003** | **0.0006** |
| Normal | 96 | 3.85% | **0.0061** | **0.0000** | **0.0000** |
| Student-t | 109 | 4.37% | 0.141 | **0.0000** | **0.0000** |
| EWMA-normal | 128 | 5.13% | 0.763 | **0.0001** | **0.0004** |
| EWMA-t | 144 | 5.77% | 0.083 | **0.0001** | **0.0001** |
| Filtered HS | 130 | 5.21% | 0.629 | **0.0004** | **0.0016** |
| Monte Carlo normal | 128 | 5.13% | 0.763 | **0.0001** | **0.0004** |
| Monte Carlo t | 144 | 5.77% | 0.083 | **0.0001** | **0.0001** |

Two things change:

**The normal and the Student-t swap places.** With the degrees of freedom estimated at about 5.2, the two distributions give the same VaR at 97.0% confidence. Above that the Student-t is more conservative, below it the normal is. The violation counts follow: EWMA-normal against EWMA-t is 55 to 33 at 99%, and 128 to 144 at 95%. EWMA-normal goes from the worst model on the count to the best one. So which model looks best depends on the confidence level you test it at, not just on the model.

**Every model now fails the clustering test, including the ones that passed at 99%.** The models have not gotten worse; the test has gotten sharper. The clustering was there all along and the 99% test could not see it. Comparing the statistics against the same simulation suggests every model still has clustering of roughly 2.5 to 3 times. EWMA at λ = 0.94 cuts the clustering down a lot compared with a fixed window, but it does not get rid of it.


## Repository structure

```
market-risk-engine/
├── notebooks/
│   ├── 00_load_prices.ipynb             # download or refresh the price snapshot
│   ├── 01_returns_and_losses.ipynb      # data, returns, stylised facts
│   ├── 02_historical_simulation.ipynb   # historical simulation and first backtest
│   └── 03_parametric_and_mc.ipynb       # parametric, EWMA, filtered HS, Monte Carlo
│   └── 04_backtesting.ipynb             # Kupiec, Christoffersen, Basel, ES
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

Then open the notebooks in order. The first run downloads prices and caches them in `data/`. It should be ran after building the portfolio and selecting the date range used by editing `src/varengine/config.py`.

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
