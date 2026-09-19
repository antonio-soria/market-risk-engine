from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_prices(tickers, start, end=None, cache="prices.csv", refresh=False):
    """Adjusted close prices, downloaded once and cached to CSV.

    end=None downloads up to the latest completed trading day. The cache
    freezes that snapshot: pass refresh=True to download a newer one.
    """
    path = DATA_DIR / cache
    if path.exists() and not refresh:
        px = pd.read_csv(path, index_col=0, parse_dates=True)
        if set(px.columns) != set(tickers):
            raise ValueError("Cached tickers differ from those requested. "
                             "Call load_prices(..., refresh=True).")
        print(f"Loaded cached prices: {px.index[0].date()} to {px.index[-1].date()}")
        return px[tickers]

    px = yf.download(tickers, start=start, end=end, auto_adjust=True,
                     progress=False, threads=False)["Close"]
    missing = [t for t in tickers if t not in px or px[t].isna().all()]
    if missing:
        raise RuntimeError(f"Download failed for {missing}; nothing cached. Try again.")

    px = px[px.index.normalize() < pd.Timestamp.today().normalize()]
    px = px.dropna(how="all").ffill().dropna()[tickers]
    path.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(path)
    print(f"Downloaded prices: {px.index[0].date()} to {px.index[-1].date()}")
    return px
