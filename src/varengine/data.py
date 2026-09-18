from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_prices(tickers, start, end, cache="prices.csv"):
    """Adjusted close prices, downloaded once and cached to CSV."""
    path = DATA_DIR / cache
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)

    px = yf.download(tickers, start=start, end=end, auto_adjust=True,
                     progress=False, threads=False)["Close"]

    missing = [t for t in tickers if t not in px or px[t].isna().all()]
    if missing:
        raise RuntimeError(f"Download failed for {missing}; nothing cached. Try again.")

    px = px.dropna(how="all").ffill().dropna()
    path.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(path)
    return px