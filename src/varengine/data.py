"""Price download and local caching."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_prices(tickers, start, end=None, cache="prices.csv", refresh=False):
    """Adjusted close prices, downloaded once and cached to CSV.

    Parameters
    ----------
    tickers : list of str
        Yahoo Finance symbols, e.g. ``["SAN.MC", "BBVA.MC"]``.
    start, end : str or None
        Sample bounds. ``end=None`` downloads up to the latest completed
        trading day.
    cache : str
        File name inside ``data/``. The cache freezes the snapshot so that
        results stay reproducible.
    refresh : bool
        Ignore the cache and download again.

    Returns
    -------
    pandas.DataFrame
        One column per ticker, in the order requested, indexed by date.
    """
    path = DATA_DIR / cache
    if path.exists() and not refresh:
        px = pd.read_csv(path, index_col=0, parse_dates=True)
        if set(px.columns) != set(tickers):
            raise ValueError("Cached tickers differ from those requested. "
                             "Call load_prices(..., refresh=True).")
        print(f"Loaded cached prices: {px.index[0].date()} to {px.index[-1].date()}")
        return px[tickers]

    # Imported here, not at module level: yfinance is only needed to download,
    # so `import varengine` works without it and the tests stay offline.
    import yfinance as yf

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
