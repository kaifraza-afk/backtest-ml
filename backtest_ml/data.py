"""Data loading: yfinance (with CSV cache), local CSV, or synthetic data for smoke tests."""
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Data is missing columns: {missing}")
    df = df[COLUMNS].astype(float)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.dropna()
    return df[(df["open"] > 0) & (df["close"] > 0)]


def load_yfinance(ticker: str, start: str, end: str, cache_dir: str = "data") -> pd.DataFrame:
    cache = Path(cache_dir) / f"{ticker}_{start}_{end}.csv"
    if cache.exists():
        return _clean(pd.read_csv(cache, index_col=0, parse_dates=True))
    import yfinance as yf  # imported lazily so tests run without network

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(
            f"No data returned for {ticker}. Check internet/ticker, or use "
            "--source csv --csv-path <file> or --source synthetic."
        )
    if isinstance(raw.columns, pd.MultiIndex):  # newer yfinance returns (field, ticker)
        raw.columns = raw.columns.get_level_values(0)
    df = _clean(raw)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df


def load_csv(path: str) -> pd.DataFrame:
    return _clean(pd.read_csv(path, index_col=0, parse_dates=True))


def make_synthetic(n_days: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Random-walk prices with slowly switching drift and mild momentum.
    ONLY for smoke-testing the pipeline. Never report results from this."""
    rng = np.random.default_rng(seed)
    drift = np.zeros(n_days)
    state = 0.0004
    for i in range(n_days):
        if rng.random() < 0.01:
            state = rng.choice([-0.0004, 0.0001, 0.0006])
        drift[i] = state
    rets = np.zeros(n_days)
    for i in range(1, n_days):
        rets[i] = drift[i] + 0.05 * rets[i - 1] + rng.normal(0, 0.01)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.002, n_days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n_days)))
    volume = rng.integers(1_000_000, 5_000_000, n_days).astype(float)
    idx = pd.bdate_range("2010-01-01", periods=n_days)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def load_data(source: str, ticker: str, start: str, end: str, csv_path: str | None = None) -> pd.DataFrame:
    if source == "yfinance":
        return load_yfinance(ticker, start, end)
    if source == "csv":
        if not csv_path:
            raise ValueError("--csv-path is required with --source csv")
        return load_csv(csv_path)
    if source == "synthetic":
        return make_synthetic()
    raise ValueError(f"Unknown source: {source}")
