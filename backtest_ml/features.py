"""Features and labels.

Rule: the feature row at date t may only use data up to and including the close of t.
Label at date t = did price go up over the next `horizon` days (uses the future, so it
must never be used for anything except training on rows whose label is already known).
"""
import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    ret1 = close.pct_change()
    X = pd.DataFrame(index=df.index)
    for n in (1, 5, 10, 21):
        X[f"ret_{n}"] = close.pct_change(n)
    X["vol_10"] = ret1.rolling(10).std()
    X["vol_21"] = ret1.rolling(21).std()
    X["sma_ratio"] = close.rolling(10).mean() / close.rolling(50).mean() - 1
    X["rsi_14"] = rsi(close, 14)
    vol = df["volume"]
    vstd = vol.rolling(20).std()
    X["volume_z"] = ((vol - vol.rolling(20).mean()) / vstd.replace(0, np.nan)).fillna(0.0)
    return X.replace([np.inf, -np.inf], np.nan)


def build_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    fwd = df["close"].shift(-horizon) / df["close"] - 1
    y = (fwd > 0).astype(float)
    y[fwd.isna()] = np.nan
    return y
