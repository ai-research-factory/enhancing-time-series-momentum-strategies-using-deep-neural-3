"""
Data loading and preprocessing for deep momentum network.

Fetches OHLCV from the ARF Data API, computes daily returns, and creates
lookback-window features for the MLP model.
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
API_BASE = "https://ai.1s.xyz/api/data/ohlcv"


def fetch_ohlcv(ticker: str = "^N225", interval: str = "1d", period: str = "10y",
                cache: bool = True) -> pd.DataFrame:
    """
    Fetch OHLCV data from ARF Data API (or local cache).

    Returns DataFrame indexed by timestamp with columns:
    open, high, low, close, volume.
    """
    safe_name = ticker.replace("/", "_").replace("^", "")
    cache_path = os.path.join(DATA_DIR, f"{safe_name}_{interval}.csv")

    if cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
    else:
        url = f"{API_BASE}?ticker={ticker}&interval={interval}&period={period}"
        df = pd.read_csv(url)
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(cache_path, index=False)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


def prepare_features(df: pd.DataFrame, lookback: int = 60) -> tuple:
    """
    Compute daily returns and create lookback-window feature matrix.

    Args:
        df: OHLCV DataFrame indexed by timestamp
        lookback: number of past return days to use as features

    Returns:
        (features, forward_returns, dates) where:
        - features: (N, lookback) array of past returns
        - forward_returns: (N,) array of next-day returns (for training signal)
        - dates: DatetimeIndex of length N
    """
    close = df["close"].copy()
    # Daily log returns
    log_returns = np.log(close / close.shift(1))

    features = []
    forward_returns = []
    dates = []

    for i in range(lookback, len(log_returns) - 1):
        # Features: returns from t-lookback to t-1
        window = log_returns.iloc[i - lookback:i].values
        if np.isnan(window).any():
            continue
        # Forward return: return at t (used as trading signal target)
        fwd = log_returns.iloc[i + 1]
        if np.isnan(fwd):
            continue
        features.append(window)
        forward_returns.append(fwd)
        dates.append(log_returns.index[i])

    return (
        np.array(features, dtype=np.float32),
        np.array(forward_returns, dtype=np.float32),
        pd.DatetimeIndex(dates),
    )
