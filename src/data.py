"""
Data loading and preprocessing for deep momentum network.

Fetches OHLCV from the ARF Data API, computes daily returns, and creates
lookback-window features for the model. Supports both single-asset (legacy)
and multi-asset pipelines.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
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


class DataLoader:
    """
    Multi-asset data loader for deep momentum network.

    Reads ticker list from config/assets.json, fetches OHLCV data via
    the ARF Data API, computes log returns, handles NaN values, and
    creates rolling-window features suitable for model input.
    """

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(CONFIG_DIR, "assets.json")
        with open(config_path) as f:
            self.config = json.load(f)
        self.tickers = self.config["tickers"]
        self.interval = self.config.get("interval", "1d")
        self.period = self.config.get("period", "10y")
        self.lookback = self.config.get("lookback", 60)
        self.raw_data = {}
        self.returns_df = None

    def fetch_all(self) -> dict:
        """Fetch OHLCV data for all tickers. Returns dict of DataFrames."""
        for ticker in self.tickers:
            print(f"Fetching {ticker}...")
            try:
                df = fetch_ohlcv(
                    ticker=ticker,
                    interval=self.interval,
                    period=self.period,
                    cache=True,
                )
                if len(df) > 0:
                    self.raw_data[ticker] = df
                    print(f"  {ticker}: {len(df)} rows, "
                          f"{df.index[0].date()} to {df.index[-1].date()}")
                else:
                    print(f"  {ticker}: no data returned, skipping")
            except Exception as e:
                print(f"  {ticker}: fetch failed ({e}), skipping")
        return self.raw_data

    def compute_returns(self) -> pd.DataFrame:
        """
        Compute daily log returns for all fetched assets.

        Returns a DataFrame with tickers as columns and dates as index.
        NaN handling: forward-fill prices before computing returns,
        then drop any remaining leading NaNs.
        """
        close_frames = {}
        for ticker, df in self.raw_data.items():
            close = df["close"].copy()
            # Forward-fill missing prices
            close = close.ffill()
            close_frames[ticker] = close

        # Align all tickers to a common date index
        close_df = pd.DataFrame(close_frames)
        # Forward-fill across the aligned frame to handle missing dates
        close_df = close_df.ffill()

        # Compute log returns
        self.returns_df = np.log(close_df / close_df.shift(1))
        # Drop the first row (NaN from shift)
        self.returns_df = self.returns_df.iloc[1:]

        return self.returns_df

    def create_rolling_windows(self) -> tuple:
        """
        Create 3D rolling-window feature array for model input.

        Returns:
            (features, forward_returns, dates, tickers) where:
            - features: (N, lookback, n_assets) ndarray
            - forward_returns: (N, n_assets) ndarray
            - dates: DatetimeIndex of length N
            - tickers: list of ticker names (column order)
        """
        if self.returns_df is None:
            self.compute_returns()

        returns = self.returns_df
        lookback = self.lookback
        n_assets = returns.shape[1]
        tickers = list(returns.columns)

        features = []
        fwd_returns = []
        dates = []

        for i in range(lookback, len(returns) - 1):
            # Window: returns from i-lookback to i-1 (past data only)
            window = returns.iloc[i - lookback:i].values  # (lookback, n_assets)
            # Forward returns at i+1
            fwd = returns.iloc[i + 1].values  # (n_assets,)

            # Skip if any NaN in window or forward returns
            if np.isnan(window).any() or np.isnan(fwd).any():
                continue

            features.append(window)
            fwd_returns.append(fwd)
            dates.append(returns.index[i])

        features_arr = np.array(features, dtype=np.float32)  # (N, lookback, n_assets)
        fwd_arr = np.array(fwd_returns, dtype=np.float32)    # (N, n_assets)

        return features_arr, fwd_arr, pd.DatetimeIndex(dates), tickers

    def get_data_summary(self) -> pd.DataFrame:
        """
        Generate summary statistics for each asset.

        Returns DataFrame with columns:
        ticker, start_date, end_date, n_rows, n_missing_close, mean_return, std_return
        """
        rows = []
        for ticker, df in self.raw_data.items():
            close = df["close"]
            log_ret = np.log(close / close.shift(1)).iloc[1:]
            rows.append({
                "ticker": ticker,
                "start_date": df.index[0].date(),
                "end_date": df.index[-1].date(),
                "n_rows": len(df),
                "n_missing_close": int(close.isna().sum()),
                "mean_daily_return": round(float(log_ret.mean()), 6),
                "std_daily_return": round(float(log_ret.std()), 6),
                "annualized_return": round(float(log_ret.mean() * 252), 4),
                "annualized_vol": round(float(log_ret.std() * np.sqrt(252)), 4),
            })
        return pd.DataFrame(rows)

    def save_processed(self, output_path: str = None) -> str:
        """
        Run full pipeline and save processed data as pickle.

        Saved dict contains:
            features: (N, lookback, n_assets) ndarray
            forward_returns: (N, n_assets) ndarray
            dates: DatetimeIndex
            tickers: list of str
            lookback: int

        Returns the output path.
        """
        if output_path is None:
            output_path = os.path.join(PROCESSED_DIR, "timeseries.pkl")

        features, fwd_returns, dates, tickers = self.create_rolling_windows()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = {
            "features": features,
            "forward_returns": fwd_returns,
            "dates": dates,
            "tickers": tickers,
            "lookback": self.lookback,
        }
        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved processed data to {output_path}")
        print(f"  Features shape: {features.shape}")
        print(f"  Forward returns shape: {fwd_returns.shape}")
        print(f"  Date range: {dates[0].date()} to {dates[-1].date()}")
        print(f"  Tickers: {tickers}")

        return output_path
