# Technical Findings — Cycle 2: Data Pipeline Construction

## Implementation Summary

Phase 2 builds the multi-asset data pipeline that will feed the Deep Momentum Network in subsequent phases. The pipeline fetches daily OHLCV data for 20 liquid ETFs via the ARF Data API, computes log returns, and creates 3D rolling-window features.

## Key Components

### 1. `config/assets.json`
Defines 20 ETF tickers covering major asset classes:
- **Equity (broad)**: SPY, QQQ, IWM, EFA, EEM, DIA
- **Equity (sector)**: XLE, XLF, XLK, XLV, XLI, XLU, VNQ
- **Bonds**: TLT, IEF, HYG, LQD
- **Commodities**: GLD, SLV, USO

### 2. `src/data.py` — `DataLoader` class
- `fetch_all()`: Fetches OHLCV for all tickers with local CSV caching
- `compute_returns()`: Computes daily log returns with forward-fill for NaN handling
- `create_rolling_windows()`: Produces 3D array (N, lookback, n_assets) — no future data leakage
- `get_data_summary()`: Per-asset statistics (date range, missing values, return stats)
- `save_processed()`: Saves pickle with features, forward_returns, dates, tickers

### 3. `scripts/prepare_data.py`
Entry point that orchestrates the pipeline: fetch → summarize → create windows → validate → save.

## Data Statistics (from metrics.json)

| Metric | Value |
|---|---|
| Tickers fetched | 20 / 20 |
| Raw rows per ticker | 2,516 |
| Raw date range | 2016-03-28 to 2026-03-27 |
| Processed samples | 2,454 |
| Features shape | (2454, 60, 20) |
| Forward returns shape | (2454, 20) |
| Processed date range | 2016-06-22 to 2026-03-26 |
| NaN in features | False |
| NaN in forward returns | False |
| Missing close values | 0 (all tickers) |

## Feature Engineering Details

- **Log returns**: `log(close_t / close_{t-1})`, computed after forward-filling prices
- **Lookback window**: 60 trading days of past returns (features use t-60 to t-1)
- **Forward returns**: Return at t+1 (strict no-lookahead guarantee)
- **NaN handling**: Forward-fill prices across aligned date index, then skip any remaining NaN windows

## Data Leakage Prevention

1. Feature windows use strictly past data: `returns[i-lookback:i]`
2. Forward returns are at `i+1` (one step ahead of feature date)
3. No centered rolling windows used
4. Scalers not yet applied (Phase 3 will fit on train data only)
5. All 7 new DataLoader tests verify these properties

## Tests Added

7 new tests in `TestDataLoader` class:
- `test_compute_returns_shape` — correct DataFrame dimensions
- `test_compute_returns_no_future_leak` — manual verification of first return value
- `test_rolling_windows_shape` — 3D output with correct dimensions
- `test_rolling_windows_no_nan` — no NaN in output arrays
- `test_rolling_windows_use_only_past` — feature window and forward return alignment
- `test_data_summary_columns` — summary has expected columns
- `test_processed_pickle_output` — pickle contains required keys with valid data

All 19 tests pass (12 existing + 7 new).

## Observations

1. All 20 ETFs have identical row counts (2,516), indicating clean aligned data from the API
2. Zero missing close values across all tickers — no imputation needed
3. After applying 60-day lookback and removing the first/last rows, 2,454 usable samples remain
4. Annualized volatility ranges from 6.6% (IEF) to 39.0% (USO) — good diversity for a momentum model
5. The backward-compatible single-asset functions (`fetch_ohlcv`, `prepare_features`) remain unchanged for existing code in `train.py`

## Next Steps (Phase 3)

The 3D feature array is ready for multi-asset model training. Phase 3 will:
- Adapt the model to accept multi-asset input
- Implement walk-forward training on the full 20-asset universe
- Compare against baselines using the backtest framework
