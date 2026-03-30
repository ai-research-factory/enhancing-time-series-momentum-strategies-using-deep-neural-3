# Preflight Check — Cycle 3

## 1. Data Boundary Table

| Item | Value |
|---|---|
| Data acquisition end date | 2026-03-27 (before today 2026-03-30) |
| Train period | 2016-06-22 ~ 2024-05-20 (first 80% of 2454 samples) |
| Validation period | N/A (single train/test split for Phase 3) |
| Test period | 2024-05-20 ~ 2026-03-26 (last 20% of 2454 samples) |
| No overlap confirmed | Yes |
| No future dates confirmed | Yes |

## 2. Feature Timestamp Contract

- All features use only data at t-1 or earlier for prediction at time t? → **Yes**
  - `create_rolling_windows()` uses `returns.iloc[i-lookback:i]` (past data only)
  - Forward returns use `returns.iloc[i+1]` (strictly future)
- Scaler/Imputer fitted on train data only? → **Yes**
  - `StandardScaler.fit_transform(train_X)` then `.transform(test_X)`
- No centered rolling windows used? → **Yes** (default `center=False`)

## 3. Paper Spec Difference Table

| Parameter | Paper value | Current implementation | Match? |
|---|---|---|---|
| Universe | ~50 commodity/equity futures | 20 liquid ETFs | No (closest proxy with available data) |
| Lookback period | 20-day returns + features | 60-day rolling window of log returns | No (wider window, will evaluate) |
| Rebalance frequency | Daily | Daily | Yes |
| Features | Past returns (multi-horizon) | Past daily log returns (single horizon) | Partial |
| Cost model | Proportional transaction costs | 10 bps fee + 5 bps slippage | Yes (similar approach) |
| Model architecture | LSTM | MLP (being replaced with LSTM this cycle) | Yes (after this cycle) |
| Loss function | Negative Sharpe + turnover penalty | Negative Sharpe + turnover penalty | Yes |
| Position sizing | Continuous (-1, +1) via tanh | Continuous (-1, +1) via tanh | Yes |

## Notes

- This cycle implements a single 80/20 train/test split (not full walk-forward yet — Phase 5).
- The LSTM model replaces the MLP per review feedback to match the paper's architecture.
- Data source: ARF Data API, cached locally in `data/` directory.
