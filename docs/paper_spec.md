# Paper Specification: Deep Momentum Networks

## Reference
"Enhancing Time-Series Momentum Strategies Using Deep Neural Networks"

## Core Parameters

| Parameter | Paper Value | Notes |
|---|---|---|
| Model | LSTM (2-layer, hidden=64) | Processes all assets jointly |
| Lookback | 12 months (~252 trading days) | Past return series |
| Rebalance frequency | Monthly | Position updated once per month |
| Universe | ~88 commodity/equity futures | We use 20 ETFs due to data constraints |
| Position sizing | Continuous [-1, +1] via tanh | Per-asset position output |
| Loss function | -Sharpe + gamma * turnover | End-to-end Sharpe optimization |
| Turnover penalty (gamma) | ~0.1 | Controls transaction cost sensitivity |
| Transaction costs | 5-10 bps per trade | Applied to position changes |
| Validation | Walk-forward (5-10 folds) | Expanding/rolling window |
| Train window | ~3 years | Expanding window preferred |
| Test window | ~1 year | Out-of-sample evaluation |

## Feature Construction
- Input: sequence of past daily log returns over lookback period
- Shape: (batch, lookback_days, n_assets)
- No additional features beyond raw returns

## Training
- Optimizer: Adam
- Mini-batch training with shuffled time windows
- Early stopping on training loss
- Gradient clipping for stability

## Evaluation
- Primary metric: Out-of-sample Sharpe ratio (net of costs)
- Walk-forward aggregation across all OOS periods
- Baselines: 1/N equal weight, vol-targeted 1/N, simple momentum
