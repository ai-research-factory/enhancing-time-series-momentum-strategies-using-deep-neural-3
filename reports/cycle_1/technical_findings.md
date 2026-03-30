# Technical Findings — Cycle 1

## Objective
Implement the core deep momentum network algorithm (MLP variant) with differentiable
Sharpe ratio loss and verify basic operation using real market data.

## Implementation Summary

### Model Architecture
- **Type**: MLP (simplified from paper's LSTM per design brief)
- **Input**: 60-day lookback window of daily log returns
- **Hidden layers**: 64 → ReLU → Dropout(0.1) → 32 → ReLU → Dropout(0.1)
- **Output**: Single neuron with tanh activation → position size in (-1, +1)

### Loss Function
- **Differentiable Sharpe ratio**: `loss = -mean(pnl) / std(pnl) + λ * mean(|Δposition|)`
- Turnover regularization coefficient λ = 0.01
- Directly optimizes risk-adjusted returns as described in the paper

### Training
- Optimizer: Adam (lr=1e-3, weight_decay=1e-5)
- Gradient clipping: max_norm=1.0
- Early stopping: patience=15 epochs
- Features standardized per window (scaler fitted on train only)

### Data
- **Asset**: Nikkei 225 (^N225) daily OHLCV from ARF Data API
- **Period**: 2016-03-30 to 2026-03-28 (2442 trading days)
- **Features**: 2380 samples after lookback window construction

### Walk-Forward Validation
- 4 out-of-sample windows (min train size: 302 days)
- Gap of 1 day between train and test to prevent leakage

## Results (from metrics.json)

| Metric | Value |
|---|---|
| Gross Sharpe (avg) | -0.2305 |
| Net Sharpe (avg) | -2.056 |
| Annual Return | -3.58% |
| Max Drawdown | -9.12% |
| Hit Rate | 36.29% |
| Total Trades | 1565 |
| Walk-Forward Windows | 4 |
| Positive Windows | 0 |

### Baseline Comparison

| Strategy | Sharpe | Notes |
|---|---|---|
| **Deep Momentum MLP (ours)** | -2.056 (net) | Underperforms all baselines |
| 1/N Buy & Hold | 0.5797 | Simple benchmark |
| Vol-Targeted 1/N | 0.5592 | Risk-adjusted benchmark |
| Simple Momentum (252d) | 0.9871 | Best performing baseline |

**Strategy loses to all baselines** on Sharpe ratio (net of costs), annual return, and
hit rate. The strategy has lower max drawdown (-9.12% vs -31.87% for buy-and-hold) but
this is due to low exposure magnitude rather than skill.

### Defeat Analysis
- **Sharpe**: Strategy net Sharpe (-2.056) is 2.46 below 1/N baseline (0.58)
- **Return**: Strategy underperforms 1/N by 14.35% annually
- **Turnover**: 1565 trades across 1700 test days — near-daily position changes
- **Cost sensitivity**: Transaction costs (15 bps total) are the dominant factor;
  gross Sharpe (-0.23) is poor but the gap to net Sharpe (-2.06) shows costs destroy returns

## Key Observations

1. **Core algorithm works**: The differentiable Sharpe ratio loss successfully trains
   the MLP, positions are generated in (-1, +1), and the walk-forward framework operates
   correctly.

2. **Excessive turnover**: The MLP changes positions almost every day because the
   continuous tanh output varies with each new input. This generates high transaction costs.

3. **Weak signal**: Even before costs (gross Sharpe ≈ -0.23), the model fails to capture
   momentum effectively. The 60-day return lookback may not provide strong enough signal
   for a single-asset model.

4. **Training instability**: Results vary across runs due to random initialization and
   the non-convex Sharpe ratio objective.

## Recommendations for Cycle 2

1. **Reduce turnover**: Add position smoothing or increase turnover penalty significantly
2. **Multi-horizon features**: Use returns at multiple lookback periods (5, 21, 63, 126, 252 days)
3. **Data preprocessing**: Add volatility normalization and trend features
4. **Larger training windows**: Current minimum of 302 days may be insufficient
5. **Batch training**: Use mini-batches with sequential sampling to improve gradient estimates
