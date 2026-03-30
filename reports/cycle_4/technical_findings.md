# Technical Findings — Cycle 4

## Phase 4: Turnover Regularization and Cost Model

### Implementation Summary

1. **`src/loss.py`**: Created dedicated loss module with enhanced `sharpe_loss` function.
   - Loss = `-Sharpe + gamma * mean(|Δposition|)`
   - Supports optional `prev_positions` for cross-batch continuity
   - Backward-compatible re-export in `src/model.py`

2. **`src/backtest.py`**: Updated `Backtester.evaluate_positions()`:
   - Per-asset cost calculation: `|Δposition| * cost_bps / 10000`
   - Separate gross and net Sharpe computation
   - Fixed `totalTrades` bug: now counts days with meaningful position changes (>1%)
   - Returns `grossSharpe`, `grossReturn` alongside net metrics

3. **`src/main.py`**: Added `run-cost-analysis` command with walk-forward validation:
   - Trains two models: gamma=0.0 (no regularization) and gamma=0.1 (regularized)
   - 5-fold walk-forward out-of-sample evaluation
   - Saves `cost_analysis.json` and `metrics.json`

### Reviewer Feedback Addressed

- **Walk-forward validation**: Implemented with 5 expanding-window folds (n_walk_forward_windows=5)
- **totalTrades bug**: Fixed — now correctly counts 2,195 trades across all folds
- **Fold Sharpe ratios**: Recorded in metrics.json as `fold_sharpe_ratios` list
- **Net vs Gross Sharpe separation**: `transactionCosts.netSharpe` now correctly differs from `sharpeRatio` (gross)

### Results (from metrics.json)

#### Regularization Comparison

| Metric | No Regularization (γ=0) | With Regularization (γ=0.1) |
|---|---|---|
| Avg Gross Sharpe | 0.5962 | 0.6956 |
| Avg Net Sharpe | -0.2712 | -0.2193 |
| Std Net Sharpe | 0.9361 | 0.8268 |
| Avg Turnover (annualized) | 8.49x | 8.01x |
| Positive Folds | 2/5 | 2/5 |
| Total Trades | 2,195 | 2,195 |

**Turnover reduction**: 5.66% lower with regularization.
**Net Sharpe improvement**: -0.27 → -0.22 (0.05 improvement).
**Stability improvement**: std of fold Sharpe dropped from 0.94 to 0.83.

#### Fold-Level Results (Regularized Model)

| Fold | Period | Gross Sharpe | Net Sharpe | Turnover |
|---|---|---|---|---|
| 1 | 2017-06 to 2019-03 | 1.92 | 0.98 | 6.34 |
| 2 | 2019-03 to 2020-12 | 0.49 | -0.29 | 9.12 |
| 3 | 2020-12 to 2022-09 | -0.52 | -1.24 | 8.62 |
| 4 | 2022-09 to 2024-06 | 1.43 | 0.40 | 7.92 |
| 5 | 2024-06 to 2026-03 | 0.16 | -0.95 | 8.07 |

#### Baseline Comparison (from metrics.json)

| Strategy | Sharpe |
|---|---|
| DNN (regularized, net) | -0.2193 |
| 1/N Equal Weight | 1.1777 |
| Vol-Targeted 1/N | 1.2580 |
| Simple Momentum | 1.7772 |
| SMA Crossover | 1.1512 |

### Key Observations

1. **Turnover regularization works**: γ=0.1 reduces turnover by 5.66% and improves net Sharpe by 0.05. The effect is modest because the model still produces high-turnover positions (8x annualized).

2. **Cost impact is significant**: Gross Sharpe of 0.70 drops to net Sharpe of -0.22 after 5bps costs. Transaction costs are the dominant factor destroying DNN returns.

3. **DNN underperforms all baselines**: The strategy loses on return, Sharpe, and drawdown vs 1/N equal weight (Sharpe diff: -1.40). The model produces low-conviction, high-frequency position changes that are costly to execute.

4. **Regime dependency**: Folds 1 and 4 show positive net Sharpe (0.98 and 0.40), while folds 2, 3, 5 are negative. The model may work in some market regimes but not others.

5. **Walk-forward validation reveals instability**: The high standard deviation of fold Sharpe (0.83) indicates the strategy is not robust across different time periods.

### Defeat Analysis (vs baselines)

The DNN strategy loses on all metrics vs 1/N:
- **Sharpe**: -0.22 vs 1.18 (lost)
- **Return**: -0.15% vs 14.15% (lost)
- **Drawdown**: -1.59% vs -12.33% (won — lower drawdown, but due to near-zero positions)
- **Turnover**: 8.01x annualized vs ~0 (lost — excessive trading)
- **Cost sensitivity**: High — 5bps costs turn a marginally positive gross Sharpe into negative net Sharpe

### Next Steps (Phase 5+)
- Explore higher γ values (0.5, 1.0) to further reduce turnover
- Consider position smoothing (EMA) to reduce high-frequency changes
- Multi-horizon features may improve signal quality
- Larger batch sizes for more stable Sharpe estimation during training
