# Technical Findings — Cycle 5

## Overview
Implemented walk-forward validation framework with monthly rebalancing as specified in the paper. Changed lookback from 60 days to 252 days (~12 months) to match paper specification. Rebalancing frequency changed from daily to monthly.

## Key Changes
1. **`src/evaluation.py`**: New `WalkForwardValidator` class with `WalkForwardConfig` supporting monthly rebalancing. Positions are held constant between month-end rebalance dates.
2. **`src/main.py`**: Added `run-walk-forward` command integrating the new validator.
3. **`config/assets.json`**: Lookback updated from 60 to 252 days.
4. **`docs/paper_spec.md`**: Documented paper parameters for reference.
5. **Code cleanup**: Moved legacy `src/train.py` and `src/cli.py` to `src/deprecated/`.

## Walk-Forward Validation Results

### Configuration
- 5 folds, expanding window (train ~3yr, test ~1yr)
- Lookback: 252 days (paper spec)
- Rebalance: monthly
- Costs: 10 bps total (5 fee + 5 slippage)
- Turnover penalty: γ=0.1

### Per-Fold Results (metrics.json source)

| Fold | Period | Net SR (Monthly) | Net SR (Daily) | Monthly Turnover |
|------|--------|-------------------|----------------|-----------------|
| 1 | 2021-03 → 2022-03 | 1.2996 | -1.3310 | 1.2624 |
| 2 | 2022-03 → 2023-03 | -1.7488 | -0.8222 | 0.7226 |
| 3 | 2023-03 → 2024-03 | 1.2289 | -3.0628 | 0.9847 |
| 4 | 2024-03 → 2025-03 | -0.8241 | -0.9130 | 0.9866 |
| 5 | 2025-03 → 2026-03 | 0.2454 | -1.0919 | 0.8285 |

### Aggregate (from metrics.json)
- **Avg Net Sharpe (monthly)**: 0.0402 ± 1.1823
- **Avg Gross Sharpe**: 0.3366
- **Monthly Turnover**: 0.957x annualized
- **Daily Turnover**: 6.8145x annualized
- **Turnover Reduction**: 85.96%
- **Positive Folds**: 3/5 (60%)
- **Total Trades**: 65

### Baseline Comparison (last fold OOS period, from metrics.json)
| Strategy | Net Sharpe |
|----------|-----------|
| 1/N Equal Weight | 1.3446 |
| Vol-Targeted 1/N | 1.2936 |
| Simple Momentum | 1.9480 |
| SMA Crossover | 1.7502 |
| **DNN Monthly** | **0.0402** |

### Performance Gap Analysis
The DNN strategy underperforms all baselines:
- **vs 1/N**: -1.3044 Sharpe difference
- **vs Momentum**: -1.9078 Sharpe difference
- **Primary cause**: Low-conviction positions from LSTM produce near-zero average returns
- **Secondary cause**: High fold variance (σ=1.18) indicates regime sensitivity

## Monthly Rebalancing Impact
The switch from daily to monthly rebalancing was the most impactful change:
- **Turnover**: Reduced from 6.81x to 0.96x annualized (86% reduction, now < 4x target)
- **Net vs Gross gap**: Narrowed significantly — monthly net Sharpe (0.04) much closer to gross (0.34) than daily net (-1.44) was to daily gross (0.50)
- **Cost drag**: Minimal at monthly frequency (~0.30 SR units) vs catastrophic at daily (~1.94 SR units)

## Observations
1. Monthly rebalancing successfully addresses the transaction cost problem identified in Cycle 4
2. The model's gross Sharpe (0.34) is still weak — the core signal is underpowered
3. Fold 2 (2022-03 to 2023-03, covering the rate hiking cycle) is the worst performer (-1.75 SR)
4. Folds 1 and 3 show promising performance (>1.0 SR each)
5. The 252-day lookback provides the model with a full year of context as per paper spec

## Defeat Analysis
The DNN strategy **loses to all baselines** on the primary metric (net Sharpe). Key factors:
- **Return**: DNN avg annual return 0.04% vs 1/N 18.32%
- **Sharpe**: DNN 0.04 vs 1/N 1.34, Momentum 1.95
- **Drawdown**: DNN -0.42% avg vs 1/N -10.54% (DNN has lower drawdown due to low conviction)
- **Turnover**: DNN 0.96x is now competitive with baselines (target < 4x achieved)
- **Cost**: At 10bps, costs are no longer the dominant factor — the weak signal is

The model produces positions that are too hedged/cautious, resulting in near-zero returns with low volatility. This is a signal quality issue, not a cost issue.
