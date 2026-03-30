# Technical Findings — Cycle 3: Basic Learning & Backtest Loop

## Summary

Implemented a complete training and backtesting pipeline for the Deep Momentum Network using an LSTM architecture on 20 ETFs. The model was trained on an 80/20 time-series split and evaluated against four baseline strategies.

## Review Feedback Addressed

1. **MLP → LSTM replacement**: Replaced `MomentumMLP` with `DeepMomentumLSTM` in `src/model.py`. The new model accepts 3D input `(batch, timesteps, n_assets)` and outputs `(batch, n_assets)` positions via tanh.
2. **Training loop for 3D data**: Created `src/training.py` with `train_single_split()` that handles 3D multi-asset data, mini-batch training, per-asset StandardScaler fitted on train data only.
3. **Model I/O tests**: Created `tests/test_model_io.py` with 7 tests verifying 3D input/output shapes, output range, training step, and multi-asset Sharpe loss.

## Implementation Details

### Model Architecture
- **DeepMomentumLSTM**: 2-layer LSTM (hidden_size=64) → FC(64→32→ReLU→Dropout→20→Tanh)
- Input: (batch, 60, 20) — 60-day lookback, 20 assets
- Output: (batch, 20) — position sizes in (-1, +1) per asset
- Loss: Negative Sharpe ratio with turnover regularization (λ=0.01)

### Training Pipeline (`src/training.py`)
- 80/20 time-series split: 1963 train / 491 test samples
- Train period: 2016-06-22 to 2024-04-10
- Test period: 2024-04-11 to 2026-03-26
- Mini-batch training (batch_size=256) with Adam optimizer
- Early stopping (patience=20 epochs)
- StandardScaler fitted on train data only (per-asset)

### Backtester (`src/backtest.py`)
- `Backtester` class evaluates multi-asset positions with transaction costs
- SMA crossover baseline (20-day/60-day) implemented
- All four required baselines computed: 1/N, Vol-targeted 1/N, Simple Momentum, SMA Crossover

## Results (from metrics.json)

| Strategy | Sharpe Ratio | Annual Return | Max Drawdown |
|---|---|---|---|
| **DNN (LSTM)** | 0.2444 | 0.10% | -0.62% |
| 1/N Equal Weight | 1.1819 | 13.88% | -12.33% |
| Vol-Targeted 1/N | 1.2378 | — | — |
| Simple Momentum | 2.0670 | — | — |
| SMA Crossover (20/60) | 0.9432 | 7.15% | -7.75% |

### DNN Model Characteristics
- Hit rate: 51.12%
- Annualized turnover: 8.26x
- The model produces very small positions (low conviction), resulting in minimal returns and drawdown

## Analysis of DNN Underperformance

The DNN model underperforms all baselines on Sharpe ratio. Key observations:

1. **Low conviction positions**: The model outputs very small position sizes, leading to near-zero returns. This is reflected in the tiny annual return (0.10%) and tiny max drawdown (-0.62%).
2. **High turnover**: Annualized turnover of 8.26x despite small positions suggests the model is learning noise rather than sustained trends.
3. **Insufficient turnover penalty**: λ=0.01 is too low to effectively constrain position changes.
4. **Single split limitation**: An 80/20 split gives only one evaluation window. Walk-forward validation (Phase 5) will provide more robust evaluation.

## Defeated Metrics

The DNN strategy loses to 1/N on:
- **Sharpe ratio**: -0.9375 difference (0.24 vs 1.18)
- **Return**: -13.78% difference
- **Turnover**: 8.26x vs ~0 for buy-and-hold

## Recommendations for Next Cycles

1. **Phase 4**: Increase turnover penalty (λ=0.05–0.10) to reduce excessive trading
2. **Phase 5**: Walk-forward validation will provide more robust out-of-sample evaluation
3. Consider position smoothing (exponential moving average of positions)
4. Multi-horizon features may help the LSTM capture different momentum regimes
