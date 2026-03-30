# Open Questions — Cycle 1

## Model Architecture
- **MLP vs LSTM**: The design brief specifies MLP for v1, but the paper uses LSTM.
  How much performance gap is due to this simplification vs other factors?
- **Lookback period**: Paper uses multiple horizons; current implementation uses a single
  60-day window. Multi-horizon features may capture different momentum regimes.

## Turnover Problem
- The MLP with continuous tanh output changes position daily, generating ~1565 trades
  across 1700 test days. The paper addresses this with turnover regularization, but
  the current penalty (λ=0.01) is insufficient.
- **Question**: Should we add position smoothing (e.g., exponential moving average of
  positions) or rely solely on increased turnover penalty?

## Single-Asset Limitation
- The paper operates on 88 futures across asset classes, providing diversification.
  A single-asset model has higher variance and no cross-asset signal.
- **Question**: Should we expand to multiple assets in Cycle 2, or focus on improving
  the single-asset model first?

## Data
- Nikkei 225 data from ARF API covers 2016-03-30 to 2026-03-28 (~10 years, 2442 days).
  The paper uses longer history. This may limit the number and quality of walk-forward windows.

## Training Stability
- Results vary across runs due to random initialization and the non-convex Sharpe objective.
  Consider setting random seeds or using ensemble of models for more stable results.
