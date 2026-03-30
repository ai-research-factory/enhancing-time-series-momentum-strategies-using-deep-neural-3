# Open Questions — Cycle 3

## Model Architecture
- **LSTM implemented**: Replaced MLP with 2-layer LSTM as per paper. The model now processes
  3D input (batch, timesteps, n_assets) correctly.
- **Lookback period**: Paper uses multiple horizons; current implementation uses a single
  60-day window. Multi-horizon features may capture different momentum regimes.
- **Shared vs per-asset LSTM**: Current implementation uses a single LSTM processing all
  assets together. The paper may use per-asset processing — needs further investigation.

## Turnover Problem (Ongoing)
- The LSTM model with λ=0.01 turnover penalty produces annualized turnover of 8.26x.
- The model outputs low-conviction positions (near zero) that change frequently.
- **Next steps**: Phase 4 will increase λ to 0.05–0.10 and add explicit position smoothing.

## DNN Underperformance
- DNN Sharpe (0.24) vs 1/N (1.18): the model underperforms on all metrics.
- Low conviction positions result in 0.10% annual return vs 13.88% for 1/N.
- This suggests the model is not effectively learning momentum signals from the data.
- **Possible causes**: (1) insufficient training epochs, (2) mini-batch Sharpe estimation
  is noisy, (3) need for multi-horizon features, (4) lookback period mismatch with paper.

## Data Constraints
- Using 20 ETFs vs paper's ~50+ commodity/equity futures — smaller and different universe.
- ETF universe covers major asset classes but lacks the commodity futures exposure in the paper.
- Data covers 2016–2026 (10 years), which may limit walk-forward evaluation quality.

## Evaluation Limitations
- Current evaluation uses single 80/20 split — walk-forward validation needed (Phase 5).
- The test period (2024-04-11 to 2026-03-26) covers a specific market regime and may not
  be representative.

## Training Stability
- Results vary across runs due to random initialization and the non-convex Sharpe objective.
- Consider setting random seeds or using ensemble of models for more stable results.
