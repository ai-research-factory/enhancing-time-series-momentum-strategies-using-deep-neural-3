# Open Questions — Cycle 5

## Model Architecture
- **Shared vs per-asset LSTM**: Current implementation uses a single LSTM processing all
  assets together. The paper may use per-asset processing — needs further investigation.
- **Multi-horizon features**: Paper may use multiple lookback windows. Current implementation
  uses a single 252-day window (updated from 60 in Cycle 5).

## Signal Quality (New Cycle 5)
- Monthly rebalancing solved the turnover problem (0.96x, < 4x target).
- The core issue is now **weak gross signal**: avg gross Sharpe 0.34 across 5 folds.
- The model produces low-conviction positions near zero, resulting in near-zero returns.
- Possible directions for improvement:
  - Stronger turnover penalty (γ=0.5-1.0) to force the model into more committed positions
  - Per-asset LSTM or attention mechanism for better cross-asset learning
  - Additional features beyond raw returns (e.g., volatility, momentum indicators)
  - Larger training epochs or learning rate scheduling

## Regime Sensitivity (Updated Cycle 5)
- Fold performance varies widely: [1.30, -1.75, 1.23, -0.82, 0.25] net Sharpe
- Fold 2 (2022-03 to 2023-03, rate hiking cycle) is consistently worst
- Folds 1 and 3 show strong positive Sharpe (>1.0), suggesting the model works in some regimes
- Regime detection or conditional model switching could help

## Monthly Rebalancing (Resolved)
- Monthly rebalancing reduced turnover from 6.81x to 0.96x (86% reduction)
- Annualized turnover now below the 4x target specified by reviewer
- Cost drag is minimal at monthly frequency (~0.30 SR units vs ~1.94 at daily)

## Data Constraints
- Using 20 ETFs vs paper's ~88 commodity/equity futures — smaller and different universe.
- ETF universe covers major asset classes but lacks the commodity futures exposure in the paper.
- Data covers 2016–2026 (10 years). With 252-day lookback, effective data starts from 2017-03.

## Cost Model (Updated)
- Using 10bps total (5bps fee + 5bps slippage) as per paper.
- With monthly rebalancing, costs are no longer the dominant performance drag.
- The gross-to-net Sharpe gap is now small (~0.30), confirming the cost model is reasonable.

## Paper Spec Alignment (Updated Cycle 5)
- Lookback: 252 days (aligned with paper's 12-month specification)
- Rebalance frequency: monthly (aligned with paper specification)
- Remaining gaps: universe size (20 vs 88), no per-asset LSTM variant tested
- See `docs/paper_spec.md` for full parameter comparison
