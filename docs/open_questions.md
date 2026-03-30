# Open Questions — Cycle 4

## Model Architecture
- **Shared vs per-asset LSTM**: Current implementation uses a single LSTM processing all
  assets together. The paper may use per-asset processing — needs further investigation.
- **Multi-horizon features**: Paper may use multiple lookback windows. Current implementation
  uses a single 60-day window.

## Turnover Problem (Updated Cycle 4)
- With γ=0.1, turnover reduced from 8.49x to 8.01x (5.66% reduction).
- This is a modest improvement. Higher γ values (0.5–1.0) should be explored in future cycles.
- The model still produces frequent small position changes. Position smoothing (EMA) or
  discretizing positions may help further.
- **Key insight**: The gap between gross Sharpe (0.70) and net Sharpe (-0.22) is large,
  indicating transaction costs are the dominant drag on performance.

## DNN Underperformance (Updated Cycle 4)
- DNN net Sharpe (-0.22) vs 1/N (1.18): the model still underperforms all baselines.
- Walk-forward validation confirms the result is not an artifact of a single split.
- Fold-level analysis shows the model works in some periods (folds 1, 4) but not others,
  suggesting regime sensitivity.
- **Possible causes**: (1) γ=0.1 still too low, (2) LSTM struggling with noisy daily returns,
  (3) need for multi-horizon features, (4) ETF universe may not have strong momentum signals.

## Data Constraints
- Using 20 ETFs vs paper's ~88 commodity/equity futures — smaller and different universe.
- ETF universe covers major asset classes but lacks the commodity futures exposure in the paper.
- Data covers 2016–2026 (10 years), which limits the amount of training data for earlier folds.

## Walk-Forward Validation
- 5 folds with expanding windows implemented in Cycle 4.
- First fold has only 253 training samples (~1 year), which may be insufficient for the LSTM.
- Later folds have more training data and tend to perform better on gross Sharpe.

## Cost Model
- Using 5bps per trade as per paper. No slippage modeled separately.
- Real-world costs for ETFs may be lower (1-3 bps), which would improve net performance.
- The cost model applies costs proportional to position change magnitude, which penalizes
  continuous-valued position changes more than discrete (0/1) strategies.
