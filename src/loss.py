"""
Loss functions for Deep Momentum Network training.

Implements differentiable Sharpe ratio loss with turnover regularization
as described in the paper: loss = -Sharpe + gamma * turnover.
"""
import torch


def sharpe_loss(
    positions: torch.Tensor,
    returns: torch.Tensor,
    turnover_penalty: float = 0.0,
    prev_positions: torch.Tensor = None,
) -> torch.Tensor:
    """
    Differentiable Sharpe ratio loss with turnover regularization.

    Loss = -Sharpe + gamma * mean(|delta_position|)

    The turnover term penalizes position changes within the batch. If
    prev_positions is provided, the first position change is computed
    relative to it (enabling cross-batch continuity).

    Args:
        positions: (T,) or (T, n_assets) predicted position sizes
        returns: (T,) or (T, n_assets) next-day returns
        turnover_penalty: gamma coefficient for turnover regularization
        prev_positions: optional (n_assets,) or scalar — last positions
            from previous batch for continuity

    Returns:
        Scalar loss = -Sharpe + gamma * mean(|delta_position|)
    """
    pnl = positions * returns

    # For multi-asset: compute portfolio-level PnL (equal-weighted)
    if pnl.dim() == 2:
        pnl = pnl.mean(dim=1)  # (T,) average across assets

    mean_pnl = pnl.mean()
    std_pnl = pnl.std()

    sharpe = mean_pnl / (std_pnl + 1e-8)
    loss = -sharpe

    if turnover_penalty > 0.0:
        # Compute position changes within the batch
        delta = positions[1:] - positions[:-1]

        # If we have previous positions, include the transition from prev to first
        if prev_positions is not None:
            first_delta = (positions[0:1] - prev_positions).unsqueeze(0) if prev_positions.dim() == 1 and positions.dim() == 2 else (positions[0:1] - prev_positions)
            if first_delta.dim() < delta.dim():
                first_delta = first_delta.unsqueeze(0)
            delta = torch.cat([first_delta, delta], dim=0)

        turnover = delta.abs().mean()
        loss = loss + turnover_penalty * turnover

    return loss
