"""
Deep Momentum Network — MLP variant.

Simplified from LSTM to MLP per design brief. Takes a lookback window of past
returns and outputs a position size in (-1, +1) via tanh.
"""
import torch
import torch.nn as nn


class MomentumMLP(nn.Module):
    """MLP that maps a lookback window of returns to a position size."""

    def __init__(self, lookback: int = 60, hidden_sizes: tuple = (64, 32)):
        super().__init__()
        layers = []
        in_dim = lookback
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, lookback) tensor of past returns
        Returns:
            (batch,) tensor of position sizes in (-1, +1)
        """
        return self.net(x).squeeze(-1)


def sharpe_loss(positions: torch.Tensor, returns: torch.Tensor,
                turnover_penalty: float = 0.0) -> torch.Tensor:
    """
    Differentiable Sharpe ratio loss (negated for minimization).

    Args:
        positions: (T,) predicted position sizes
        returns: (T,) next-day returns
        turnover_penalty: coefficient for turnover regularization

    Returns:
        Scalar loss = -Sharpe + turnover_penalty * mean(|delta_position|)
    """
    pnl = positions * returns

    mean_pnl = pnl.mean()
    std_pnl = pnl.std()

    # Avoid division by zero
    sharpe = mean_pnl / (std_pnl + 1e-8)

    loss = -sharpe

    if turnover_penalty > 0.0:
        turnover = positions[1:] - positions[:-1]
        loss = loss + turnover_penalty * turnover.abs().mean()

    return loss
