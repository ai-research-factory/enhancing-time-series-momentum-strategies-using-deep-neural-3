"""
Deep Momentum Network — LSTM variant.

Implements the paper's LSTM-based model that takes a sequence of past returns
for multiple assets and outputs position sizes in (-1, +1) via tanh.
Also retains a backward-compatible MomentumMLP for existing tests.
"""
import torch
import torch.nn as nn


class DeepMomentumLSTM(nn.Module):
    """LSTM-based deep momentum network for multi-asset position sizing.

    Takes a 3D input (batch, timesteps, n_assets) of past returns and outputs
    position sizes in (-1, +1) for each asset.
    """

    def __init__(self, n_assets: int = 20, lookback: int = 60,
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_assets = n_assets
        self.lstm = nn.LSTM(
            input_size=n_assets,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_assets),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, timesteps, n_assets) tensor of past returns
        Returns:
            (batch, n_assets) tensor of position sizes in (-1, +1)
        """
        lstm_out, _ = self.lstm(x)  # (batch, timesteps, hidden_size)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        return self.fc(last_hidden)  # (batch, n_assets)


class MomentumMLP(nn.Module):
    """MLP that maps a lookback window of returns to a position size.
    Retained for backward compatibility with existing tests.
    """

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

    Supports both single-asset (1D) and multi-asset (2D) positions/returns.

    Args:
        positions: (T,) or (T, n_assets) predicted position sizes
        returns: (T,) or (T, n_assets) next-day returns
        turnover_penalty: coefficient for turnover regularization

    Returns:
        Scalar loss = -Sharpe + turnover_penalty * mean(|delta_position|)
    """
    pnl = positions * returns

    # For multi-asset: compute portfolio-level PnL (equal-weighted)
    if pnl.dim() == 2:
        pnl = pnl.mean(dim=1)  # (T,) average across assets

    mean_pnl = pnl.mean()
    std_pnl = pnl.std()

    # Avoid division by zero
    sharpe = mean_pnl / (std_pnl + 1e-8)

    loss = -sharpe

    if turnover_penalty > 0.0:
        if positions.dim() == 2:
            turnover = (positions[1:] - positions[:-1]).abs().mean()
        else:
            turnover = (positions[1:] - positions[:-1]).abs().mean()
        loss = loss + turnover_penalty * turnover

    return loss
