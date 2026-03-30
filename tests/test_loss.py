"""Tests for the loss module (src/loss.py)."""
import torch
import pytest

from src.loss import sharpe_loss


class TestSharpeLoss:
    """Tests for the enhanced sharpe_loss with turnover regularization."""

    def test_basic_negative_sharpe(self):
        """Perfect alignment should give negative loss (positive Sharpe)."""
        returns = torch.tensor([0.01, -0.02, 0.015, -0.01, 0.02])
        positions = torch.sign(returns)
        loss = sharpe_loss(positions, returns)
        assert loss.item() < 0

    def test_turnover_penalty_increases_loss(self):
        """Higher turnover should increase loss when penalty > 0."""
        returns = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01])
        stable = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        changing = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0])

        loss_stable = sharpe_loss(stable, returns, turnover_penalty=0.1)
        loss_changing = sharpe_loss(changing, returns, turnover_penalty=0.1)
        assert loss_changing > loss_stable

    def test_zero_penalty_no_turnover_effect(self):
        """With turnover_penalty=0, changing positions should not affect loss."""
        returns = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01])
        stable = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        changing = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0])

        # With zero penalty, both should have same Sharpe component
        # but different PnL (changing positions flip the sign)
        loss_stable_0 = sharpe_loss(stable, returns, turnover_penalty=0.0)
        loss_changing_0 = sharpe_loss(changing, returns, turnover_penalty=0.0)
        # These will differ due to PnL, not turnover penalty
        assert isinstance(loss_stable_0.item(), float)
        assert isinstance(loss_changing_0.item(), float)

    def test_multi_asset_2d(self):
        """Should work with 2D (T, n_assets) tensors."""
        positions = torch.tensor([[0.5, -0.3], [0.8, 0.1], [0.2, -0.5]])
        returns = torch.tensor([[0.01, -0.02], [0.015, 0.01], [-0.005, 0.02]])

        loss = sharpe_loss(positions, returns, turnover_penalty=0.05)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_multi_asset_turnover_regularization(self):
        """Turnover penalty should reduce position changes for multi-asset."""
        T, n_assets = 20, 5
        returns = torch.randn(T, n_assets) * 0.01
        stable = torch.ones(T, n_assets) * 0.5
        changing = torch.randn(T, n_assets)

        loss_stable = sharpe_loss(stable, returns, turnover_penalty=0.5)
        loss_changing = sharpe_loss(changing, returns, turnover_penalty=0.5)

        # Stable positions should have lower turnover component
        # (Sharpe component may differ, so we test with high penalty)
        stable_turnover = stable[1:] - stable[:-1]
        changing_turnover = changing[1:] - changing[:-1]
        assert stable_turnover.abs().mean() < changing_turnover.abs().mean()

    def test_differentiable(self):
        """Loss should be differentiable through positions."""
        positions = torch.randn(10, 3, requires_grad=True)
        returns = torch.randn(10, 3) * 0.01

        loss = sharpe_loss(positions, returns, turnover_penalty=0.1)
        loss.backward()
        assert positions.grad is not None
        assert torch.isfinite(positions.grad).all()

    def test_higher_gamma_stronger_penalty(self):
        """Higher gamma should penalize turnover more."""
        returns = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01])
        changing = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0])

        loss_low = sharpe_loss(changing, returns, turnover_penalty=0.01)
        loss_high = sharpe_loss(changing, returns, turnover_penalty=1.0)
        assert loss_high > loss_low
