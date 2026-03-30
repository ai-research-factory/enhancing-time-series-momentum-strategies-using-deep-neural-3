"""
Tests for LSTM model I/O and training step verification.

Verifies that the DeepMomentumLSTM model:
1. Accepts 3D input tensors (batch, timesteps, n_assets)
2. Produces correctly shaped output
3. Runs a full training step (forward, backward, optimizer step) without error
"""
import numpy as np
import torch
import pytest

from src.model import DeepMomentumLSTM, sharpe_loss


class TestDeepMomentumLSTM:
    """Tests for the LSTM-based deep momentum network."""

    def test_output_shape_default(self):
        """Model output shape should be (batch, n_assets)."""
        model = DeepMomentumLSTM(n_assets=20, lookback=60)
        x = torch.randn(32, 60, 20)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (32, 20)

    def test_output_shape_various_batches(self):
        """Test with different batch sizes."""
        model = DeepMomentumLSTM(n_assets=5, lookback=30, hidden_size=16, num_layers=1)
        for batch in [1, 10, 64]:
            x = torch.randn(batch, 30, 5)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (batch, 5), f"Failed for batch={batch}"

    def test_output_range(self):
        """Output should be in (-1, +1) due to tanh activation."""
        model = DeepMomentumLSTM(n_assets=10, lookback=20, hidden_size=32)
        x = torch.randn(100, 20, 10)
        with torch.no_grad():
            out = model(x)
        assert (out >= -1).all() and (out <= 1).all()

    def test_training_step_no_error(self):
        """Forward, backward, and optimizer step should run without error."""
        n_assets = 5
        model = DeepMomentumLSTM(n_assets=n_assets, lookback=10, hidden_size=16, num_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(32, 10, n_assets)
        returns = torch.randn(32, n_assets) * 0.01

        # Forward
        positions = model(x)
        assert positions.shape == (32, n_assets)

        # Loss
        loss = sharpe_loss(positions, returns, turnover_penalty=0.01)
        assert loss.dim() == 0  # scalar

        # Backward
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

        # Optimizer step
        optimizer.step()

    def test_sharpe_loss_multi_asset(self):
        """Sharpe loss should work with 2D positions and returns."""
        positions = torch.tensor([[0.5, -0.3], [0.8, 0.1], [0.2, -0.5]])
        returns = torch.tensor([[0.01, -0.02], [0.015, 0.01], [-0.005, 0.02]])

        loss = sharpe_loss(positions, returns, turnover_penalty=0.01)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_model_deterministic_eval_mode(self):
        """Same input in eval mode should give same output (no dropout)."""
        torch.manual_seed(42)
        model = DeepMomentumLSTM(n_assets=3, lookback=10, hidden_size=8, num_layers=1)
        model.eval()
        x = torch.randn(5, 10, 3)

        with torch.no_grad():
            out1 = model(x).clone()
            out2 = model(x).clone()

        torch.testing.assert_close(out1, out2)

    def test_different_lookback_lengths(self):
        """Model should handle different lookback window sizes."""
        for lookback in [10, 30, 60, 120]:
            model = DeepMomentumLSTM(n_assets=5, lookback=lookback, hidden_size=16, num_layers=1)
            x = torch.randn(8, lookback, 5)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (8, 5)
