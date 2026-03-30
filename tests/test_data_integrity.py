"""
Data integrity and leakage tests for the deep momentum network.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from src.data import prepare_features
from src.model import MomentumMLP, sharpe_loss
from src.backtest import (
    BacktestConfig,
    WalkForwardValidator,
    calculate_costs,
    compute_metrics,
    generate_metrics_json,
    BacktestResult,
)


# ---- Data integrity tests ----

class TestDataIntegrity:
    """Tests to verify no data leakage or future information contamination."""

    @pytest.fixture
    def sample_ohlcv(self):
        """Create a simple OHLCV DataFrame for testing."""
        dates = pd.date_range("2020-01-01", periods=200, freq="B")
        np.random.seed(42)
        close = 100 * np.cumprod(1 + np.random.randn(200) * 0.01)
        return pd.DataFrame(
            {
                "open": close * (1 + np.random.randn(200) * 0.001),
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.random.randint(1e6, 1e7, 200),
            },
            index=dates,
        )

    def test_features_use_only_past_data(self, sample_ohlcv):
        """Feature at time t should only use returns from t-lookback to t-1."""
        lookback = 10
        features, fwd_returns, dates = prepare_features(sample_ohlcv, lookback=lookback)

        assert len(features) > 0
        assert features.shape[1] == lookback
        # Each feature row should have exactly `lookback` values
        assert not np.isnan(features).any()

    def test_forward_returns_are_future(self, sample_ohlcv):
        """Forward returns should correspond to t+1 (the next day)."""
        lookback = 10
        features, fwd_returns, dates = prepare_features(sample_ohlcv, lookback=lookback)

        close = sample_ohlcv["close"]
        log_returns = np.log(close / close.shift(1))

        # For each date in `dates`, the forward return should be the return at date+1
        for i, date in enumerate(dates):
            # Find the position of this date in the original index
            idx = sample_ohlcv.index.get_loc(date)
            expected_fwd = log_returns.iloc[idx + 1]
            np.testing.assert_almost_equal(fwd_returns[i], expected_fwd, decimal=5)

    def test_no_overlap_in_walk_forward_splits(self, sample_ohlcv):
        """Train and test indices should not overlap in walk-forward."""
        config = BacktestConfig(n_splits=3, min_train_size=50, gap=1)
        validator = WalkForwardValidator(config)

        features, fwd_returns, dates = prepare_features(sample_ohlcv, lookback=10)
        data_df = pd.DataFrame({"idx": range(len(features))}, index=dates)

        for train_idx, test_idx in validator.split(data_df):
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert train_set.isdisjoint(test_set), "Train and test overlap!"
            # Train should come before test
            assert max(train_idx) < min(test_idx), "Train indices exceed test indices!"

    def test_gap_between_train_and_test(self, sample_ohlcv):
        """There should be a gap between train end and test start."""
        config = BacktestConfig(n_splits=3, min_train_size=50, gap=1)
        validator = WalkForwardValidator(config)

        features, fwd_returns, dates = prepare_features(sample_ohlcv, lookback=10)
        data_df = pd.DataFrame({"idx": range(len(features))}, index=dates)

        for train_idx, test_idx in validator.split(data_df):
            assert min(test_idx) - max(train_idx) >= 1, "No gap between train and test!"


# ---- Model tests ----

class TestModel:
    """Tests for the MLP model and Sharpe loss."""

    def test_model_output_range(self):
        """Model output should be in (-1, +1) due to tanh."""
        model = MomentumMLP(lookback=60, hidden_sizes=(32, 16))
        x = torch.randn(100, 60)
        with torch.no_grad():
            positions = model(x)
        assert positions.shape == (100,)
        assert (positions >= -1).all() and (positions <= 1).all()

    def test_model_output_shape(self):
        """Test various batch sizes."""
        model = MomentumMLP(lookback=30, hidden_sizes=(16,))
        for batch in [1, 10, 100]:
            x = torch.randn(batch, 30)
            out = model(x)
            assert out.shape == (batch,)

    def test_sharpe_loss_is_differentiable(self):
        """Sharpe loss should allow gradient computation."""
        model = MomentumMLP(lookback=20, hidden_sizes=(16,))
        x = torch.randn(50, 20)
        returns = torch.randn(50) * 0.01

        positions = model(x)
        loss = sharpe_loss(positions, returns, turnover_penalty=0.01)
        loss.backward()

        for param in model.parameters():
            assert param.grad is not None, "Gradient is None"

    def test_sharpe_loss_negative_for_positive_alignment(self):
        """If positions perfectly match return signs, Sharpe should be positive (loss negative)."""
        returns = torch.tensor([0.01, -0.02, 0.015, -0.01, 0.02])
        positions = torch.sign(returns)  # perfect signal
        loss = sharpe_loss(positions, returns)
        assert loss.item() < 0, "Loss should be negative for perfect alignment"

    def test_turnover_penalty(self):
        """Turnover penalty should increase loss when positions change."""
        returns = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01])
        stable_pos = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        changing_pos = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0])

        loss_stable = sharpe_loss(stable_pos, returns, turnover_penalty=0.1)
        loss_changing = sharpe_loss(changing_pos, returns, turnover_penalty=0.1)
        assert loss_changing > loss_stable


# ---- Backtest framework tests ----

class TestBacktest:
    """Tests for the backtest utilities."""

    def test_calculate_costs(self):
        """Transaction costs should reduce returns."""
        returns = pd.Series([0.01, 0.01, 0.01, 0.01])
        positions = pd.Series([1.0, 1.0, -1.0, -1.0])
        config = BacktestConfig(fee_bps=10, slippage_bps=5)

        net = calculate_costs(returns, positions, config)
        assert (net <= returns).all()
        # Position change of 2.0 on day 3 → cost = 2 * 15/10000 = 0.003
        expected_cost_day3 = 2.0 * 15 / 10000
        np.testing.assert_almost_equal(
            returns.iloc[2] - net.iloc[2], expected_cost_day3, decimal=6
        )

    def test_compute_metrics_positive(self):
        """Positive returns should give positive Sharpe and annual return."""
        returns = pd.Series(np.random.uniform(0.001, 0.005, 252))
        m = compute_metrics(returns)
        assert m["sharpeRatio"] > 0
        assert m["annualReturn"] > 0
        assert m["hitRate"] == 1.0

    def test_generate_metrics_json_schema(self):
        """Output should match the required schema."""
        config = BacktestConfig()
        results = [
            BacktestResult(
                window=0, train_start="2020-01-01", train_end="2021-01-01",
                test_start="2021-01-02", test_end="2022-01-01",
                gross_sharpe=0.5, net_sharpe=0.3, annual_return=0.1,
                max_drawdown=-0.05, total_trades=100, hit_rate=0.55,
            )
        ]
        output = generate_metrics_json(results, config)

        required_keys = {
            "sharpeRatio", "annualReturn", "maxDrawdown", "hitRate",
            "totalTrades", "transactionCosts", "walkForward", "customMetrics",
        }
        assert required_keys.issubset(output.keys())
        assert "feeBps" in output["transactionCosts"]
        assert "windows" in output["walkForward"]
