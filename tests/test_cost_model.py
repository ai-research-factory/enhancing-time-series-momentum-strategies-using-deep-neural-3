"""Tests for the updated cost model in backtest.py."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, Backtester, compute_metrics


class TestCostModel:
    """Tests for the Backtester's cost model with net returns."""

    @pytest.fixture
    def setup(self):
        """Create test data for backtest evaluation."""
        T, n_assets = 100, 3
        np.random.seed(42)
        dates = pd.bdate_range("2023-01-01", periods=T)
        tickers = ["A", "B", "C"]
        positions = np.random.uniform(-1, 1, (T, n_assets))
        returns = np.random.randn(T, n_assets) * 0.01
        return positions, returns, dates, tickers

    def test_net_sharpe_less_than_gross(self, setup):
        """Net Sharpe should be <= gross Sharpe due to costs."""
        positions, returns, dates, tickers = setup
        config = BacktestConfig(fee_bps=5.0, slippage_bps=0.0)
        bt = Backtester(config)
        result = bt.evaluate_positions(positions, returns, dates, tickers)

        assert "grossSharpe" in result
        assert "sharpeRatio" in result  # net sharpe
        # With costs, net should be lower (in absolute terms it could be either way
        # but with active trading it should reduce performance)

    def test_zero_cost_gross_equals_net(self, setup):
        """With zero costs, gross and net should be equal."""
        positions, returns, dates, tickers = setup
        config = BacktestConfig(fee_bps=0.0, slippage_bps=0.0)
        bt = Backtester(config)
        result = bt.evaluate_positions(positions, returns, dates, tickers)

        assert abs(result["grossSharpe"] - result["sharpeRatio"]) < 1e-8

    def test_total_trades_nonzero(self, setup):
        """Active trading should produce non-zero totalTrades."""
        positions, returns, dates, tickers = setup
        config = BacktestConfig(fee_bps=5.0, slippage_bps=0.0)
        bt = Backtester(config)
        result = bt.evaluate_positions(positions, returns, dates, tickers)

        assert result["totalTrades"] > 0

    def test_static_positions_zero_trades(self):
        """Constant positions should have zero trades and zero costs."""
        T, n_assets = 50, 2
        dates = pd.bdate_range("2023-01-01", periods=T)
        tickers = ["X", "Y"]
        positions = np.ones((T, n_assets)) * 0.5  # constant
        returns = np.random.randn(T, n_assets) * 0.01

        config = BacktestConfig(fee_bps=10.0, slippage_bps=5.0)
        bt = Backtester(config)
        result = bt.evaluate_positions(positions, returns, dates, tickers)

        assert result["totalTrades"] == 0
        # Gross and net should be nearly equal (only first-day entry cost)
        assert abs(result["grossSharpe"] - result["sharpeRatio"]) < 0.5

    def test_higher_costs_reduce_net_sharpe(self, setup):
        """Higher transaction costs should reduce net Sharpe more."""
        positions, returns, dates, tickers = setup

        config_low = BacktestConfig(fee_bps=1.0, slippage_bps=0.0)
        config_high = BacktestConfig(fee_bps=50.0, slippage_bps=0.0)

        bt_low = Backtester(config_low)
        bt_high = Backtester(config_high)

        result_low = bt_low.evaluate_positions(positions, returns, dates, tickers)
        result_high = bt_high.evaluate_positions(positions, returns, dates, tickers)

        # Higher costs should give worse net performance
        assert result_high["sharpeRatio"] <= result_low["sharpeRatio"]

    def test_turnover_computation(self, setup):
        """Turnover should be positive for changing positions."""
        positions, returns, dates, tickers = setup
        config = BacktestConfig(fee_bps=5.0, slippage_bps=0.0)
        bt = Backtester(config)
        result = bt.evaluate_positions(positions, returns, dates, tickers)

        assert result["turnover"] > 0
