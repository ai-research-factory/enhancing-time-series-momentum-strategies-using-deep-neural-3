"""
ARF Standard Backtest Framework
Walk-forward validation with transaction cost accounting.
Includes Backtester class for model vs baseline comparison.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    fee_bps: float = 10.0       # Transaction fee in basis points
    slippage_bps: float = 5.0   # Slippage in basis points
    train_ratio: float = 0.7    # Train window ratio for walk-forward
    n_splits: int = 10          # Number of walk-forward windows
    gap: int = 1                # Gap between train and test (prevent leakage)
    min_train_size: int = 252   # Minimum training samples (~1 year daily)


@dataclass
class BacktestResult:
    """Results from a single walk-forward window."""
    window: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    gross_sharpe: float = 0.0
    net_sharpe: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    hit_rate: float = 0.0
    pnl_series: Optional[pd.Series] = field(default=None, repr=False)


class WalkForwardValidator:
    """
    Walk-forward out-of-sample validation.

    Usage:
        validator = WalkForwardValidator(config)
        for train_idx, test_idx in validator.split(df):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            # Train model on train_df, evaluate on test_df
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def split(self, data: pd.DataFrame):
        """Generate train/test index pairs for walk-forward validation."""
        n = len(data)
        cfg = self.config
        test_size = max(1, (n - cfg.min_train_size) // cfg.n_splits)

        for i in range(cfg.n_splits):
            test_end = n - (cfg.n_splits - 1 - i) * test_size
            test_start = test_end - test_size
            train_end = test_start - cfg.gap
            train_start = max(0, int(train_end * (1 - cfg.train_ratio))) if cfg.train_ratio < 1.0 else 0

            if train_end - train_start < cfg.min_train_size:
                continue
            if test_start >= test_end:
                continue

            yield (
                list(range(train_start, train_end)),
                list(range(test_start, test_end)),
            )


def calculate_costs(returns: pd.Series, positions: pd.Series, config: BacktestConfig) -> pd.Series:
    """
    Calculate transaction costs from position changes.

    Args:
        returns: Gross returns series
        positions: Position series (-1, 0, 1 or continuous)
        config: Backtest configuration with fee/slippage settings

    Returns:
        Net returns after costs
    """
    trades = positions.diff().abs().fillna(0)
    cost_per_trade = (config.fee_bps + config.slippage_bps) / 10000
    costs = trades * cost_per_trade
    return returns - costs


def compute_metrics(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> dict:
    """
    Compute standard performance metrics from a returns series.

    Args:
        returns: Daily (or periodic) returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year (252 for daily, 365 for crypto)

    Returns:
        Dict with sharpeRatio, annualReturn, maxDrawdown, hitRate, totalTrades
    """
    if len(returns) == 0:
        return {"sharpeRatio": 0.0, "annualReturn": 0.0, "maxDrawdown": 0.0, "hitRate": 0.0}

    excess = returns - risk_free_rate / periods_per_year
    sharpe = float(np.sqrt(periods_per_year) * excess.mean() / excess.std()) if excess.std() > 0 else 0.0

    cumulative = (1 + returns).cumprod()
    annual_return = float(cumulative.iloc[-1] ** (periods_per_year / len(returns)) - 1)

    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min())

    hit_rate = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0.0

    return {
        "sharpeRatio": round(sharpe, 4),
        "annualReturn": round(annual_return, 4),
        "maxDrawdown": round(max_drawdown, 4),
        "hitRate": round(hit_rate, 4),
    }


class Backtester:
    """Backtester for comparing DNN model positions against baselines.

    Takes pre-computed positions and returns, computes portfolio-level
    cumulative returns, and evaluates against SMA crossover baseline.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def evaluate_positions(
        self,
        positions: np.ndarray,
        returns: np.ndarray,
        dates: pd.DatetimeIndex,
        tickers: list,
    ) -> dict:
        """Evaluate a set of multi-asset positions.

        Args:
            positions: (T, n_assets) position sizes
            returns: (T, n_assets) forward returns
            dates: DatetimeIndex of length T
            tickers: list of asset names

        Returns:
            dict with gross/net sharpe, annual_return, max_drawdown, hit_rate,
            turnover, totalTrades, cumulative_returns series.
        """
        pos_df = pd.DataFrame(positions, index=dates, columns=tickers)
        ret_df = pd.DataFrame(returns, index=dates, columns=tickers)

        # Portfolio return: equal-weighted average of position * return across assets
        port_gross = (pos_df * ret_df).mean(axis=1)

        # Compute per-asset costs, then average for portfolio-level net return
        cost_bps = (self.config.fee_bps + self.config.slippage_bps) / 10000
        pos_changes = pos_df.diff().abs().fillna(0)
        per_asset_costs = pos_changes * cost_bps
        port_costs = per_asset_costs.mean(axis=1)
        port_net = port_gross - port_costs

        gross_metrics = compute_metrics(port_gross)
        net_metrics = compute_metrics(port_net)
        cumulative = (1 + port_net).cumprod()

        # Turnover: average absolute position change per day, annualized
        turnover = pos_df.diff().abs().mean().mean() * 252

        # Total trades: count days where any asset had a meaningful position change
        trade_threshold = 0.01  # position change > 1% counts as a trade
        total_trades = int((pos_changes > trade_threshold).any(axis=1).sum())

        result = net_metrics.copy()
        result["grossSharpe"] = gross_metrics["sharpeRatio"]
        result["grossReturn"] = gross_metrics["annualReturn"]
        result["turnover"] = round(float(turnover), 4)
        result["totalTrades"] = total_trades
        result["cumulative_returns"] = cumulative
        result["daily_returns"] = port_net
        result["daily_gross_returns"] = port_gross

        return result

    @staticmethod
    def sma_crossover_positions(
        raw_prices: pd.DataFrame,
        short_window: int = 20,
        long_window: int = 60,
    ) -> pd.DataFrame:
        """Generate SMA crossover positions for all assets.

        Long (+1) when short SMA > long SMA, else flat (0).

        Args:
            raw_prices: DataFrame of close prices (dates x tickers)
            short_window: short moving average window (default 20)
            long_window: long moving average window (default 60)

        Returns:
            DataFrame of positions (dates x tickers), values in {0, 1}
        """
        sma_short = raw_prices.rolling(short_window, min_periods=short_window).mean()
        sma_long = raw_prices.rolling(long_window, min_periods=long_window).mean()
        positions = (sma_short > sma_long).astype(float)
        return positions

    def compute_baselines(
        self,
        returns: np.ndarray,
        dates: pd.DatetimeIndex,
        tickers: list,
        raw_prices: pd.DataFrame = None,
    ) -> dict:
        """Compute baseline strategy metrics.

        Args:
            returns: (T, n_assets) forward returns for the test period
            dates: test period dates
            tickers: asset names
            raw_prices: full price DataFrame for SMA computation

        Returns:
            dict of baseline metrics for customMetrics
        """
        ret_df = pd.DataFrame(returns, index=dates, columns=tickers)
        custom = {}

        # Baseline 1: 1/N equal weight (fully invested)
        ones = pd.DataFrame(1.0, index=dates, columns=tickers)
        port_1n = ret_df.mean(axis=1)  # equal-weighted return
        avg_pos_1n = ones.mean(axis=1)
        net_1n = calculate_costs(port_1n, avg_pos_1n, self.config)
        m_1n = compute_metrics(net_1n)
        custom["baseline_1n_sharpe"] = m_1n["sharpeRatio"]
        custom["baseline_1n_return"] = m_1n["annualReturn"]
        custom["baseline_1n_drawdown"] = m_1n["maxDrawdown"]

        # Baseline 2: Vol-targeted 1/N (10% target vol)
        target_vol = 0.10
        rolling_vol = port_1n.rolling(60, min_periods=20).std() * np.sqrt(252)
        vol_scale = (target_vol / (rolling_vol + 1e-8)).clip(0, 2).fillna(1.0)
        port_vol = vol_scale * port_1n
        avg_pos_vol = vol_scale
        net_vol = calculate_costs(port_vol, avg_pos_vol, self.config)
        m_vol = compute_metrics(net_vol)
        custom["baseline_voltarget_sharpe"] = m_vol["sharpeRatio"]

        # Baseline 3: Simple momentum (long top 50% by 252-day return)
        cum_ret = ret_df.rolling(252, min_periods=60).sum()
        # Rank: top 50% get long position
        mom_pos = cum_ret.apply(
            lambda row: (row >= row.median()).astype(float), axis=1
        ).fillna(0)
        port_mom = (mom_pos * ret_df).mean(axis=1)
        avg_pos_mom = mom_pos.mean(axis=1)
        net_mom = calculate_costs(port_mom, avg_pos_mom, self.config)
        m_mom = compute_metrics(net_mom)
        custom["baseline_momentum_sharpe"] = m_mom["sharpeRatio"]

        # Baseline 4: SMA crossover (20/60 day)
        if raw_prices is not None:
            sma_pos = self.sma_crossover_positions(raw_prices, 20, 60)
            # Align SMA positions with test dates
            sma_pos_test = sma_pos.reindex(dates).fillna(0)
            # Ensure columns match
            common_cols = [c for c in tickers if c in sma_pos_test.columns]
            sma_pos_aligned = sma_pos_test[common_cols]
            ret_aligned = ret_df[common_cols]
            port_sma = (sma_pos_aligned * ret_aligned).mean(axis=1)
            avg_pos_sma = sma_pos_aligned.mean(axis=1)
            net_sma = calculate_costs(port_sma, avg_pos_sma, self.config)
            m_sma = compute_metrics(net_sma)
            custom["baseline_sma_crossover_sharpe"] = m_sma["sharpeRatio"]
            custom["baseline_sma_crossover_return"] = m_sma["annualReturn"]
            custom["baseline_sma_crossover_drawdown"] = m_sma["maxDrawdown"]

        return custom


def generate_metrics_json(
    results: list[BacktestResult],
    config: BacktestConfig,
    custom_metrics: Optional[dict] = None,
) -> dict:
    """
    Generate ARF-standard metrics.json from walk-forward results.

    Args:
        results: List of BacktestResult from each window
        config: Backtest configuration
        custom_metrics: Optional paper-specific metrics

    Returns:
        Dict matching ARF metrics.json schema
    """
    if not results:
        return {
            "sharpeRatio": 0.0, "annualReturn": 0.0, "maxDrawdown": 0.0,
            "hitRate": 0.0, "totalTrades": 0,
            "transactionCosts": {"feeBps": config.fee_bps, "slippageBps": config.slippage_bps, "netSharpe": 0.0},
            "walkForward": {"windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0},
            "customMetrics": custom_metrics or {},
        }

    net_sharpes = [r.net_sharpe for r in results]
    positive_windows = sum(1 for s in net_sharpes if s > 0)

    return {
        "sharpeRatio": round(float(np.mean([r.gross_sharpe for r in results])), 4),
        "annualReturn": round(float(np.mean([r.annual_return for r in results])), 4),
        "maxDrawdown": round(float(min(r.max_drawdown for r in results)), 4),
        "hitRate": round(float(np.mean([r.hit_rate for r in results])), 4),
        "totalTrades": sum(r.total_trades for r in results),
        "transactionCosts": {
            "feeBps": config.fee_bps,
            "slippageBps": config.slippage_bps,
            "netSharpe": round(float(np.mean(net_sharpes)), 4),
        },
        "walkForward": {
            "windows": len(results),
            "positiveWindows": positive_windows,
            "avgOosSharpe": round(float(np.mean(net_sharpes)), 4),
        },
        "customMetrics": custom_metrics or {},
    }
