"""
Walk-forward validation framework with monthly rebalancing support.

Implements WalkForwardValidator for strict out-of-sample evaluation
using expanding windows with configurable train/test periods.
Supports monthly rebalancing as specified in the paper.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.backtest import (
    BacktestConfig, BacktestResult, Backtester,
    compute_metrics, generate_metrics_json,
)
from src.data import DataLoader
from src.loss import sharpe_loss
from src.model import DeepMomentumLSTM


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    n_splits: int = 5
    train_years: int = 3
    test_years: int = 1
    gap_days: int = 1
    rebalance_frequency: str = "monthly"  # "daily" or "monthly"
    lookback: int = 252  # 12 months as per paper spec
    # Training params
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 256
    turnover_penalty: float = 0.1
    # Cost params
    fee_bps: float = 5.0
    slippage_bps: float = 5.0


class WalkForwardValidator:
    """Walk-forward out-of-sample validator with monthly rebalancing.

    Splits the dataset into n_splits chronological folds using
    expanding training windows and fixed-length test windows.
    Supports monthly rebalancing: positions are only updated at
    month-end boundaries within each test fold.

    Usage:
        config = WalkForwardConfig(n_splits=5)
        validator = WalkForwardValidator(config)
        for fold_info in validator.generate_folds(dates):
            train_idx, test_idx = fold_info["train_idx"], fold_info["test_idx"]
            # Train on train_idx, evaluate on test_idx
    """

    def __init__(self, config: WalkForwardConfig):
        self.config = config

    def generate_folds(self, dates: pd.DatetimeIndex) -> list:
        """Generate train/test index splits for walk-forward validation.

        Uses expanding windows: training always starts from the beginning.
        Test windows are approximately test_years long.

        Args:
            dates: DatetimeIndex of all available sample dates

        Returns:
            List of dicts with keys: fold, train_idx, test_idx,
            train_start, train_end, test_start, test_end
        """
        n = len(dates)
        cfg = self.config
        trading_days_per_year = 252
        min_train_size = cfg.train_years * trading_days_per_year
        test_size = cfg.test_years * trading_days_per_year

        # Calculate folds from the end backwards
        folds = []
        for i in range(cfg.n_splits):
            test_end = n - (cfg.n_splits - 1 - i) * test_size
            test_start = test_end - test_size
            train_end = test_start - cfg.gap_days
            train_start = 0  # expanding window

            if train_end - train_start < min_train_size:
                continue
            if test_start < 0 or test_end > n:
                continue
            if test_start >= test_end:
                continue

            train_idx = list(range(train_start, train_end))
            test_idx = list(range(test_start, min(test_end, n)))

            folds.append({
                "fold": len(folds),
                "train_idx": train_idx,
                "test_idx": test_idx,
                "train_start": str(dates[train_idx[0]].date()),
                "train_end": str(dates[train_idx[-1]].date()),
                "test_start": str(dates[test_idx[0]].date()),
                "test_end": str(dates[test_idx[-1]].date()),
            })

        return folds

    def get_monthly_rebalance_mask(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Create boolean mask for month-end rebalancing dates.

        Returns a boolean array where True indicates the last trading
        day of each month — the only days positions should be updated.

        Args:
            dates: DatetimeIndex for the period

        Returns:
            Boolean array of length len(dates)
        """
        dates_series = pd.Series(dates)
        # Mark last trading day of each month
        month_groups = dates_series.dt.to_period("M")
        is_month_end = month_groups != month_groups.shift(-1)
        # First day is also a rebalance day (initial position)
        is_month_end.iloc[0] = True
        return is_month_end.values

    def apply_monthly_rebalancing(
        self, positions: np.ndarray, dates: pd.DatetimeIndex
    ) -> np.ndarray:
        """Apply monthly rebalancing: hold positions until next month-end.

        Between rebalance dates, positions are carried forward unchanged.

        Args:
            positions: (T, n_assets) raw model positions
            dates: DatetimeIndex of length T

        Returns:
            (T, n_assets) positions with monthly rebalancing applied
        """
        if self.config.rebalance_frequency == "daily":
            return positions

        rebal_mask = self.get_monthly_rebalance_mask(dates)
        rebalanced = np.zeros_like(positions)
        current_pos = np.zeros(positions.shape[1])

        for t in range(len(positions)):
            if rebal_mask[t]:
                current_pos = positions[t].copy()
            rebalanced[t] = current_pos

        return rebalanced


def _scale_features(train_X: np.ndarray, test_X: np.ndarray) -> tuple:
    """Scale 3D features using train data only."""
    n_train, lookback, n_assets = train_X.shape
    n_test = test_X.shape[0]

    train_flat = train_X.reshape(-1, n_assets)
    test_flat = test_X.reshape(-1, n_assets)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_flat).reshape(n_train, lookback, n_assets)
    test_scaled = scaler.transform(test_flat).reshape(n_test, lookback, n_assets)

    return train_scaled.astype(np.float32), test_scaled.astype(np.float32), scaler


def _train_fold(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    config: WalkForwardConfig,
    device: str,
) -> np.ndarray:
    """Train model on one fold and return test positions."""
    n_assets = train_X.shape[2]
    lookback = train_X.shape[1]

    train_scaled, test_scaled, _ = _scale_features(train_X, test_X)

    train_Xt = torch.tensor(train_scaled, dtype=torch.float32, device=device)
    train_yt = torch.tensor(train_y, dtype=torch.float32, device=device)
    test_Xt = torch.tensor(test_scaled, dtype=torch.float32, device=device)

    model = DeepMomentumLSTM(
        n_assets=n_assets, lookback=lookback,
        hidden_size=config.hidden_size, num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=1e-5
    )

    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = 20
    n_train = len(train_Xt)

    for epoch in range(config.epochs):
        model.train()
        epoch_losses = []
        indices = torch.randperm(n_train, device=device)

        for start in range(0, n_train, config.batch_size):
            end = min(start + config.batch_size, n_train)
            batch_idx = indices[start:end]
            batch_X = train_Xt[batch_idx]
            batch_y = train_yt[batch_idx]

            optimizer.zero_grad()
            positions = model(batch_X)
            loss = sharpe_loss(
                positions, batch_y,
                turnover_penalty=config.turnover_penalty,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_positions = model(test_Xt).cpu().numpy()

    return test_positions


def run_walk_forward(
    config: WalkForwardConfig = None,
    output_dir: str = "reports/cycle_5",
) -> dict:
    """Run full walk-forward validation with monthly rebalancing.

    Loads data, generates folds, trains model per fold, applies
    monthly rebalancing, evaluates with costs, computes baselines,
    and saves results.

    Args:
        config: Walk-forward configuration
        output_dir: Directory for output files

    Returns:
        metrics_json dict
    """
    if config is None:
        config = WalkForwardConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    # Load data - need to rebuild with new lookback if different
    print(f"Loading data with lookback={config.lookback}...")
    loader = DataLoader()
    loader.fetch_all()
    loader.lookback = config.lookback
    features, forward_returns, dates, tickers = loader.create_rolling_windows()
    print(f"Data shape: {features.shape}, dates: {dates[0].date()} to {dates[-1].date()}")

    # Also get raw prices for baselines
    close_frames = {}
    for ticker, df in loader.raw_data.items():
        close_frames[ticker] = df["close"].ffill()
    raw_prices = pd.DataFrame(close_frames).ffill()

    # Generate walk-forward folds
    validator = WalkForwardValidator(config)
    folds = validator.generate_folds(dates)
    print(f"Generated {len(folds)} walk-forward folds")

    if not folds:
        print("ERROR: No valid folds generated. Check data size vs config.")
        return {}

    # Cost config for backtester
    cost_config = BacktestConfig(
        fee_bps=config.fee_bps,
        slippage_bps=config.slippage_bps,
        n_splits=config.n_splits,
        gap=config.gap_days,
        min_train_size=config.train_years * 252,
    )
    backtester = Backtester(cost_config)

    # Train and evaluate each fold
    fold_results = []
    all_oos_positions = []
    all_oos_returns = []
    all_oos_dates = []

    for fold_info in folds:
        fold_i = fold_info["fold"]
        train_idx = fold_info["train_idx"]
        test_idx = fold_info["test_idx"]

        print(f"\n{'='*60}")
        print(f"Fold {fold_i+1}/{len(folds)}: "
              f"train={len(train_idx)}, test={len(test_idx)}")
        print(f"  Train: {fold_info['train_start']} to {fold_info['train_end']}")
        print(f"  Test:  {fold_info['test_start']} to {fold_info['test_end']}")

        train_X = features[train_idx]
        train_y = forward_returns[train_idx]
        test_X = features[test_idx]
        test_y = forward_returns[test_idx]
        test_dates = dates[test_idx]

        # Train and get raw positions
        raw_positions = _train_fold(train_X, train_y, test_X, config, device)

        # Apply monthly rebalancing
        rebalanced_positions = validator.apply_monthly_rebalancing(
            raw_positions, test_dates
        )

        # Evaluate with costs
        fold_eval = backtester.evaluate_positions(
            rebalanced_positions, test_y, test_dates, tickers
        )

        # Also evaluate raw (daily) positions for comparison
        raw_eval = backtester.evaluate_positions(
            raw_positions, test_y, test_dates, tickers
        )

        fold_result = {
            "fold": fold_i,
            "train_start": fold_info["train_start"],
            "train_end": fold_info["train_end"],
            "test_start": fold_info["test_start"],
            "test_end": fold_info["test_end"],
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            # Monthly rebalanced metrics
            "monthly_net_sharpe": fold_eval["sharpeRatio"],
            "monthly_gross_sharpe": fold_eval["grossSharpe"],
            "monthly_annual_return": fold_eval["annualReturn"],
            "monthly_max_drawdown": fold_eval["maxDrawdown"],
            "monthly_hit_rate": fold_eval["hitRate"],
            "monthly_turnover": fold_eval["turnover"],
            "monthly_total_trades": fold_eval["totalTrades"],
            # Daily rebalanced metrics (for comparison)
            "daily_net_sharpe": raw_eval["sharpeRatio"],
            "daily_gross_sharpe": raw_eval["grossSharpe"],
            "daily_turnover": raw_eval["turnover"],
        }
        fold_results.append(fold_result)

        all_oos_positions.append(rebalanced_positions)
        all_oos_returns.append(test_y)
        all_oos_dates.extend(test_dates)

        print(f"  Monthly: Net SR={fold_eval['sharpeRatio']:.4f}, "
              f"Gross SR={fold_eval['grossSharpe']:.4f}, "
              f"Turnover={fold_eval['turnover']:.4f}")
        print(f"  Daily:   Net SR={raw_eval['sharpeRatio']:.4f}, "
              f"Turnover={raw_eval['turnover']:.4f}")

    # Aggregate metrics across folds
    monthly_net_sharpes = [f["monthly_net_sharpe"] for f in fold_results]
    monthly_gross_sharpes = [f["monthly_gross_sharpe"] for f in fold_results]
    monthly_turnovers = [f["monthly_turnover"] for f in fold_results]
    daily_turnovers = [f["daily_turnover"] for f in fold_results]

    aggregate = {
        "n_folds": len(folds),
        "rebalance_frequency": config.rebalance_frequency,
        "lookback": config.lookback,
        "turnover_penalty": config.turnover_penalty,
        "avg_monthly_net_sharpe": round(float(np.mean(monthly_net_sharpes)), 4),
        "std_monthly_net_sharpe": round(float(np.std(monthly_net_sharpes)), 4),
        "avg_monthly_gross_sharpe": round(float(np.mean(monthly_gross_sharpes)), 4),
        "avg_monthly_turnover": round(float(np.mean(monthly_turnovers)), 4),
        "avg_daily_turnover": round(float(np.mean(daily_turnovers)), 4),
        "turnover_reduction_pct": round(
            (1 - np.mean(monthly_turnovers) / (np.mean(daily_turnovers) + 1e-8)) * 100, 2
        ),
        "avg_annual_return": round(
            float(np.mean([f["monthly_annual_return"] for f in fold_results])), 4
        ),
        "avg_max_drawdown": round(
            float(np.mean([f["monthly_max_drawdown"] for f in fold_results])), 4
        ),
        "avg_hit_rate": round(
            float(np.mean([f["monthly_hit_rate"] for f in fold_results])), 4
        ),
        "positive_folds": sum(1 for s in monthly_net_sharpes if s > 0),
        "fold_sharpe_ratios": [round(s, 4) for s in monthly_net_sharpes],
    }

    print(f"\n{'='*60}")
    print("Walk-Forward Summary (Monthly Rebalancing)")
    print(f"{'='*60}")
    print(f"  Avg Net Sharpe:  {aggregate['avg_monthly_net_sharpe']:.4f} "
          f"± {aggregate['std_monthly_net_sharpe']:.4f}")
    print(f"  Avg Gross Sharpe: {aggregate['avg_monthly_gross_sharpe']:.4f}")
    print(f"  Monthly Turnover: {aggregate['avg_monthly_turnover']:.4f}")
    print(f"  Daily Turnover:   {aggregate['avg_daily_turnover']:.4f}")
    print(f"  Turnover reduction: {aggregate['turnover_reduction_pct']:.1f}%")
    print(f"  Positive folds:   {aggregate['positive_folds']}/{len(folds)}")

    # Compute baselines on concatenated OOS returns
    last_fold_test_idx = folds[-1]["test_idx"]
    last_test_ret = forward_returns[last_fold_test_idx]
    last_test_dates = dates[last_fold_test_idx]
    custom_metrics = backtester.compute_baselines(
        last_test_ret, last_test_dates, tickers, raw_prices=raw_prices
    )

    # Add strategy metrics to custom
    custom_metrics["rebalance_frequency"] = "monthly"
    custom_metrics["lookback_days"] = config.lookback
    custom_metrics["dnn_monthly_net_sharpe"] = aggregate["avg_monthly_net_sharpe"]
    custom_metrics["dnn_monthly_gross_sharpe"] = aggregate["avg_monthly_gross_sharpe"]
    custom_metrics["dnn_monthly_turnover"] = aggregate["avg_monthly_turnover"]
    custom_metrics["dnn_daily_turnover"] = aggregate["avg_daily_turnover"]
    custom_metrics["dnn_hit_rate"] = aggregate["avg_hit_rate"]
    custom_metrics["turnover_reduction_pct"] = aggregate["turnover_reduction_pct"]
    custom_metrics["strategy_vs_1n_sharpe_diff"] = round(
        aggregate["avg_monthly_net_sharpe"] - custom_metrics.get("baseline_1n_sharpe", 0), 4
    )
    custom_metrics["strategy_vs_1n_return_diff"] = round(
        aggregate["avg_annual_return"] - custom_metrics.get("baseline_1n_return", 0), 4
    )
    custom_metrics["strategy_vs_1n_drawdown_diff"] = round(
        aggregate["avg_max_drawdown"] - custom_metrics.get("baseline_1n_drawdown", 0), 4
    )
    custom_metrics["strategy_vs_1n_turnover_ratio"] = round(
        aggregate["avg_monthly_turnover"], 4
    )
    custom_metrics["n_walk_forward_windows"] = len(folds)
    custom_metrics["mean_fold_sharpe"] = aggregate["avg_monthly_net_sharpe"]
    custom_metrics["std_fold_sharpe"] = aggregate["std_monthly_net_sharpe"]
    custom_metrics["fold_sharpe_ratios"] = aggregate["fold_sharpe_ratios"]

    # Build BacktestResult list for metrics.json
    backtest_results = []
    for fr in fold_results:
        backtest_results.append(BacktestResult(
            window=fr["fold"],
            train_start=fr["train_start"],
            train_end=fr["train_end"],
            test_start=fr["test_start"],
            test_end=fr["test_end"],
            gross_sharpe=fr["monthly_gross_sharpe"],
            net_sharpe=fr["monthly_net_sharpe"],
            annual_return=fr["monthly_annual_return"],
            max_drawdown=fr["monthly_max_drawdown"],
            total_trades=fr["monthly_total_trades"],
            hit_rate=fr["monthly_hit_rate"],
        ))

    # Generate metrics.json
    metrics_json = generate_metrics_json(backtest_results, cost_config, custom_metrics)

    # Save walk_forward_summary.json
    summary = {
        "description": "Walk-forward validation with monthly rebalancing (Cycle 5)",
        "config": {
            "n_splits": config.n_splits,
            "train_years": config.train_years,
            "test_years": config.test_years,
            "lookback": config.lookback,
            "rebalance_frequency": config.rebalance_frequency,
            "turnover_penalty": config.turnover_penalty,
            "fee_bps": config.fee_bps,
            "slippage_bps": config.slippage_bps,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
        },
        "aggregate": aggregate,
        "fold_results": fold_results,
    }

    summary_path = os.path.join(output_dir, "walk_forward_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWalk-forward summary saved to {summary_path}")

    # Save monthly_rebalance_results.json
    monthly_results = {
        "description": "Monthly rebalancing results vs daily (Cycle 5)",
        "rebalance_frequency": "monthly",
        "cost_model": {
            "fee_bps": config.fee_bps,
            "slippage_bps": config.slippage_bps,
        },
        "monthly_rebalance": {
            "avg_net_sharpe": aggregate["avg_monthly_net_sharpe"],
            "avg_gross_sharpe": aggregate["avg_monthly_gross_sharpe"],
            "avg_turnover": aggregate["avg_monthly_turnover"],
            "avg_annual_return": aggregate["avg_annual_return"],
            "avg_max_drawdown": aggregate["avg_max_drawdown"],
            "positive_folds": aggregate["positive_folds"],
            "fold_sharpes": aggregate["fold_sharpe_ratios"],
        },
        "daily_comparison": {
            "avg_daily_turnover": aggregate["avg_daily_turnover"],
            "turnover_reduction_pct": aggregate["turnover_reduction_pct"],
        },
    }

    monthly_path = os.path.join(output_dir, "monthly_rebalance_results.json")
    with open(monthly_path, "w") as f:
        json.dump(monthly_results, f, indent=2)
    print(f"Monthly rebalance results saved to {monthly_path}")

    # Save metrics.json
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics_json
