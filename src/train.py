"""
Training and evaluation pipeline for Deep Momentum Network.

Implements walk-forward validation using the ARF backtest framework,
trains the MLP model with differentiable Sharpe ratio loss, and
produces metrics.json output.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.backtest import (
    BacktestConfig,
    BacktestResult,
    WalkForwardValidator,
    calculate_costs,
    compute_metrics,
    generate_metrics_json,
)
from src.data import fetch_ohlcv, prepare_features
from src.model import MomentumMLP, sharpe_loss


def train_one_window(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    lookback: int = 60,
    hidden_sizes: tuple = (64, 32),
    lr: float = 1e-3,
    epochs: int = 100,
    turnover_penalty: float = 0.01,
    device: str = "cpu",
) -> tuple:
    """
    Train model on one walk-forward window and return test positions.

    Returns:
        (test_positions, test_returns) as numpy arrays
    """
    # Standardize features using train data only
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)

    # Convert to tensors
    train_Xt = torch.tensor(train_X_scaled, dtype=torch.float32, device=device)
    train_yt = torch.tensor(train_y, dtype=torch.float32, device=device)
    test_Xt = torch.tensor(test_X_scaled, dtype=torch.float32, device=device)

    model = MomentumMLP(lookback=lookback, hidden_sizes=hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Training loop
    model.train()
    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = 15

    for epoch in range(epochs):
        optimizer.zero_grad()
        positions = model(train_Xt)
        loss = sharpe_loss(positions, train_yt, turnover_penalty=turnover_penalty)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        current_loss = loss.item()
        if current_loss < best_loss - 1e-6:
            best_loss = current_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Generate test predictions
    model.eval()
    with torch.no_grad():
        test_positions = model(test_Xt).cpu().numpy()

    return test_positions, test_y


def run_experiment(
    ticker: str = "^N225",
    lookback: int = 60,
    hidden_sizes: tuple = (64, 32),
    lr: float = 1e-3,
    epochs: int = 100,
    turnover_penalty: float = 0.01,
    n_splits: int = 5,
    output_dir: str = "reports/cycle_1",
) -> dict:
    """
    Run full walk-forward experiment and save results.

    Returns:
        metrics dict (same as metrics.json)
    """
    # Fetch and prepare data
    print(f"Fetching data for {ticker}...")
    df = fetch_ohlcv(ticker=ticker)
    print(f"Data shape: {df.shape}, range: {df.index[0]} to {df.index[-1]}")

    features, forward_returns, dates = prepare_features(df, lookback=lookback)
    print(f"Feature matrix: {features.shape}, forward returns: {forward_returns.shape}")

    # Build a DataFrame for walk-forward splitting
    data_df = pd.DataFrame(
        {"features_idx": range(len(features))},
        index=dates,
    )

    config = BacktestConfig(
        fee_bps=10.0,
        slippage_bps=5.0,
        n_splits=n_splits,
        gap=1,
        min_train_size=max(252, lookback + 50),
        train_ratio=0.7,
    )
    validator = WalkForwardValidator(config)

    results = []
    all_oos_positions = []
    all_oos_returns = []
    all_oos_dates = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for window_idx, (train_idx, test_idx) in enumerate(validator.split(data_df)):
        train_X = features[train_idx]
        train_y = forward_returns[train_idx]
        test_X = features[test_idx]
        test_y = forward_returns[test_idx]

        print(f"Window {window_idx}: train={len(train_idx)}, test={len(test_idx)}, "
              f"train_dates={dates[train_idx[0]].date()}..{dates[train_idx[-1]].date()}, "
              f"test_dates={dates[test_idx[0]].date()}..{dates[test_idx[-1]].date()}")

        test_positions, test_returns = train_one_window(
            train_X, train_y, test_X, test_y,
            lookback=lookback,
            hidden_sizes=hidden_sizes,
            lr=lr,
            epochs=epochs,
            turnover_penalty=turnover_penalty,
            device=device,
        )

        # Compute metrics for this window
        pos_series = pd.Series(test_positions, index=dates[test_idx])
        ret_series = pd.Series(test_returns, index=dates[test_idx])

        gross_returns = pos_series * ret_series
        net_returns = calculate_costs(gross_returns, pos_series, config)

        gross_metrics = compute_metrics(gross_returns)
        net_metrics = compute_metrics(net_returns)

        total_trades = int((pos_series.diff().abs() > 0.01).sum())

        result = BacktestResult(
            window=window_idx,
            train_start=str(dates[train_idx[0]].date()),
            train_end=str(dates[train_idx[-1]].date()),
            test_start=str(dates[test_idx[0]].date()),
            test_end=str(dates[test_idx[-1]].date()),
            gross_sharpe=gross_metrics["sharpeRatio"],
            net_sharpe=net_metrics["sharpeRatio"],
            annual_return=net_metrics["annualReturn"],
            max_drawdown=net_metrics["maxDrawdown"],
            total_trades=total_trades,
            hit_rate=net_metrics["hitRate"],
            pnl_series=net_returns,
        )
        results.append(result)

        all_oos_positions.extend(test_positions.tolist())
        all_oos_returns.extend(test_returns.tolist())
        all_oos_dates.extend(dates[test_idx].tolist())

        print(f"  Gross Sharpe: {gross_metrics['sharpeRatio']:.4f}, "
              f"Net Sharpe: {net_metrics['sharpeRatio']:.4f}")

    # Compute baselines on out-of-sample data
    oos_returns = pd.Series(all_oos_returns, index=pd.DatetimeIndex(all_oos_dates))
    oos_positions = pd.Series(all_oos_positions, index=pd.DatetimeIndex(all_oos_dates))

    custom_metrics = _compute_baselines(oos_returns, config)

    # Add strategy vs baseline comparisons
    strategy_net = compute_metrics(
        calculate_costs(oos_positions * oos_returns, oos_positions, config)
    )
    custom_metrics["strategy_vs_1n_sharpe_diff"] = round(
        strategy_net["sharpeRatio"] - custom_metrics.get("baseline_1n_sharpe", 0), 4
    )
    custom_metrics["strategy_vs_1n_return_diff"] = round(
        strategy_net["annualReturn"] - custom_metrics.get("baseline_1n_return", 0), 4
    )
    custom_metrics["strategy_vs_1n_drawdown_diff"] = round(
        strategy_net["maxDrawdown"] - custom_metrics.get("baseline_1n_drawdown", 0), 4
    )

    # Generate final metrics
    metrics = generate_metrics_json(results, config, custom_metrics)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    return metrics


def _compute_baselines(oos_returns: pd.Series, config: BacktestConfig) -> dict:
    """Compute baseline strategy metrics on out-of-sample returns."""
    custom = {}

    # Baseline 1: 1/N (buy and hold, position=1)
    buy_hold_pos = pd.Series(1.0, index=oos_returns.index)
    buy_hold_gross = oos_returns.copy()
    buy_hold_net = calculate_costs(buy_hold_gross, buy_hold_pos, config)
    bh_metrics = compute_metrics(buy_hold_net)
    custom["baseline_1n_sharpe"] = bh_metrics["sharpeRatio"]
    custom["baseline_1n_return"] = bh_metrics["annualReturn"]
    custom["baseline_1n_drawdown"] = bh_metrics["maxDrawdown"]

    # Baseline 2: Vol-targeted 1/N (target 10% annualized vol)
    target_vol = 0.10
    rolling_vol = oos_returns.rolling(60, min_periods=20).std() * np.sqrt(252)
    vol_scale = target_vol / (rolling_vol + 1e-8)
    vol_scale = vol_scale.clip(0, 2)  # cap leverage
    vol_pos = vol_scale.fillna(1.0)
    vol_gross = vol_pos * oos_returns
    vol_net = calculate_costs(vol_gross, vol_pos, config)
    vol_metrics = compute_metrics(vol_net)
    custom["baseline_voltarget_sharpe"] = vol_metrics["sharpeRatio"]

    # Baseline 3: Simple momentum (sign of 252-day cumulative return)
    # For single asset: long if trailing return > 0, else flat
    cum_ret = oos_returns.rolling(252, min_periods=60).sum()
    mom_pos = (cum_ret > 0).astype(float).fillna(0)
    mom_gross = mom_pos * oos_returns
    mom_net = calculate_costs(mom_gross, mom_pos, config)
    mom_metrics = compute_metrics(mom_net)
    custom["baseline_momentum_sharpe"] = mom_metrics["sharpeRatio"]

    return custom


if __name__ == "__main__":
    metrics = run_experiment()
    print("\n=== Final Metrics ===")
    print(json.dumps(metrics, indent=2))
