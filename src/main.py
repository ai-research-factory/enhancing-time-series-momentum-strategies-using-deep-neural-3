"""Main entry point for Cycle 3: basic backtest with DNN vs baselines."""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

from src.backtest import BacktestConfig, Backtester, compute_metrics, calculate_costs
from src.data import DataLoader
from src.training import train_single_split, load_processed_data


def run_basic_backtest(
    epochs: int = 100,
    hidden_size: int = 64,
    num_layers: int = 2,
    lr: float = 1e-3,
    turnover_penalty: float = 0.01,
    batch_size: int = 256,
    output_dir: str = "reports/cycle_3",
) -> dict:
    """Run basic backtest: train DNN on 80/20 split, compare with baselines.

    Returns:
        dict with full results including metrics for DNN and baselines.
    """
    # Load processed data
    data = load_processed_data()
    features = data["features"]
    forward_returns = data["forward_returns"]
    dates = data["dates"]
    tickers = data["tickers"]

    print(f"Loaded data: {features.shape}, {len(tickers)} assets")
    print(f"Date range: {dates[0].date()} to {dates[-1].date()}")

    # Train on single 80/20 split
    result = train_single_split(
        features=features,
        forward_returns=forward_returns,
        train_ratio=0.8,
        hidden_size=hidden_size,
        num_layers=num_layers,
        lr=lr,
        epochs=epochs,
        turnover_penalty=turnover_penalty,
        batch_size=batch_size,
    )

    test_positions = result["test_positions"]
    test_returns = result["test_returns"]
    test_dates = dates[result["test_indices"]]

    print(f"\nTest period: {test_dates[0].date()} to {test_dates[-1].date()}")
    print(f"Test samples: {len(test_dates)}")

    # Evaluate DNN positions
    config = BacktestConfig(fee_bps=10.0, slippage_bps=5.0)
    backtester = Backtester(config)

    dnn_metrics = backtester.evaluate_positions(
        test_positions, test_returns, test_dates, tickers
    )

    print(f"\n=== DNN Model Results ===")
    print(f"Sharpe Ratio: {dnn_metrics['sharpeRatio']:.4f}")
    print(f"Annual Return: {dnn_metrics['annualReturn']:.4f}")
    print(f"Max Drawdown: {dnn_metrics['maxDrawdown']:.4f}")
    print(f"Hit Rate: {dnn_metrics['hitRate']:.4f}")
    print(f"Turnover: {dnn_metrics['turnover']:.4f}")

    # Load raw prices for SMA baseline
    loader = DataLoader()
    loader.fetch_all()
    close_frames = {}
    for ticker, df in loader.raw_data.items():
        close_frames[ticker] = df["close"].ffill()
    raw_prices = pd.DataFrame(close_frames).ffill()

    # Compute baselines
    custom_metrics = backtester.compute_baselines(
        test_returns, test_dates, tickers, raw_prices=raw_prices
    )

    # Add DNN vs baseline comparisons
    custom_metrics["strategy_vs_1n_sharpe_diff"] = round(
        dnn_metrics["sharpeRatio"] - custom_metrics.get("baseline_1n_sharpe", 0), 4
    )
    custom_metrics["strategy_vs_1n_return_diff"] = round(
        dnn_metrics["annualReturn"] - custom_metrics.get("baseline_1n_return", 0), 4
    )
    custom_metrics["strategy_vs_1n_drawdown_diff"] = round(
        dnn_metrics["maxDrawdown"] - custom_metrics.get("baseline_1n_drawdown", 0), 4
    )

    # Turnover ratio
    baseline_1n_turnover = 0.0  # buy-and-hold has ~0 turnover
    if baseline_1n_turnover > 0:
        custom_metrics["strategy_vs_1n_turnover_ratio"] = round(
            dnn_metrics["turnover"] / baseline_1n_turnover, 2
        )
    else:
        custom_metrics["strategy_vs_1n_turnover_ratio"] = round(dnn_metrics["turnover"], 4)

    # DNN-specific metrics
    custom_metrics["dnn_turnover"] = dnn_metrics["turnover"]
    custom_metrics["dnn_hit_rate"] = dnn_metrics["hitRate"]

    print(f"\n=== Baseline Comparisons ===")
    for k, v in custom_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # Build metrics.json (ARF schema)
    metrics_json = {
        "sharpeRatio": dnn_metrics["sharpeRatio"],
        "annualReturn": dnn_metrics["annualReturn"],
        "maxDrawdown": dnn_metrics["maxDrawdown"],
        "hitRate": dnn_metrics["hitRate"],
        "totalTrades": 0,
        "transactionCosts": {
            "feeBps": config.fee_bps,
            "slippageBps": config.slippage_bps,
            "netSharpe": dnn_metrics["sharpeRatio"],
        },
        "walkForward": {
            "windows": 0,
            "positiveWindows": 0,
            "avgOosSharpe": 0.0,
        },
        "customMetrics": custom_metrics,
    }

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    # basic_backtest.json — detailed results
    backtest_json = {
        "dnn_model": {
            "sharpe_ratio": dnn_metrics["sharpeRatio"],
            "annual_return": dnn_metrics["annualReturn"],
            "max_drawdown": dnn_metrics["maxDrawdown"],
            "hit_rate": dnn_metrics["hitRate"],
            "turnover": dnn_metrics["turnover"],
        },
        "baselines": {
            "equal_weight_1n": {
                "sharpe_ratio": custom_metrics.get("baseline_1n_sharpe", 0),
                "annual_return": custom_metrics.get("baseline_1n_return", 0),
                "max_drawdown": custom_metrics.get("baseline_1n_drawdown", 0),
            },
            "vol_targeted_1n": {
                "sharpe_ratio": custom_metrics.get("baseline_voltarget_sharpe", 0),
            },
            "simple_momentum": {
                "sharpe_ratio": custom_metrics.get("baseline_momentum_sharpe", 0),
            },
            "sma_crossover_20_60": {
                "sharpe_ratio": custom_metrics.get("baseline_sma_crossover_sharpe", 0),
                "annual_return": custom_metrics.get("baseline_sma_crossover_return", 0),
                "max_drawdown": custom_metrics.get("baseline_sma_crossover_drawdown", 0),
            },
        },
        "config": {
            "train_ratio": 0.8,
            "epochs": epochs,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "lr": lr,
            "turnover_penalty": turnover_penalty,
            "batch_size": batch_size,
            "n_assets": len(tickers),
            "tickers": tickers,
        },
        "data": {
            "total_samples": len(features),
            "train_samples": int(len(features) * 0.8),
            "test_samples": len(features) - int(len(features) * 0.8),
            "test_start": str(test_dates[0].date()),
            "test_end": str(test_dates[-1].date()),
        },
    }

    backtest_path = os.path.join(output_dir, "basic_backtest.json")
    with open(backtest_path, "w") as f:
        json.dump(backtest_json, f, indent=2)
    print(f"\nBasic backtest results saved to {backtest_path}")

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics_json


def main():
    parser = argparse.ArgumentParser(description="Deep Momentum Network — Cycle 3")
    parser.add_argument("command", choices=["run-basic-backtest"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--turnover-penalty", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", default="reports/cycle_3")
    args = parser.parse_args()

    if args.command == "run-basic-backtest":
        run_basic_backtest(
            epochs=args.epochs,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            lr=args.lr,
            turnover_penalty=args.turnover_penalty,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
