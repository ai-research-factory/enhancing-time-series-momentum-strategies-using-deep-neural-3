"""Main entry point for Deep Momentum Network experiments.

Supports:
- run-cost-analysis: Cycle 4 cost analysis with turnover regularization
- run-walk-forward: Cycle 5 walk-forward validation with monthly rebalancing
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from src.backtest import (
    BacktestConfig, Backtester, WalkForwardValidator,
    compute_metrics, calculate_costs, generate_metrics_json, BacktestResult,
)
from src.data import DataLoader
from src.model import DeepMomentumLSTM
from src.loss import sharpe_loss
from src.training import load_processed_data, _scale_features


def _train_and_evaluate_fold(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    turnover_penalty: float,
    epochs: int,
    hidden_size: int,
    num_layers: int,
    lr: float,
    batch_size: int,
    device: str,
) -> dict:
    """Train model on one fold and return test positions/returns."""
    n_assets = train_X.shape[2]
    lookback = train_X.shape[1]

    # Scale features using train data only
    train_scaled, test_scaled, scaler = _scale_features(train_X, test_X)

    train_Xt = torch.tensor(train_scaled, dtype=torch.float32, device=device)
    train_yt = torch.tensor(train_y, dtype=torch.float32, device=device)
    test_Xt = torch.tensor(test_scaled, dtype=torch.float32, device=device)

    model = DeepMomentumLSTM(
        n_assets=n_assets, lookback=lookback,
        hidden_size=hidden_size, num_layers=num_layers, dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = 20
    n_train = len(train_Xt)

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        indices = torch.randperm(n_train, device=device)

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_idx = indices[start:end]
            batch_X = train_Xt[batch_idx]
            batch_y = train_yt[batch_idx]

            optimizer.zero_grad()
            positions = model(batch_X)
            loss = sharpe_loss(positions, batch_y, turnover_penalty=turnover_penalty)
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

    return {"test_positions": test_positions, "test_returns": test_y}


def run_cost_analysis(
    epochs: int = 100,
    hidden_size: int = 64,
    num_layers: int = 2,
    lr: float = 1e-3,
    batch_size: int = 256,
    n_splits: int = 5,
    output_dir: str = "reports/cycle_4",
) -> dict:
    """Run cost analysis comparing regularized vs unregularized models.

    Trains two models using walk-forward validation:
    1. No turnover regularization (gamma=0.0)
    2. With turnover regularization (gamma=0.1)

    Evaluates both with transaction costs (5bps) and saves comparison.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processed data
    data = load_processed_data()
    features = data["features"]
    forward_returns = data["forward_returns"]
    dates = data["dates"]
    tickers = data["tickers"]
    n_samples = len(features)

    print(f"Loaded data: {features.shape}, {len(tickers)} assets")
    print(f"Date range: {dates[0].date()} to {dates[-1].date()}")

    # Cost config: 5bps as per paper
    # Use min_train_size=252 (1 year) to ensure >= 5 walk-forward windows
    cost_config = BacktestConfig(
        fee_bps=5.0, slippage_bps=0.0,
        n_splits=n_splits, gap=1, min_train_size=252,
    )

    # Create walk-forward folds manually to guarantee exactly n_splits folds
    min_train = 252
    test_size = (n_samples - min_train) // n_splits
    folds = []
    for i in range(n_splits):
        test_end = n_samples - (n_splits - 1 - i) * test_size
        test_start = test_end - test_size
        train_end = test_start - 1  # gap=1
        train_start = 0
        if train_end - train_start >= min_train and test_start < test_end:
            folds.append(
                (list(range(train_start, train_end)),
                 list(range(test_start, test_end)))
            )
    print(f"Walk-forward folds: {len(folds)}")

    # Load raw prices for baselines
    loader = DataLoader()
    loader.fetch_all()
    close_frames = {}
    for ticker, df in loader.raw_data.items():
        close_frames[ticker] = df["close"].ffill()
    raw_prices = pd.DataFrame(close_frames).ffill()

    # Run for both gamma values
    gamma_configs = [
        {"name": "no_regularization", "gamma": 0.0},
        {"name": "with_regularization", "gamma": 0.1},
    ]

    backtester = Backtester(cost_config)
    results = {}
    all_fold_results = []  # for metrics.json walk-forward

    for gc in gamma_configs:
        gamma = gc["gamma"]
        name = gc["name"]
        print(f"\n{'='*60}")
        print(f"Training: {name} (gamma={gamma})")
        print(f"{'='*60}")

        fold_metrics = []
        all_test_positions = []
        all_test_returns = []
        all_test_dates = []

        for fold_i, (train_idx, test_idx) in enumerate(folds):
            print(f"\n  Fold {fold_i+1}/{len(folds)}: "
                  f"train={len(train_idx)}, test={len(test_idx)}")
            print(f"  Train: {dates[train_idx[0]].date()} to {dates[train_idx[-1]].date()}")
            print(f"  Test:  {dates[test_idx[0]].date()} to {dates[test_idx[-1]].date()}")

            train_X = features[train_idx]
            train_y = forward_returns[train_idx]
            test_X = features[test_idx]
            test_y = forward_returns[test_idx]

            fold_result = _train_and_evaluate_fold(
                train_X, train_y, test_X, test_y,
                turnover_penalty=gamma,
                epochs=epochs, hidden_size=hidden_size,
                num_layers=num_layers, lr=lr,
                batch_size=batch_size, device=device,
            )

            test_pos = fold_result["test_positions"]
            test_ret = fold_result["test_returns"]
            test_dt = dates[test_idx]

            # Evaluate this fold
            fold_eval = backtester.evaluate_positions(
                test_pos, test_ret, test_dt, tickers
            )

            fold_metrics.append({
                "fold": fold_i,
                "net_sharpe": fold_eval["sharpeRatio"],
                "gross_sharpe": fold_eval["grossSharpe"],
                "annual_return": fold_eval["annualReturn"],
                "max_drawdown": fold_eval["maxDrawdown"],
                "hit_rate": fold_eval["hitRate"],
                "turnover": fold_eval["turnover"],
                "total_trades": fold_eval["totalTrades"],
                "test_start": str(test_dt[0].date()),
                "test_end": str(test_dt[-1].date()),
            })

            all_test_positions.append(test_pos)
            all_test_returns.append(test_ret)
            all_test_dates.extend(test_dt)

            print(f"  Net Sharpe: {fold_eval['sharpeRatio']:.4f}, "
                  f"Gross Sharpe: {fold_eval['grossSharpe']:.4f}, "
                  f"Turnover: {fold_eval['turnover']:.4f}")

        # Aggregate fold metrics
        net_sharpes = [f["net_sharpe"] for f in fold_metrics]
        gross_sharpes = [f["gross_sharpe"] for f in fold_metrics]
        turnovers = [f["turnover"] for f in fold_metrics]

        results[name] = {
            "gamma": gamma,
            "n_folds": len(folds),
            "fold_metrics": fold_metrics,
            "avg_net_sharpe": round(float(np.mean(net_sharpes)), 4),
            "std_net_sharpe": round(float(np.std(net_sharpes)), 4),
            "avg_gross_sharpe": round(float(np.mean(gross_sharpes)), 4),
            "avg_turnover": round(float(np.mean(turnovers)), 4),
            "avg_annual_return": round(float(np.mean([f["annual_return"] for f in fold_metrics])), 4),
            "avg_max_drawdown": round(float(np.mean([f["max_drawdown"] for f in fold_metrics])), 4),
            "avg_hit_rate": round(float(np.mean([f["hit_rate"] for f in fold_metrics])), 4),
            "total_trades": sum(f["total_trades"] for f in fold_metrics),
            "fold_sharpe_ratios": [round(s, 4) for s in net_sharpes],
            "positive_folds": sum(1 for s in net_sharpes if s > 0),
        }

        print(f"\n  Summary ({name}):")
        print(f"    Avg Net Sharpe:  {results[name]['avg_net_sharpe']:.4f} ± {results[name]['std_net_sharpe']:.4f}")
        print(f"    Avg Gross Sharpe: {results[name]['avg_gross_sharpe']:.4f}")
        print(f"    Avg Turnover:    {results[name]['avg_turnover']:.4f}")
        print(f"    Positive folds:  {results[name]['positive_folds']}/{len(folds)}")

        # Build walk-forward BacktestResults for the regularized model
        if name == "with_regularization":
            for fm in fold_metrics:
                all_fold_results.append(BacktestResult(
                    window=fm["fold"],
                    train_start=str(dates[folds[fm["fold"]][0][0]].date()),
                    train_end=str(dates[folds[fm["fold"]][0][-1]].date()),
                    test_start=fm["test_start"],
                    test_end=fm["test_end"],
                    gross_sharpe=fm["gross_sharpe"],
                    net_sharpe=fm["net_sharpe"],
                    annual_return=fm["annual_return"],
                    max_drawdown=fm["max_drawdown"],
                    total_trades=fm["total_trades"],
                    hit_rate=fm["hit_rate"],
                ))

    # Compute baselines on the last fold's test period
    last_fold_test_idx = folds[-1][1]
    last_test_ret = forward_returns[last_fold_test_idx]
    last_test_dates = dates[last_fold_test_idx]
    custom_metrics = backtester.compute_baselines(
        last_test_ret, last_test_dates, tickers, raw_prices=raw_prices
    )

    # Add regularization comparison to custom metrics
    reg = results["with_regularization"]
    noreg = results["no_regularization"]

    custom_metrics["dnn_no_reg_gross_sharpe"] = noreg["avg_gross_sharpe"]
    custom_metrics["dnn_no_reg_net_sharpe"] = noreg["avg_net_sharpe"]
    custom_metrics["dnn_no_reg_turnover"] = noreg["avg_turnover"]
    custom_metrics["dnn_reg_gross_sharpe"] = reg["avg_gross_sharpe"]
    custom_metrics["dnn_reg_net_sharpe"] = reg["avg_net_sharpe"]
    custom_metrics["dnn_reg_turnover"] = reg["avg_turnover"]
    custom_metrics["dnn_turnover"] = reg["avg_turnover"]
    custom_metrics["dnn_hit_rate"] = reg["avg_hit_rate"]
    custom_metrics["turnover_reduction_pct"] = round(
        (1 - reg["avg_turnover"] / (noreg["avg_turnover"] + 1e-8)) * 100, 2
    )
    custom_metrics["strategy_vs_1n_sharpe_diff"] = round(
        reg["avg_net_sharpe"] - custom_metrics.get("baseline_1n_sharpe", 0), 4
    )
    custom_metrics["strategy_vs_1n_return_diff"] = round(
        reg["avg_annual_return"] - custom_metrics.get("baseline_1n_return", 0), 4
    )
    custom_metrics["strategy_vs_1n_drawdown_diff"] = round(
        reg["avg_max_drawdown"] - custom_metrics.get("baseline_1n_drawdown", 0), 4
    )
    custom_metrics["strategy_vs_1n_turnover_ratio"] = round(reg["avg_turnover"], 4)
    custom_metrics["n_walk_forward_windows"] = len(folds)
    custom_metrics["mean_fold_sharpe"] = reg["avg_net_sharpe"]
    custom_metrics["std_fold_sharpe"] = reg["std_net_sharpe"]
    custom_metrics["fold_sharpe_ratios"] = reg["fold_sharpe_ratios"]

    # Save cost_analysis.json
    os.makedirs(output_dir, exist_ok=True)

    cost_analysis = {
        "description": "Comparison of DNN models with and without turnover regularization",
        "cost_model": {
            "fee_bps": cost_config.fee_bps,
            "slippage_bps": cost_config.slippage_bps,
            "total_cost_bps": cost_config.fee_bps + cost_config.slippage_bps,
        },
        "walk_forward": {
            "n_folds": len(folds),
            "min_train_size": cost_config.min_train_size,
            "gap": cost_config.gap,
        },
        "no_regularization": results["no_regularization"],
        "with_regularization": results["with_regularization"],
        "config": {
            "epochs": epochs,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "lr": lr,
            "batch_size": batch_size,
        },
    }

    cost_path = os.path.join(output_dir, "cost_analysis.json")
    with open(cost_path, "w") as f:
        json.dump(cost_analysis, f, indent=2)
    print(f"\nCost analysis saved to {cost_path}")

    # Build metrics.json (ARF schema) using regularized model
    metrics_json = generate_metrics_json(all_fold_results, cost_config, custom_metrics)
    # Override with computed values
    metrics_json["n_walk_forward_windows"] = len(folds)
    metrics_json["fold_sharpe_ratios"] = reg["fold_sharpe_ratios"]

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics_json


def main():
    parser = argparse.ArgumentParser(description="Deep Momentum Network")
    parser.add_argument(
        "command",
        choices=["run-basic-backtest", "run-cost-analysis", "run-walk-forward"],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--turnover-penalty", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--rebalance-frequency", default="monthly",
                        choices=["daily", "monthly"])
    parser.add_argument("--output-dir", default="reports/cycle_5")
    args = parser.parse_args()

    if args.command == "run-basic-backtest":
        from src.training import train_single_split
        train_single_split(
            epochs=args.epochs,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            lr=args.lr,
            turnover_penalty=args.turnover_penalty,
            batch_size=args.batch_size,
        )
    elif args.command == "run-cost-analysis":
        run_cost_analysis(
            epochs=args.epochs,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            lr=args.lr,
            batch_size=args.batch_size,
            n_splits=args.n_splits,
            output_dir=args.output_dir,
        )
    elif args.command == "run-walk-forward":
        from src.evaluation import WalkForwardConfig, run_walk_forward
        wf_config = WalkForwardConfig(
            n_splits=args.n_splits,
            lookback=args.lookback,
            rebalance_frequency=args.rebalance_frequency,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            turnover_penalty=args.turnover_penalty,
        )
        run_walk_forward(config=wf_config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
