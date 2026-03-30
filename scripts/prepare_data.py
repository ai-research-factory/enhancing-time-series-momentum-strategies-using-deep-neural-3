#!/usr/bin/env python3
"""
Prepare processed data for the deep momentum network.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --config config/assets.json

Fetches OHLCV data for all tickers in assets.json, computes log returns,
creates rolling-window features, and saves to data/processed/timeseries.pkl.
Also outputs data_summary.csv to reports/cycle_2/.
"""
import argparse
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Prepare data for deep momentum network")
    parser.add_argument("--config", default=None, help="Path to assets.json config")
    args = parser.parse_args()

    loader = DataLoader(config_path=args.config)

    # Step 1: Fetch all OHLCV data
    print("=" * 60)
    print("Step 1: Fetching OHLCV data from ARF Data API")
    print("=" * 60)
    loader.fetch_all()
    print(f"\nSuccessfully fetched {len(loader.raw_data)} / {len(loader.tickers)} tickers\n")

    # Step 2: Save data summary
    print("=" * 60)
    print("Step 2: Generating data summary")
    print("=" * 60)
    summary = loader.get_data_summary()
    summary_dir = os.path.join("reports", "cycle_2")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "data_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved to {summary_path}\n")

    # Step 3: Create rolling windows and save processed data
    print("=" * 60)
    print("Step 3: Creating rolling-window features and saving")
    print("=" * 60)
    output_path = loader.save_processed()

    # Step 4: Validate output
    print("\n" + "=" * 60)
    print("Step 4: Validation")
    print("=" * 60)
    import pickle
    import numpy as np

    with open(output_path, "rb") as f:
        data = pickle.load(f)

    features = data["features"]
    fwd = data["forward_returns"]
    print(f"Features shape: {features.shape} (samples, lookback, assets)")
    print(f"Forward returns shape: {fwd.shape} (samples, assets)")
    print(f"Features dtype: {features.dtype}")
    print(f"NaN in features: {np.isnan(features).any()}")
    print(f"NaN in forward_returns: {np.isnan(fwd).any()}")
    print(f"Date range: {data['dates'][0].date()} to {data['dates'][-1].date()}")
    print(f"Tickers ({len(data['tickers'])}): {data['tickers']}")

    assert features.ndim == 3, f"Expected 3D features, got {features.ndim}D"
    assert not np.isnan(features).any(), "Features contain NaN!"
    assert not np.isnan(fwd).any(), "Forward returns contain NaN!"
    print("\nAll validations passed!")


if __name__ == "__main__":
    main()
