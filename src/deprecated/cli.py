"""CLI entry point for running experiments."""
import argparse
import yaml
from src.train import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Deep Momentum Network Experiment")
    parser.add_argument("command", choices=["run-experiment"])
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Convert hidden_sizes list to tuple
    if "hidden_sizes" in config:
        config["hidden_sizes"] = tuple(config["hidden_sizes"])

    run_experiment(**config)


if __name__ == "__main__":
    main()
