"""Command-line entry point for the clinical risk modeling lab."""

import argparse
import json

from .data import generate_synthetic_dataset, load_dataset
from .experiment import run_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare independent tabular risk models")
    parser.add_argument("--data", help="Path to a CSV dataset")
    parser.add_argument(
        "--generate-demo-data",
        metavar="PATH",
        help="Generate reproducible synthetic data at PATH before running",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    data_path = args.data
    if args.generate_demo_data:
        data_path = generate_synthetic_dataset(args.generate_demo_data, seed=args.seed)
    if not data_path:
        parser.error("provide --data or --generate-demo-data")

    dataset = load_dataset(data_path)
    results = run_comparison(dataset, seed=args.seed, epochs=args.epochs)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
