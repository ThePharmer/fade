#!/usr/bin/env python3
"""
Quick script to run the FADE proof of concept.

Usage:
    python run_poc.py [--quick] [--epochs N]
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fade.config import get_default_config
from fade.main import run_experiment, print_config


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run FADE POC")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer epochs")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline comparison")
    args = parser.parse_args()

    config = get_default_config()

    if args.quick:
        # Quick settings for testing
        config.training.num_epochs = 10
        config.training.num_key_value_pairs = 50
        config.training.eval_every = 2
        config.model.d_model = 64
        config.model.n_layers = 1
        print("Running in quick mode...")

    if args.epochs:
        config.training.num_epochs = args.epochs

    print_config(config)

    results = run_experiment(config, run_baseline=not args.no_baseline)

    # Summary
    print("\n" + "=" * 60)
    print("POC SUMMARY")
    print("=" * 60)

    fade_ece = results["fade"]["best_ece"]
    fade_corr = results["fade"]["best_correlation"]

    print(f"\nFADE Results:")
    print(f"  ECE: {fade_ece:.4f}")
    print(f"  Fuzziness-Error Correlation: {fade_corr:.4f}")

    if "baseline" in results:
        baseline_ece = results["baseline"]["best_ece"]
        baseline_corr = results["baseline"]["best_correlation"]

        print(f"\nBaseline Results:")
        print(f"  ECE: {baseline_ece:.4f}")
        print(f"  Fuzziness-Error Correlation: {baseline_corr:.4f}")

        if baseline_ece > 0:
            improvement = (baseline_ece - fade_ece) / baseline_ece * 100
            print(f"\nECE Improvement: {improvement:.1f}%")

        if fade_corr > 0:
            print(f"\nFuzziness-Error Correlation is POSITIVE ({fade_corr:.4f})")
            print("This means fuzziness successfully predicts errors!")

    print("\n" + "=" * 60)

    return results


if __name__ == "__main__":
    main()
