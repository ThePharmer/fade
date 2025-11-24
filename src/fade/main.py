"""
Main entry point for FADE proof of concept.

Usage:
    python -m fade.main [--epochs N] [--batch-size N] [--seed N]
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

from .config import (
    FADEConfig,
    ModelConfig,
    StrengthConfig,
    DegradationConfig,
    TrainingConfig,
)
from .fade_model import FADEModel
from .data import create_data_loaders
from .trainer import FADETrainer, BaselineTrainer


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_config(config: FADEConfig):
    """Print configuration summary."""
    print("\n" + "=" * 60)
    print("FADE Proof of Concept Configuration")
    print("=" * 60)
    print(f"\nModel:")
    print(f"  Vocab size: {config.model.vocab_size}")
    print(f"  Embedding dim: {config.model.d_model}")
    print(f"  Heads: {config.model.n_heads}")
    print(f"  Layers: {config.model.n_layers}")
    print(f"  FF dim: {config.model.d_ff}")
    print(f"\nStrength:")
    print(f"  Decay rate: {config.strength.decay_rate}")
    print(f"  Attention boost: {config.strength.attention_boost}")
    print(f"  Fuzzy threshold: {config.strength.fuzzy_threshold}")
    print(f"\nDegradation:")
    print(f"  Method: {config.degradation.method}")
    print(f"  Noise scale: {config.degradation.noise_scale}")
    print(f"\nTraining:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  KV pairs: {config.training.num_key_value_pairs}")
    print("=" * 60 + "\n")


def run_experiment(config: FADEConfig, run_baseline: bool = True) -> dict:
    """
    Run the full FADE experiment.

    Args:
        config: Configuration
        run_baseline: Whether to also train a baseline for comparison

    Returns:
        Dict with results
    """
    set_seed(config.seed)

    # Create data
    print("Creating dataset...")
    train_loader, eval_loader, dataset = create_data_loaders(
        num_pairs=config.training.num_key_value_pairs,
        key_length=config.training.key_length,
        value_length=config.training.value_length,
        vocab_size=config.model.vocab_size,
        batch_size=config.training.batch_size,
        seed=config.seed,
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Eval batches: {len(eval_loader)}")

    results = {}

    # Train FADE model
    print("\n" + "=" * 60)
    print("Training FADE Model")
    print("=" * 60)

    fade_model = FADEModel(config)
    print(f"Model parameters: {fade_model.count_parameters():,}")

    fade_trainer = FADETrainer(
        model=fade_model,
        config=config,
        train_loader=train_loader,
        eval_loader=eval_loader,
        dataset=dataset,
        device=config.device,
    )

    fade_history = fade_trainer.train()
    results["fade"] = {
        "history": fade_history,
        "best_ece": fade_trainer.state.best_ece,
        "best_correlation": fade_trainer.state.best_correlation,
    }

    # Train baseline for comparison
    if run_baseline:
        print("\n" + "=" * 60)
        print("Training Baseline Model (no degradation)")
        print("=" * 60)

        set_seed(config.seed)  # Reset seed for fair comparison

        baseline_model = FADEModel(config)

        baseline_trainer = BaselineTrainer(
            model=baseline_model,
            config=config,
            train_loader=train_loader,
            eval_loader=eval_loader,
            dataset=dataset,
            device=config.device,
        )

        baseline_history = baseline_trainer.train()
        results["baseline"] = {
            "history": baseline_history,
            "best_ece": baseline_trainer.state.best_ece,
            "best_correlation": baseline_trainer.state.best_correlation,
        }

        # Compare results
        print("\n" + "=" * 60)
        print("Results Comparison")
        print("=" * 60)
        print(f"\nFADE Model:")
        print(f"  Best ECE: {results['fade']['best_ece']:.4f}")
        print(f"  Best Fuzz-Error Correlation: {results['fade']['best_correlation']:.4f}")
        print(f"\nBaseline Model:")
        print(f"  Best ECE: {results['baseline']['best_ece']:.4f}")
        print(f"  Best Fuzz-Error Correlation: {results['baseline']['best_correlation']:.4f}")

        # Calculate improvement
        if results['baseline']['best_ece'] > 0:
            ece_improvement = (results['baseline']['best_ece'] - results['fade']['best_ece']) / results['baseline']['best_ece'] * 100
            print(f"\nECE Improvement: {ece_improvement:.1f}%")

            if ece_improvement > 5:
                print("SUCCESS: ECE improvement > 5% (minimum viable)")
            if ece_improvement > 10:
                print("STRONG SUCCESS: ECE improvement > 10%")

        corr_improvement = results['fade']['best_correlation'] - results['baseline']['best_correlation']
        print(f"Correlation Improvement: {corr_improvement:.4f}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="FADE Proof of Concept")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model arguments
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")

    # Task arguments
    parser.add_argument("--num-pairs", type=int, default=100, help="Number of KV pairs")
    parser.add_argument("--key-length", type=int, default=4, help="Key length")
    parser.add_argument("--value-length", type=int, default=4, help="Value length")

    # FADE arguments
    parser.add_argument("--decay-rate", type=float, default=0.1, help="Memory decay rate")
    parser.add_argument("--degradation", type=str, default="combined",
                       choices=["noise", "mask", "quantize", "combined"],
                       help="Degradation method")

    # Other arguments
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline training")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")

    args = parser.parse_args()

    # Build config with frozen dataclasses - construct with values upfront
    model_config = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )
    training_config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_key_value_pairs=args.num_pairs,
        key_length=args.key_length,
        value_length=args.value_length,
    )
    strength_config = StrengthConfig(
        decay_rate=args.decay_rate,
    )
    degradation_config = DegradationConfig(
        method=args.degradation,
    )
    config = FADEConfig(
        model=model_config,
        training=training_config,
        strength=strength_config,
        degradation=degradation_config,
        seed=args.seed,
        device=args.device,
    )

    print_config(config)

    # Run experiment
    results = run_experiment(config, run_baseline=not args.no_baseline)

    print("\nExperiment complete!")

    return results


if __name__ == "__main__":
    main()
