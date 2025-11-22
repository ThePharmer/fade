#!/usr/bin/env python3
"""
DEFINITIVE FADE TEST - Run this for a clear answer.

Tests whether FADE's improvement comes from:
A) The specific strength-computation mechanism (targeted degradation)
B) General noise regularization (any noise helps)
C) The calibration loss term (explicitly training fuzziness to predict errors)

Hardware: Designed for GTX 1080 + i9 9900k
Runtime: ~10-15 minutes

Usage:
    python definitive_test.py [--quick]
"""

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from fade.config import get_default_config
from fade.fade_model import FADEModel
from fade.data import create_data_loaders
from fade.strength import StrengthTracker


@dataclass
class TestResult:
    name: str
    ece: float
    correlation: float
    accuracy: float
    final_loss: float


def compute_ece(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == targets).float()

    confidences = confidences.flatten().cpu().numpy()
    accuracies = accuracies.flatten().cpu().numpy()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += in_bin.sum() * abs(avg_accuracy - avg_confidence)

    return ece / len(confidences) if len(confidences) > 0 else 0.0


def compute_correlation(fuzziness: torch.Tensor, errors: torch.Tensor) -> float:
    """Compute correlation between fuzziness and errors."""
    fuzz = fuzziness.flatten().cpu().numpy()
    err = errors.flatten().cpu().numpy()

    if fuzz.std() < 1e-6 or err.std() < 1e-6:
        return 0.0

    corr = np.corrcoef(fuzz, err)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


class TestTrainer:
    """Unified trainer for all test conditions."""

    def __init__(
        self,
        model: FADEModel,
        config,
        train_loader,
        eval_loader,
        device: str,
        strength_mode: str = "computed",  # "computed", "random", "constant", "none"
        use_calibration_loss: bool = True,
        use_degradation: bool = True,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.strength_mode = strength_mode
        self.use_calibration_loss = use_calibration_loss
        self.use_degradation = use_degradation

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.training.num_epochs
        )

        # Store original method for restoration
        self._original_get_strengths = StrengthTracker.get_strengths

        # Patch strength computation based on mode
        if strength_mode == "random":
            def random_strengths(tracker_self, seq_len):
                batch_size = tracker_self.strengths.shape[0]
                # Match the distribution of computed strengths (roughly 0.3-0.8)
                return torch.rand(batch_size, seq_len, device=tracker_self.strengths.device) * 0.5 + 0.3
            StrengthTracker.get_strengths = random_strengths

        elif strength_mode == "constant":
            def constant_strengths(tracker_self, seq_len):
                batch_size = tracker_self.strengths.shape[0]
                return torch.ones(batch_size, seq_len, device=tracker_self.strengths.device) * 0.5
            StrengthTracker.get_strengths = constant_strengths

    def restore_strengths(self):
        """Restore original strength computation."""
        StrengthTracker.get_strengths = self._original_get_strengths

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        n_batches = 0

        time_per_batch = self.config.training.time_steps_per_epoch / len(self.train_loader)

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)

            if self.strength_mode == "computed":
                self.model.advance_time(time_per_batch)

            self.optimizer.zero_grad()
            output = self.model(
                input_ids,
                apply_degradation=self.use_degradation,
                return_fuzziness=True
            )

            # Prediction loss
            pred_loss = F.cross_entropy(
                output["logits"].view(-1, output["logits"].size(-1)),
                targets.view(-1),
                reduction="none"
            ).view(targets.shape)
            pred_loss = (pred_loss * mask).sum() / mask.sum().clamp(min=1)

            # Calibration loss (optional)
            loss = pred_loss
            if self.use_calibration_loss and "fuzziness" in output:
                predictions = output["logits"].argmax(dim=-1)
                errors = (predictions != targets).float()
                cal_loss = F.mse_loss(output["fuzziness"], errors)
                loss = loss + 0.1 * cal_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()
        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()

        all_logits = []
        all_targets = []
        all_fuzziness = []
        all_masks = []

        for batch in self.eval_loader:
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)

            output = self.model(input_ids, apply_degradation=False, return_fuzziness=True)

            all_logits.append(output["logits"])
            all_targets.append(targets)
            all_fuzziness.append(output.get("fuzziness", torch.zeros_like(mask.float())))
            all_masks.append(mask)

        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        fuzziness = torch.cat(all_fuzziness, dim=0)
        masks = torch.cat(all_masks, dim=0)

        # Apply mask
        logits_flat = logits[masks.bool()]
        targets_flat = targets[masks.bool()]
        fuzz_flat = fuzziness[masks.bool()]

        # Metrics
        ece = compute_ece(logits_flat.unsqueeze(0), targets_flat.unsqueeze(0))

        predictions = logits_flat.argmax(dim=-1)
        errors = (predictions != targets_flat).float()
        accuracy = 1.0 - errors.mean().item()
        correlation = compute_correlation(fuzz_flat, errors)

        return {"ece": ece, "correlation": correlation, "accuracy": accuracy}

    def train(self, num_epochs: int, verbose: bool = True) -> TestResult:
        self.model.reset_memory(self.config.training.batch_size)

        best_ece = float("inf")
        best_corr = -1.0
        best_acc = 0.0
        final_loss = 0.0

        iterator = range(num_epochs)
        if verbose:
            iterator = tqdm(iterator, desc=self.strength_mode, leave=False)

        for epoch in iterator:
            self.model.reset_memory(self.config.training.batch_size)
            final_loss = self.train_epoch()

            if (epoch + 1) % self.config.training.eval_every == 0 or epoch == num_epochs - 1:
                metrics = self.evaluate()
                if metrics["ece"] < best_ece:
                    best_ece = metrics["ece"]
                if metrics["correlation"] > best_corr:
                    best_corr = metrics["correlation"]
                if metrics["accuracy"] > best_acc:
                    best_acc = metrics["accuracy"]

        self.restore_strengths()

        return TestResult(
            name=self.strength_mode,
            ece=best_ece,
            correlation=best_corr,
            accuracy=best_acc,
            final_loss=final_loss,
        )


def run_single_seed(seed: int, config, device: str, verbose: bool = True) -> Dict[str, TestResult]:
    """Run all conditions for a single seed."""

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create data (same for all conditions)
    train_loader, eval_loader, dataset = create_data_loaders(
        num_pairs=config.training.num_key_value_pairs,
        key_length=config.training.key_length,
        value_length=config.training.value_length,
        vocab_size=config.model.vocab_size,
        batch_size=config.training.batch_size,
        seed=seed,
    )

    results = {}

    # Test 1: FADE with computed strengths (original)
    torch.manual_seed(seed)
    model = FADEModel(config)
    trainer = TestTrainer(
        model, config, train_loader, eval_loader, device,
        strength_mode="computed",
        use_calibration_loss=True,
        use_degradation=True,
    )
    results["computed"] = trainer.train(config.training.num_epochs, verbose)

    # Test 2: FADE with random strengths
    torch.manual_seed(seed)
    model = FADEModel(config)
    trainer = TestTrainer(
        model, config, train_loader, eval_loader, device,
        strength_mode="random",
        use_calibration_loss=True,
        use_degradation=True,
    )
    results["random"] = trainer.train(config.training.num_epochs, verbose)

    # Test 3: FADE with constant strengths
    torch.manual_seed(seed)
    model = FADEModel(config)
    trainer = TestTrainer(
        model, config, train_loader, eval_loader, device,
        strength_mode="constant",
        use_calibration_loss=True,
        use_degradation=True,
    )
    results["constant"] = trainer.train(config.training.num_epochs, verbose)

    # Test 4: Baseline (no degradation, no calibration loss)
    torch.manual_seed(seed)
    model = FADEModel(config)
    trainer = TestTrainer(
        model, config, train_loader, eval_loader, device,
        strength_mode="computed",  # doesn't matter since degradation is off
        use_calibration_loss=False,
        use_degradation=False,
    )
    results["baseline"] = trainer.train(config.training.num_epochs, verbose)

    # Test 5: Calibration loss only (no degradation) - isolates calibration loss effect
    torch.manual_seed(seed)
    model = FADEModel(config)
    trainer = TestTrainer(
        model, config, train_loader, eval_loader, device,
        strength_mode="computed",
        use_calibration_loss=True,
        use_degradation=False,
    )
    results["cal_loss_only"] = trainer.train(config.training.num_epochs, verbose)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test (fewer epochs, 1 seed)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to run")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per run")
    args = parser.parse_args()

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Config
    config = get_default_config()
    config.device = device

    if args.quick:
        config.training.num_epochs = 30
        config.training.eval_every = 10
        num_seeds = 1
        print("\n*** QUICK MODE - Use full mode for reliable results ***\n")
    else:
        config.training.num_epochs = args.epochs
        config.training.eval_every = 20
        num_seeds = args.seeds

    # Use larger model for more signal
    config.model.d_model = 128
    config.model.n_layers = 2
    config.model.n_heads = 4
    config.training.num_key_value_pairs = 100
    config.training.batch_size = 32

    print("=" * 70)
    print("DEFINITIVE FADE TEST")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Seeds: {num_seeds}")
    print(f"  Model dim: {config.model.d_model}")
    print(f"  Layers: {config.model.n_layers}")
    print(f"  KV pairs: {config.training.num_key_value_pairs}")
    print()

    # Run tests
    all_results: Dict[str, List[TestResult]] = {
        "computed": [], "random": [], "constant": [], "baseline": [], "cal_loss_only": []
    }

    start_time = time.time()

    for seed_idx in range(num_seeds):
        seed = 42 + seed_idx * 100
        print(f"\n--- Seed {seed_idx + 1}/{num_seeds} (seed={seed}) ---")

        seed_results = run_single_seed(seed, config, device, verbose=True)

        for name, result in seed_results.items():
            all_results[name].append(result)
            print(f"  {name}: ECE={result.ece:.4f}, Corr={result.correlation:.4f}, Acc={result.accuracy:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)

    def stats(values):
        arr = np.array(values)
        return arr.mean(), arr.std()

    print(f"\n{'Method':<15} {'ECE':<20} {'Fuzz-Err Corr':<20} {'Accuracy':<15}")
    print("-" * 70)

    summary = {}
    for name in ["computed", "random", "constant", "cal_loss_only", "baseline"]:
        results = all_results[name]
        ece_mean, ece_std = stats([r.ece for r in results])
        corr_mean, corr_std = stats([r.correlation for r in results])
        acc_mean, acc_std = stats([r.accuracy for r in results])

        summary[name] = {
            "ece_mean": ece_mean, "ece_std": ece_std,
            "corr_mean": corr_mean, "corr_std": corr_std,
            "acc_mean": acc_mean, "acc_std": acc_std,
        }

        print(f"{name:<15} {ece_mean:.4f} +/- {ece_std:.4f}    {corr_mean:.4f} +/- {corr_std:.4f}    {acc_mean:.4f} +/- {acc_std:.4f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    computed_ece = summary["computed"]["ece_mean"]
    random_ece = summary["random"]["ece_mean"]
    constant_ece = summary["constant"]["ece_mean"]
    baseline_ece = summary["baseline"]["ece_mean"]
    cal_only_ece = summary["cal_loss_only"]["ece_mean"]

    computed_corr = summary["computed"]["corr_mean"]
    random_corr = summary["random"]["corr_mean"]

    print("\n1. DEGRADATION EFFECT (comparing to baseline):")
    for name in ["computed", "random", "constant", "cal_loss_only"]:
        if baseline_ece > 0:
            improvement = (baseline_ece - summary[name]["ece_mean"]) / baseline_ece * 100
            print(f"   {name}: {improvement:+.1f}% ECE improvement over baseline")

    print("\n2. MECHANISM vs REGULARIZATION:")
    if abs(computed_ece - random_ece) < 0.02:
        print("   -> Computed and random strengths give SIMILAR ECE")
        print("   -> ECE improvement is likely from NOISE REGULARIZATION, not targeting")
    else:
        better = "computed" if computed_ece < random_ece else "random"
        print(f"   -> {better} strengths give meaningfully better ECE")
        if better == "computed":
            print("   -> The strength computation mechanism DOES help calibration")
        else:
            print("   -> Surprisingly, random is better - investigate further")

    print("\n3. CALIBRATION LOSS EFFECT:")
    if abs(cal_only_ece - computed_ece) < 0.02:
        print("   -> Calibration loss alone achieves similar ECE to full FADE")
        print("   -> Degradation adds little beyond the calibration loss term")
    else:
        print("   -> Degradation provides benefit beyond calibration loss alone")

    print("\n4. FUZZ-ERROR CORRELATION:")
    if computed_corr > random_corr + 0.2:
        print(f"   -> Computed ({computed_corr:.3f}) >> Random ({random_corr:.3f})")
        print("   -> The mechanism DOES create meaningful uncertainty signals")
        print("   -> High correlation is useful for retrieval/flagging even if ECE is similar")
    else:
        print(f"   -> Computed ({computed_corr:.3f}) ~= Random ({random_corr:.3f})")
        print("   -> No meaningful difference in fuzz-error correlation")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    mechanism_helps_ece = computed_ece < random_ece * 0.9
    mechanism_helps_corr = computed_corr > random_corr + 0.2
    degradation_helps = computed_ece < cal_only_ece * 0.9

    if mechanism_helps_ece and degradation_helps:
        print("\n[STRONG SUCCESS] FADE's targeting mechanism provides real benefit")
        print("The computed strengths outperform random/constant noise for calibration.")
    elif mechanism_helps_corr and not mechanism_helps_ece:
        print("\n[PARTIAL SUCCESS] FADE creates useful uncertainty signals")
        print("The mechanism helps fuzz-error correlation but ECE improvement")
        print("is primarily from noise regularization + calibration loss.")
    elif not mechanism_helps_ece and not degradation_helps:
        print("\n[CALIBRATION LOSS DOMINATES] The explicit calibration loss term")
        print("is doing most of the work. Degradation provides regularization")
        print("but targeting doesn't matter much.")
    else:
        print("\n[INCONCLUSIVE] Results are mixed - may need more epochs or different task.")

    # Save results
    output_file = Path(__file__).parent / "definitive_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "config": {
                "epochs": config.training.num_epochs,
                "seeds": num_seeds,
                "d_model": config.model.d_model,
                "n_layers": config.model.n_layers,
            },
            "summary": summary,
            "raw_results": {
                name: [{"ece": r.ece, "corr": r.correlation, "acc": r.accuracy} for r in results]
                for name, results in all_results.items()
            }
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
