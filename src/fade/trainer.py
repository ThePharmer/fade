"""
Training Loop for FADE.

Implements the three-phase training procedure:
1. Phase 1: Learn degradation dynamics
2. Phase 2: Calibrate fuzziness to errors
3. Phase 3: Integrate with retrieval
"""

import time
from typing import Dict, Optional, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import FADEConfig, TrainingConfig
from .fade_model import FADEModel
from .metrics import MetricsTracker
from .data import KeyValueMemorizationDataset


@dataclass
class TrainingState:
    """Tracks training state."""
    epoch: int = 0
    global_step: int = 0
    best_ece: float = float("inf")
    best_correlation: float = -1.0


class FADETrainer:
    """
    Trainer for FADE model.

    Handles:
    - Training loop with time simulation
    - Evaluation with metrics
    - Checkpointing
    """

    def __init__(
        self,
        model: FADEModel,
        config: FADEConfig,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        dataset: KeyValueMemorizationDataset,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.dataset = dataset
        self.device = device

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.num_epochs,
        )

        # Metrics
        self.train_metrics = MetricsTracker()
        self.eval_metrics = MetricsTracker()

        # State
        self.state = TrainingState()

        # Callbacks
        self.on_epoch_end: Optional[Callable] = None
        self.on_eval_end: Optional[Callable] = None

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()

        # Simulate time passing during epoch
        time_per_batch = self.config.training.time_steps_per_epoch / len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.state.epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Advance time (simulates memory decay)
            self.model.advance_time(time_per_batch)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(input_ids, apply_degradation=True, return_fuzziness=True)

            # Compute loss
            losses = self.model.compute_loss(
                output["logits"],
                targets,
                output.get("fuzziness"),
                mask,
            )

            # Backward pass
            losses["total_loss"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            # Update metrics
            with torch.no_grad():
                self.train_metrics.update(
                    output["logits"],
                    targets,
                    output.get("fuzziness", torch.zeros_like(mask)),
                    output.get("should_retrieve", torch.zeros_like(mask).bool()),
                    losses,
                    mask,
                )

            # Update progress bar
            if batch_idx % self.config.training.log_every == 0:
                pbar.set_postfix({
                    "loss": f"{losses['total_loss'].item():.4f}",
                    "pred_loss": f"{losses['prediction_loss'].item():.4f}",
                })

            self.state.global_step += 1

        # Compute epoch metrics
        return self.train_metrics.compute_all()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        self.eval_metrics.reset()

        for batch in tqdm(self.eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Forward pass without degradation first (baseline)
            output_clean = self.model(input_ids, apply_degradation=False, return_fuzziness=True)

            # Forward pass with degradation
            output_degraded = self.model(input_ids, apply_degradation=True, return_fuzziness=True)

            # Compute losses
            losses = self.model.compute_loss(
                output_degraded["logits"],
                targets,
                output_degraded.get("fuzziness"),
                mask,
            )

            # Update metrics
            self.eval_metrics.update(
                output_degraded["logits"],
                targets,
                output_degraded.get("fuzziness", torch.zeros_like(mask)),
                output_degraded.get("should_retrieve", torch.zeros_like(mask).bool()),
                losses,
                mask,
            )

        return self.eval_metrics.compute_all()

    def train(self, num_epochs: Optional[int] = None) -> Dict[str, list]:
        """
        Full training loop.

        Returns:
            Dict with training history
        """
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs

        history = {
            "train_loss": [],
            "eval_loss": [],
            "ece": [],
            "fuzziness_correlation": [],
        }

        print(f"Starting training for {num_epochs} epochs")
        print(f"Model parameters: {self.model.count_parameters():,}")

        for epoch in range(num_epochs):
            self.state.epoch = epoch
            epoch_start = time.time()

            # Reset memory at start of epoch
            self.model.reset_memory(self.config.training.batch_size)

            # Train
            train_metrics = self.train_epoch()
            history["train_loss"].append(train_metrics.get("loss/total", 0))

            # Evaluate
            if (epoch + 1) % self.config.training.eval_every == 0:
                eval_metrics = self.evaluate()
                history["eval_loss"].append(eval_metrics.get("loss/total", 0))
                history["ece"].append(eval_metrics.get("ece/ece", 0))
                history["fuzziness_correlation"].append(
                    eval_metrics.get("fuzziness/correlation", 0)
                )

                # Print summary
                print(f"\nEpoch {epoch + 1}/{num_epochs} ({time.time() - epoch_start:.1f}s)")
                print(f"  Train Loss: {train_metrics.get('loss/total', 0):.4f}")
                print(f"  Eval Loss: {eval_metrics.get('loss/total', 0):.4f}")
                print(f"  ECE: {eval_metrics.get('ece/ece', 0):.4f}")
                print(f"  Fuzz-Error Correlation: {eval_metrics.get('fuzziness/correlation', 0):.4f}")
                print(f"  Fuzz-Error Separation: {eval_metrics.get('fuzziness/separation', 0):.4f}")
                print(f"  Retrieval Precision: {eval_metrics.get('retrieval/precision', 0):.4f}")

                # Track best
                ece = eval_metrics.get("ece/ece", float("inf"))
                if ece < self.state.best_ece:
                    self.state.best_ece = ece
                    print(f"  New best ECE!")

                corr = eval_metrics.get("fuzziness/correlation", -1)
                if corr > self.state.best_correlation:
                    self.state.best_correlation = corr
                    print(f"  New best correlation!")

                if self.on_eval_end:
                    self.on_eval_end(eval_metrics)

            # Step scheduler
            self.scheduler.step()

            if self.on_epoch_end:
                self.on_epoch_end(train_metrics)

        print(f"\nTraining complete!")
        print(f"Best ECE: {self.state.best_ece:.4f}")
        print(f"Best Fuzz-Error Correlation: {self.state.best_correlation:.4f}")

        return history

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "state": self.state,
            "config": self.config,
        }, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.state = checkpoint["state"]
        print(f"Loaded checkpoint from {path}")


class BaselineTrainer(FADETrainer):
    """
    Trainer for baseline model (no degradation).

    Used for comparison against FADE.
    """

    def train_epoch(self) -> Dict[str, float]:
        """Train without degradation."""
        self.model.train()
        self.train_metrics.reset()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.state.epoch + 1} (baseline)")

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Forward pass WITHOUT degradation
            self.optimizer.zero_grad()
            output = self.model(input_ids, apply_degradation=False, return_fuzziness=True)

            # Only prediction loss (no calibration)
            losses = {"total_loss": nn.functional.cross_entropy(
                output["logits"].view(-1, output["logits"].size(-1)),
                targets.view(-1),
            )}
            losses["prediction_loss"] = losses["total_loss"]

            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Update metrics
            with torch.no_grad():
                self.train_metrics.update(
                    output["logits"],
                    targets,
                    output.get("fuzziness", torch.zeros_like(mask)),
                    torch.zeros_like(mask).bool(),
                    losses,
                    mask,
                )

            if batch_idx % self.config.training.log_every == 0:
                pbar.set_postfix({"loss": f"{losses['total_loss'].item():.4f}"})

            self.state.global_step += 1

        return self.train_metrics.compute_all()
