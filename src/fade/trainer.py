"""
Training Loop for FADE.

Implements the three-phase training procedure:
1. Phase 1: Learn degradation dynamics
2. Phase 2: Calibrate fuzziness to errors
3. Phase 3: Integrate with retrieval
"""

from __future__ import annotations

__all__ = [
    "TrainingState",
    "FADETrainer",
    "BaselineTrainer",
    "PathTraversalError",
    "DEFAULT_CHECKPOINT_DIR",
]

import logging
import os
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import FADEConfig, TrainingConfig
from .fade_model import FADEModel
from .metrics import MetricsTracker


# Default checkpoint directory (relative to current working directory)
DEFAULT_CHECKPOINT_DIR = "checkpoints"


class PathTraversalError(Exception):
    """Raised when a path traversal attempt is detected."""
    pass


def _validate_checkpoint_path(path: str, base_dir: str | None = None) -> str:
    """
    Validate that a checkpoint path is within the allowed directory.

    Args:
        path: The path to validate (can be relative or absolute)
        base_dir: The allowed base directory. If None, uses DEFAULT_CHECKPOINT_DIR
                  relative to current working directory.

    Returns:
        The resolved absolute path if valid.

    Raises:
        PathTraversalError: If the path would escape the allowed directory.
        ValueError: If the path is empty or invalid.
    """
    if not path:
        raise ValueError("Checkpoint path cannot be empty")

    # Determine the base directory
    if base_dir is None:
        base_dir = os.path.join(os.getcwd(), DEFAULT_CHECKPOINT_DIR)

    # Resolve both paths to their real absolute paths
    # os.path.realpath resolves symlinks and normalizes the path
    resolved_base = os.path.realpath(base_dir)

    # If path is relative, join it with the base directory
    if not os.path.isabs(path):
        full_path = os.path.join(resolved_base, path)
    else:
        full_path = path

    resolved_path = os.path.realpath(full_path)

    # Check that the resolved path starts with the base directory
    # Use os.path.commonpath for a more robust comparison
    try:
        common = os.path.commonpath([resolved_base, resolved_path])
        if common != resolved_base:
            raise PathTraversalError(
                f"Path traversal detected: '{path}' resolves to '{resolved_path}' "
                f"which is outside the allowed directory '{resolved_base}'"
            )
    except ValueError:
        # commonpath raises ValueError if paths are on different drives (Windows)
        raise PathTraversalError(
            f"Path traversal detected: '{path}' is on a different drive "
            f"than the allowed directory '{resolved_base}'"
        )

    return resolved_path


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
        dataset: Dataset,
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
        self.on_epoch_end: Callable | None = None
        self.on_eval_end: Callable | None = None

    def train_epoch(self) -> dict[str, float]:
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip_norm)

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
    def evaluate(self) -> dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        self.eval_metrics.reset()

        for batch in tqdm(self.eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)

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

    def train(self, num_epochs: int | None = None) -> dict[str, list]:
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

        logger.info("Starting training for %d epochs", num_epochs)
        logger.info("Model parameters: %s", f"{self.model.count_parameters():,}")

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

                # Log summary
                logger.info(
                    "Epoch %d/%d (%.1fs) - Train Loss: %.4f, Eval Loss: %.4f, "
                    "ECE: %.4f, Fuzz-Error Correlation: %.4f, Fuzz-Error Separation: %.4f, "
                    "Retrieval Precision: %.4f",
                    epoch + 1,
                    num_epochs,
                    time.time() - epoch_start,
                    train_metrics.get('loss/total', 0),
                    eval_metrics.get('loss/total', 0),
                    eval_metrics.get('ece/ece', 0),
                    eval_metrics.get('fuzziness/correlation', 0),
                    eval_metrics.get('fuzziness/separation', 0),
                    eval_metrics.get('retrieval/precision', 0),
                )

                # Track best
                ece = eval_metrics.get("ece/ece", float("inf"))
                if ece < self.state.best_ece:
                    self.state.best_ece = ece
                    logger.info("New best ECE: %.4f", ece)

                corr = eval_metrics.get("fuzziness/correlation", -1)
                if corr > self.state.best_correlation:
                    self.state.best_correlation = corr
                    logger.info("New best correlation: %.4f", corr)

                if self.on_eval_end:
                    self.on_eval_end(eval_metrics)

            # Step scheduler
            self.scheduler.step()

            if self.on_epoch_end:
                self.on_epoch_end(train_metrics)

        logger.info("Training complete!")
        logger.info("Best ECE: %.4f", self.state.best_ece)
        logger.info("Best Fuzz-Error Correlation: %.4f", self.state.best_correlation)

        return history

    def save_checkpoint(self, path: str, checkpoint_dir: str | None = None):
        """Save model checkpoint.

        Args:
            path: Path to save the checkpoint. Can be relative (to checkpoint_dir)
                  or absolute (must be within checkpoint_dir).
            checkpoint_dir: Base directory for checkpoints. If None, uses
                           DEFAULT_CHECKPOINT_DIR relative to cwd.

        Raises:
            PathTraversalError: If the path would escape the allowed directory.
            ValueError: If the path is empty or invalid.
        """
        # Validate path to prevent path traversal attacks
        validated_path = _validate_checkpoint_path(path, checkpoint_dir)

        # Ensure the parent directory exists
        parent_dir = os.path.dirname(validated_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "state": self.state,
            "config": self.config,
        }, validated_path)
        logger.info("Saved checkpoint to %s", validated_path)

    def load_checkpoint(self, path: str, checkpoint_dir: str | None = None):
        """Load model checkpoint.

        Args:
            path: Path to load the checkpoint from. Can be relative (to checkpoint_dir)
                  or absolute (must be within checkpoint_dir).
            checkpoint_dir: Base directory for checkpoints. If None, uses
                           DEFAULT_CHECKPOINT_DIR relative to cwd.

        Raises:
            PathTraversalError: If the path would escape the allowed directory.
            ValueError: If the path is empty or invalid.
            FileNotFoundError: If the checkpoint file does not exist.

        Security Note: This method uses pickle-based deserialization which can
        execute arbitrary code. Only load checkpoints from trusted sources.
        """
        # Validate path to prevent path traversal attacks
        validated_path = _validate_checkpoint_path(path, checkpoint_dir)

        # Check file exists
        if not os.path.exists(validated_path):
            raise FileNotFoundError(f"Checkpoint file not found: {validated_path}")

        # Security warning for untrusted sources
        warnings.warn(
            "Loading checkpoints uses pickle deserialization which can execute "
            "arbitrary code. Only load checkpoints from trusted sources.",
            UserWarning,
            stacklevel=2,
        )

        # SECURITY: weights_only=False is required because TrainingState is a
        # dataclass that cannot be loaded with weights_only=True. This is a
        # known security consideration (CWE-502). Only load checkpoints from
        # trusted sources to mitigate arbitrary code execution risks.
        checkpoint = torch.load(validated_path, map_location=self.device, weights_only=False)

        # Validate checkpoint structure before using it
        expected_keys = {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "state", "config"}
        actual_keys = set(checkpoint.keys())
        if not expected_keys.issubset(actual_keys):
            missing_keys = expected_keys - actual_keys
            raise ValueError(
                f"Invalid checkpoint format. Missing required keys: {missing_keys}. "
                f"Expected keys: {expected_keys}, got: {actual_keys}"
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.state = checkpoint["state"]
        logger.info("Loaded checkpoint from %s", validated_path)


class BaselineTrainer(FADETrainer):
    """
    Trainer for baseline model (no degradation).

    Used for comparison against FADE.
    """

    def train_epoch(self) -> dict[str, float]:
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip_norm)
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
