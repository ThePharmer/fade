"""
Evaluation Metrics for FADE.

Includes:
- Expected Calibration Error (ECE)
- Fuzziness-Error Correlation
- Retrieval Precision/Recall
"""

from __future__ import annotations

__all__ = [
    "ExpectedCalibrationError",
    "FuzzinessErrorCorrelation",
    "RetrievalMetrics",
    "MetricsTracker",
]

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field


class ExpectedCalibrationError:
    """
    Computes Expected Calibration Error (ECE).

    ECE measures how well confidence scores align with actual accuracy.
    A well-calibrated model should have confidence ≈ accuracy.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.reset()

    def reset(self):
        """Reset accumulator."""
        # Vectorized bin accumulators instead of list of CalibrationBin objects
        self.bin_confidence_sum = torch.zeros(self.n_bins, dtype=torch.float64)
        self.bin_accuracy_sum = torch.zeros(self.n_bins, dtype=torch.float64)
        self.bin_count = torch.zeros(self.n_bins, dtype=torch.int64)
        self.total_count = 0

    def update(
        self,
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """
        Update bins with new predictions.

        Args:
            confidences: Confidence scores [batch, seq_len]
            predictions: Predicted tokens [batch, seq_len]
            targets: Target tokens [batch, seq_len]
            mask: Valid positions mask [batch, seq_len]
        """
        if mask is not None:
            confidences = confidences[mask.bool()]
            predictions = predictions[mask.bool()]
            targets = targets[mask.bool()]

        # Flatten and move to CPU as tensors (not numpy) for vectorized binning
        confidences = confidences.view(-1).detach().cpu().to(torch.float64)
        predictions = predictions.view(-1).cpu()
        targets = targets.view(-1).cpu()

        correct = (predictions == targets).to(torch.float64)

        # Vectorized binning: compute bin indices for all elements at once
        bin_indices = (confidences * self.n_bins).long().clamp(0, self.n_bins - 1)

        # Use scatter_add_ for vectorized accumulation
        self.bin_confidence_sum.scatter_add_(0, bin_indices, confidences)
        self.bin_accuracy_sum.scatter_add_(0, bin_indices, correct)
        self.bin_count.scatter_add_(0, bin_indices, torch.ones_like(bin_indices))
        self.total_count += confidences.numel()

    def compute(self) -> dict[str, float]:
        """
        Compute ECE and related metrics.

        Returns:
            Dict with ECE, MCE, and per-bin stats
        """
        # Vectorized computation across all bins
        non_empty_mask = self.bin_count > 0

        if not non_empty_mask.any():
            return {"ece": 0.0, "mce": 0.0, "total_samples": 0}

        # Compute average confidence and accuracy per bin (only for non-empty bins)
        avg_confidence = torch.zeros(self.n_bins, dtype=torch.float64)
        avg_accuracy = torch.zeros(self.n_bins, dtype=torch.float64)

        avg_confidence[non_empty_mask] = (
            self.bin_confidence_sum[non_empty_mask] / self.bin_count[non_empty_mask].to(torch.float64)
        )
        avg_accuracy[non_empty_mask] = (
            self.bin_accuracy_sum[non_empty_mask] / self.bin_count[non_empty_mask].to(torch.float64)
        )

        # Compute calibration gaps
        gaps = torch.abs(avg_accuracy - avg_confidence)
        weights = self.bin_count.to(torch.float64) / max(self.total_count, 1)

        # ECE is weighted average of gaps, MCE is maximum gap
        ece = (gaps * weights).sum().item()
        mce = gaps[non_empty_mask].max().item() if non_empty_mask.any() else 0.0

        return {
            "ece": ece,
            "mce": mce,
            "total_samples": self.total_count,
        }


class FuzzinessErrorCorrelation:
    """
    Tracks correlation between fuzziness and prediction errors.

    This is the key metric for FADE: high fuzziness should predict errors.

    Uses Welford's online algorithm for numerically stable running statistics,
    avoiding unbounded memory growth from storing all values.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulator."""
        # Running statistics using Welford's online algorithm
        # For correlation, we track: n, sum_x, sum_y, sum_xx, sum_yy, sum_xy
        self.n = 0  # Count
        self.sum_x = 0.0  # Sum of fuzziness values
        self.sum_y = 0.0  # Sum of error values
        self.sum_xx = 0.0  # Sum of fuzziness^2
        self.sum_yy = 0.0  # Sum of error^2
        self.sum_xy = 0.0  # Sum of fuzziness * error

        # For separation metric: track fuzziness sums for errors vs correct
        self.n_errors = 0
        self.n_correct = 0
        self.sum_fuzz_errors = 0.0
        self.sum_fuzz_correct = 0.0

    def update(
        self,
        fuzziness: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """
        Update with new batch.

        Args:
            fuzziness: Fuzziness scores [batch, seq_len]
            predictions: Predicted tokens [batch, seq_len]
            targets: Target tokens [batch, seq_len]
            mask: Valid positions mask
        """
        if mask is not None:
            fuzziness = fuzziness[mask.bool()]
            predictions = predictions[mask.bool()]
            targets = targets[mask.bool()]

        # Flatten and move to CPU as float64 tensors for numerical stability
        fuzziness = fuzziness.view(-1).detach().cpu().to(torch.float64)
        predictions = predictions.view(-1).cpu()
        targets = targets.view(-1).cpu()

        errors = (predictions != targets).to(torch.float64)

        # Vectorized running statistics update
        batch_n = fuzziness.numel()
        if batch_n == 0:
            return

        # Update running sums for correlation computation
        self.n += batch_n
        self.sum_x += fuzziness.sum().item()
        self.sum_y += errors.sum().item()
        self.sum_xx += (fuzziness * fuzziness).sum().item()
        self.sum_yy += (errors * errors).sum().item()
        self.sum_xy += (fuzziness * errors).sum().item()

        # Update separation statistics
        error_mask = errors > 0.5
        correct_mask = ~error_mask

        self.n_errors += error_mask.sum().item()
        self.n_correct += correct_mask.sum().item()
        self.sum_fuzz_errors += fuzziness[error_mask].sum().item()
        self.sum_fuzz_correct += fuzziness[correct_mask].sum().item()

    def compute(self) -> dict[str, float]:
        """
        Compute correlation metrics from running statistics.

        Returns:
            Dict with correlation and related stats
        """
        if self.n < 2:
            return {"correlation": 0.0, "samples": 0}

        # Compute Pearson correlation from running statistics
        # r = (n*sum_xy - sum_x*sum_y) / sqrt((n*sum_xx - sum_x^2) * (n*sum_yy - sum_y^2))
        n = self.n
        numerator = n * self.sum_xy - self.sum_x * self.sum_y
        var_x = n * self.sum_xx - self.sum_x * self.sum_x
        var_y = n * self.sum_yy - self.sum_y * self.sum_y

        if var_x > 0 and var_y > 0:
            correlation = numerator / np.sqrt(var_x * var_y)
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        # Compute separation metric
        if self.n_errors > 0 and self.n_correct > 0:
            avg_fuzz_errors = self.sum_fuzz_errors / self.n_errors
            avg_fuzz_correct = self.sum_fuzz_correct / self.n_correct
            separation = avg_fuzz_errors - avg_fuzz_correct
        else:
            separation = 0.0

        # Compute averages
        avg_fuzziness = self.sum_x / n if n > 0 else 0.0
        avg_error_rate = self.sum_y / n if n > 0 else 0.0

        return {
            "correlation": correlation,
            "separation": separation,  # Positive = fuzziness higher for errors (good!)
            "avg_fuzziness": avg_fuzziness,
            "avg_error_rate": avg_error_rate,
            "samples": self.n,
        }


class RetrievalMetrics:
    """
    Tracks retrieval precision and recall.

    Precision: When we retrieve, was it actually needed?
    Recall: When retrieval was needed, did we retrieve?
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.true_positives = 0  # Retrieved and was needed
        self.false_positives = 0  # Retrieved but wasn't needed
        self.false_negatives = 0  # Didn't retrieve but was needed
        self.true_negatives = 0  # Didn't retrieve and wasn't needed

    def update(
        self,
        retrieved: torch.Tensor,  # Boolean: did we retrieve?
        was_error: torch.Tensor,  # Boolean: was the prediction wrong?
        mask: torch.Tensor | None = None,
    ):
        """
        Update metrics.

        Args:
            retrieved: Whether retrieval was triggered [batch, seq_len]
            was_error: Whether prediction was wrong [batch, seq_len]
            mask: Valid positions mask
        """
        if mask is not None:
            retrieved = retrieved[mask.bool()]
            was_error = was_error[mask.bool()]

        retrieved = retrieved.view(-1).bool()
        was_error = was_error.view(-1).bool()

        self.true_positives += (retrieved & was_error).sum().item()
        self.false_positives += (retrieved & ~was_error).sum().item()
        self.false_negatives += (~retrieved & was_error).sum().item()
        self.true_negatives += (~retrieved & ~was_error).sum().item()

    def compute(self) -> dict[str, float]:
        """Compute precision, recall, F1."""
        precision = self.true_positives / max(self.true_positives + self.false_positives, 1)
        recall = self.true_positives / max(self.true_positives + self.false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        # Retrieval rate
        total = self.true_positives + self.false_positives + self.false_negatives + self.true_negatives
        retrieval_rate = (self.true_positives + self.false_positives) / max(total, 1)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "retrieval_rate": retrieval_rate,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
        }


@dataclass
class MetricsTracker:
    """Combined metrics tracker for FADE evaluation."""

    ece: ExpectedCalibrationError = field(default_factory=lambda: ExpectedCalibrationError())
    fuzziness_correlation: FuzzinessErrorCorrelation = field(default_factory=lambda: FuzzinessErrorCorrelation())
    retrieval: RetrievalMetrics = field(default_factory=lambda: RetrievalMetrics())

    # Running averages
    total_loss: float = 0.0
    prediction_loss: float = 0.0
    calibration_loss: float = 0.0
    num_batches: int = 0

    def reset(self):
        """Reset all metrics."""
        self.ece.reset()
        self.fuzziness_correlation.reset()
        self.retrieval.reset()
        self.total_loss = 0.0
        self.prediction_loss = 0.0
        self.calibration_loss = 0.0
        self.num_batches = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        fuzziness: torch.Tensor,
        should_retrieve: torch.Tensor,
        losses: dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
    ):
        """Update all metrics with a batch."""
        # Get predictions and confidences
        probs = F.softmax(logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)
        errors = (predictions != targets)

        # Update component metrics
        self.ece.update(confidences, predictions, targets, mask)
        self.fuzziness_correlation.update(fuzziness, predictions, targets, mask)
        self.retrieval.update(should_retrieve, errors, mask)

        # Update losses
        self.total_loss += losses.get("total_loss", torch.tensor(0.0)).item()
        self.prediction_loss += losses.get("prediction_loss", torch.tensor(0.0)).item()
        self.calibration_loss += losses.get("calibration_loss", torch.tensor(0.0)).item()
        self.num_batches += 1

    def compute_all(self) -> dict[str, float]:
        """Compute all metrics."""
        results = {}

        # ECE metrics
        ece_metrics = self.ece.compute()
        results.update({f"ece/{k}": v for k, v in ece_metrics.items()})

        # Fuzziness correlation
        fuzz_metrics = self.fuzziness_correlation.compute()
        results.update({f"fuzziness/{k}": v for k, v in fuzz_metrics.items()})

        # Retrieval metrics
        ret_metrics = self.retrieval.compute()
        results.update({f"retrieval/{k}": v for k, v in ret_metrics.items()})

        # Average losses
        if self.num_batches > 0:
            results["loss/total"] = self.total_loss / self.num_batches
            results["loss/prediction"] = self.prediction_loss / self.num_batches
            results["loss/calibration"] = self.calibration_loss / self.num_batches

        return results
