"""
Evaluation Metrics for FADE.

Includes:
- Expected Calibration Error (ECE)
- Fuzziness-Error Correlation
- Retrieval Precision/Recall
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class CalibrationBin:
    """A single bin for calibration computation."""
    confidence_sum: float = 0.0
    accuracy_sum: float = 0.0
    count: int = 0

    @property
    def avg_confidence(self) -> float:
        return self.confidence_sum / max(self.count, 1)

    @property
    def avg_accuracy(self) -> float:
        return self.accuracy_sum / max(self.count, 1)


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
        self.bins = [CalibrationBin() for _ in range(self.n_bins)]
        self.total_count = 0

    def update(
        self,
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
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

        confidences = confidences.view(-1).cpu().numpy()
        predictions = predictions.view(-1).cpu().numpy()
        targets = targets.view(-1).cpu().numpy()

        correct = (predictions == targets).astype(float)

        for conf, acc in zip(confidences, correct):
            bin_idx = min(int(conf * self.n_bins), self.n_bins - 1)
            self.bins[bin_idx].confidence_sum += conf
            self.bins[bin_idx].accuracy_sum += acc
            self.bins[bin_idx].count += 1
            self.total_count += 1

    def compute(self) -> Dict[str, float]:
        """
        Compute ECE and related metrics.

        Returns:
            Dict with ECE, MCE, and per-bin stats
        """
        ece = 0.0
        mce = 0.0  # Maximum Calibration Error

        for bin_data in self.bins:
            if bin_data.count > 0:
                gap = abs(bin_data.avg_accuracy - bin_data.avg_confidence)
                weight = bin_data.count / max(self.total_count, 1)
                ece += gap * weight
                mce = max(mce, gap)

        return {
            "ece": ece,
            "mce": mce,
            "total_samples": self.total_count,
        }


class FuzzinessErrorCorrelation:
    """
    Tracks correlation between fuzziness and prediction errors.

    This is the key metric for FADE: high fuzziness should predict errors.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulator."""
        self.fuzziness_values: List[float] = []
        self.error_values: List[float] = []

    def update(
        self,
        fuzziness: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
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

        fuzziness = fuzziness.view(-1).detach().cpu().numpy()
        predictions = predictions.view(-1).cpu().numpy()
        targets = targets.view(-1).cpu().numpy()

        errors = (predictions != targets).astype(float)

        self.fuzziness_values.extend(fuzziness.tolist())
        self.error_values.extend(errors.tolist())

    def compute(self) -> Dict[str, float]:
        """
        Compute correlation metrics.

        Returns:
            Dict with correlation and related stats
        """
        if len(self.fuzziness_values) < 2:
            return {"correlation": 0.0, "samples": 0}

        fuzz_arr = np.array(self.fuzziness_values)
        err_arr = np.array(self.error_values)

        # Pearson correlation
        correlation = np.corrcoef(fuzz_arr, err_arr)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        # Compute AUC-like metric: can fuzziness separate errors from correct?
        error_mask = err_arr > 0.5
        correct_mask = ~error_mask

        if error_mask.sum() > 0 and correct_mask.sum() > 0:
            avg_fuzz_errors = fuzz_arr[error_mask].mean()
            avg_fuzz_correct = fuzz_arr[correct_mask].mean()
            separation = avg_fuzz_errors - avg_fuzz_correct
        else:
            separation = 0.0

        return {
            "correlation": correlation,
            "separation": separation,  # Positive = fuzziness higher for errors (good!)
            "avg_fuzziness": fuzz_arr.mean(),
            "avg_error_rate": err_arr.mean(),
            "samples": len(self.fuzziness_values),
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
        mask: Optional[torch.Tensor] = None,
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

    def compute(self) -> Dict[str, float]:
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
        losses: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
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

    def compute_all(self) -> Dict[str, float]:
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
