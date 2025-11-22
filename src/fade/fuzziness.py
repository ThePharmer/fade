"""
Fuzziness Detection Module for FADE.

Detects when memories are "fuzzy" (uncertain/degraded) and should
trigger retrieval from persistent storage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .config import FuzzinessConfig


class FuzzinessDetector(nn.Module):
    """
    Detects fuzziness (uncertainty) in memory representations.

    Uses three complementary signals:
    1. Attention entropy: High entropy = uncertain about what to attend to
    2. Reconstruction error: Poor reconstruction = degraded representation
    3. Activation variance: Unusual variance = unstable representation

    These signals are combined into a single fuzziness score.
    """

    def __init__(self, config: FuzzinessConfig, d_model: int):
        super().__init__()
        self.config = config
        self.d_model = d_model

        # Reconstruction network: tries to reconstruct original from degraded
        self.reconstruction_net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Learnable baseline statistics for variance comparison
        self.register_buffer("running_mean", torch.zeros(d_model))
        self.register_buffer("running_var", torch.ones(d_model))
        self.register_buffer("num_batches", torch.tensor(0.0))

        # Store component scores for analysis
        self.last_component_scores: Dict[str, torch.Tensor] = {}

    def compute_attention_entropy(
        self,
        attention_weights: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute entropy of attention distributions.

        High entropy = attention is spread out = uncertainty about relevance.

        Args:
            attention_weights: List of attention weights from each layer
                              Each tensor: [batch, heads, seq, seq]

        Returns:
            Entropy scores per position [batch, seq_len]
        """
        if not attention_weights or attention_weights[0] is None:
            return None

        # Stack all layers
        all_attn = torch.stack(attention_weights, dim=0)  # [layers, batch, heads, seq, seq]

        # Average across layers and heads
        avg_attn = all_attn.mean(dim=(0, 2))  # [batch, seq, seq]

        # Compute entropy for each query position
        # H = -sum(p * log(p))
        eps = 1e-10
        entropy = -torch.sum(avg_attn * torch.log(avg_attn + eps), dim=-1)  # [batch, seq]

        # Normalize by max possible entropy (uniform distribution)
        max_entropy = torch.log(torch.tensor(avg_attn.shape[-1], dtype=torch.float32, device=entropy.device))
        normalized_entropy = entropy / max_entropy

        return normalized_entropy

    def compute_reconstruction_error(
        self,
        hidden_states: torch.Tensor,
        original_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reconstruction error.

        If original embeddings are available, measure how well we can
        reconstruct them. Otherwise, use self-reconstruction.

        Args:
            hidden_states: Current hidden states [batch, seq_len, d_model]
            original_embeddings: Original embeddings if available

        Returns:
            Reconstruction error per position [batch, seq_len]
        """
        # Try to reconstruct
        reconstructed = self.reconstruction_net(hidden_states)

        # Target: original if available, otherwise hidden states themselves
        target = original_embeddings if original_embeddings is not None else hidden_states.detach()

        # MSE per position
        error = F.mse_loss(reconstructed, target, reduction="none").mean(dim=-1)

        # Normalize to [0, 1] range approximately
        error = torch.tanh(error)  # Squash large values

        return error

    def compute_activation_variance(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute activation variance compared to baseline.

        Unusual variance (too high or too low) indicates instability.

        Args:
            hidden_states: Hidden states [batch, seq_len, d_model]

        Returns:
            Variance anomaly score per position [batch, seq_len]
        """
        # Compute per-position variance across features
        pos_var = hidden_states.var(dim=-1)  # [batch, seq_len]
        pos_mean = hidden_states.mean(dim=-1)  # [batch, seq_len]

        # Update running statistics
        if self.training:
            batch_mean = hidden_states.mean(dim=(0, 1))
            batch_var = hidden_states.var(dim=(0, 1))

            # Exponential moving average
            momentum = 0.1
            self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
            self.running_var = (1 - momentum) * self.running_var + momentum * batch_var
            self.num_batches = self.num_batches + 1

        # Compare to expected variance
        expected_var = self.running_var.mean()
        var_ratio = pos_var / (expected_var + 1e-6)

        # Anomaly: deviation from expected (both too high and too low are suspicious)
        var_anomaly = torch.abs(torch.log(var_ratio + 1e-6))
        var_anomaly = torch.tanh(var_anomaly)  # Normalize to [0, 1)

        return var_anomaly

    def compute_fuzziness(
        self,
        hidden_states: torch.Tensor,
        attention_weights: Optional[List[torch.Tensor]] = None,
        original_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute overall fuzziness score.

        Args:
            hidden_states: Hidden states [batch, seq_len, d_model]
            attention_weights: Attention weights from transformer
            original_embeddings: Original embeddings for reconstruction

        Returns:
            Tuple of:
                - Fuzziness scores [batch, seq_len]
                - Dict of component scores for analysis
        """
        components = {}
        weights_sum = 0.0
        fuzziness = torch.zeros(hidden_states.shape[0], hidden_states.shape[1], device=hidden_states.device)

        # Attention entropy
        if self.config.use_attention_entropy and attention_weights is not None:
            entropy = self.compute_attention_entropy(attention_weights)
            if entropy is not None:
                components["attention_entropy"] = entropy
                fuzziness = fuzziness + self.config.entropy_weight * entropy
                weights_sum += self.config.entropy_weight

        # Reconstruction error
        if self.config.use_reconstruction_error:
            recon_error = self.compute_reconstruction_error(hidden_states, original_embeddings)
            components["reconstruction_error"] = recon_error
            fuzziness = fuzziness + self.config.reconstruction_weight * recon_error
            weights_sum += self.config.reconstruction_weight

        # Activation variance
        if self.config.use_activation_variance:
            var_anomaly = self.compute_activation_variance(hidden_states)
            components["activation_variance"] = var_anomaly
            fuzziness = fuzziness + self.config.variance_weight * var_anomaly
            weights_sum += self.config.variance_weight

        # Normalize
        if weights_sum > 0:
            fuzziness = fuzziness / weights_sum

        self.last_component_scores = components
        return fuzziness, components

    def should_retrieve(
        self,
        fuzziness: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Determine which positions should trigger retrieval.

        Args:
            fuzziness: Fuzziness scores [batch, seq_len]
            threshold: Override threshold (default: config.retrieval_threshold)

        Returns:
            Boolean mask [batch, seq_len] - True where retrieval is needed
        """
        if threshold is None:
            threshold = self.config.retrieval_threshold
        return fuzziness > threshold

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_weights: Optional[List[torch.Tensor]] = None,
        original_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            hidden_states: Hidden states [batch, seq_len, d_model]
            attention_weights: Attention weights from transformer
            original_embeddings: Original embeddings

        Returns:
            Tuple of (fuzziness_scores, should_retrieve_mask)
        """
        fuzziness, _ = self.compute_fuzziness(hidden_states, attention_weights, original_embeddings)
        retrieve_mask = self.should_retrieve(fuzziness)
        return fuzziness, retrieve_mask


class RetrievalDecision(nn.Module):
    """
    Makes final retrieval decisions based on fuzziness and context.

    Can learn to adjust threshold based on:
    - Task difficulty
    - Computational budget
    - Historical accuracy
    """

    def __init__(self, d_model: int, base_threshold: float = 0.6):
        super().__init__()
        self.base_threshold = base_threshold

        # Learned threshold adjustment
        self.threshold_net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        fuzziness: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decide whether to retrieve.

        Args:
            fuzziness: Fuzziness scores [batch, seq_len]
            context: Context representation [batch, d_model]

        Returns:
            Tuple of:
                - Retrieval decisions [batch, seq_len]
                - Adjusted threshold [batch, 1]
        """
        # Compute context-dependent threshold adjustment
        threshold_adjust = self.threshold_net(context)  # [batch, 1]
        adjusted_threshold = self.base_threshold * (0.5 + threshold_adjust)  # Range: [0.3, 0.9] * base

        # Make decisions
        decisions = fuzziness > adjusted_threshold

        return decisions, adjusted_threshold
