"""
Strength Tracking Module for FADE.

Tracks the "strength" of memories over time, implementing the core
degradation dynamics: memories decay unless they are accessed.
"""

from __future__ import annotations

__all__ = [
    "StrengthTracker",
]

import torch
import torch.nn as nn

from .config import StrengthConfig


class StrengthTracker(nn.Module):
    """
    Tracks memory strength for each position in the sequence.

    Strength decays over time but can be boosted by:
    - Attention (information being actively used)
    - Repeated access (information retrieved multiple times)

    The strength formula:
        strength[i](t) = base_strength[i] * exp(-decay_rate * (t - last_access[i]))
                        + boost_factor * attention_score[i]
                        + frequency_weight * access_count[i]
    """

    def __init__(self, config: StrengthConfig, max_seq_len: int):
        super().__init__()
        self.config = config
        self.max_seq_len = max_seq_len

        # Register buffers for tracking (not trainable parameters)
        # Current strength values
        self.register_buffer("strengths", torch.ones(1, max_seq_len) * config.base_strength)
        # Time since last access
        self.register_buffer("time_since_access", torch.zeros(1, max_seq_len))
        # Access counts
        self.register_buffer("access_counts", torch.zeros(1, max_seq_len))
        # Current time step
        self.register_buffer("current_time", torch.tensor(0.0))

    def reset(self, batch_size: int = 1):
        """Reset strength tracking for new sequences."""
        device = self.strengths.device
        current_batch_size = self.strengths.shape[0]

        if batch_size == current_batch_size:
            # In-place operations preserve buffer registration
            self.strengths.fill_(self.config.base_strength)
            self.time_since_access.zero_()
            self.access_counts.zero_()
            self.current_time.zero_()
        else:
            # Re-register buffers with new batch size
            self.register_buffer("strengths", torch.ones(batch_size, self.max_seq_len, device=device) * self.config.base_strength)
            self.register_buffer("time_since_access", torch.zeros(batch_size, self.max_seq_len, device=device))
            self.register_buffer("access_counts", torch.zeros(batch_size, self.max_seq_len, device=device))
            self.register_buffer("current_time", torch.tensor(0.0, device=device))

    def step(self, time_delta: float = 1.0):
        """
        Advance time and apply decay to all memories.

        Args:
            time_delta: Amount of time to advance
        """
        self.current_time = self.current_time + time_delta
        self.time_since_access = self.time_since_access + time_delta

        # Apply exponential decay
        decay_value = torch.tensor(-self.config.decay_rate * time_delta, device=self.strengths.device)
        decay_factor = torch.exp(decay_value)
        self.strengths = self.strengths * decay_factor

        # Enforce minimum strength
        self.strengths = torch.clamp(self.strengths, min=self.config.min_strength)

    def update_from_attention(self, attention_weights: torch.Tensor, seq_len: int):
        """
        Update strengths based on attention patterns.

        Positions that receive more attention get strength boost.

        Args:
            attention_weights: Attention weights [batch, heads, seq, seq]
            seq_len: Actual sequence length
        """
        # Average attention received per position (how much each position is attended to)
        # Shape: [batch, heads, seq, seq] -> [batch, seq]
        attention_received = attention_weights.mean(dim=1).mean(dim=1)[:, :seq_len]

        # Boost strength based on attention
        boost = self.config.attention_boost * attention_received
        self.strengths[:, :seq_len] = self.strengths[:, :seq_len] + boost

        # Update access tracking for positions with significant attention
        significant_attention = attention_received > self.config.significant_attention_threshold
        self.time_since_access[:, :seq_len] = torch.where(
            significant_attention,
            torch.zeros_like(self.time_since_access[:, :seq_len]),
            self.time_since_access[:, :seq_len],
        )
        self.access_counts[:, :seq_len] = self.access_counts[:, :seq_len] + significant_attention.float()

        # Clamp to valid range
        self.strengths = torch.clamp(self.strengths, min=self.config.min_strength, max=1.0)

    def get_strengths(self, seq_len: int | None = None) -> torch.Tensor:
        """
        Get current strength values.

        Args:
            seq_len: If provided, return only first seq_len positions

        Returns:
            Strength tensor [batch, seq_len]
        """
        if seq_len is not None:
            return self.strengths[:, :seq_len]
        return self.strengths

    def get_statistics(self) -> dict[str, float]:
        """Get summary statistics about current memory strengths."""
        return {
            "mean_strength": self.strengths.mean().item(),
            "min_strength": self.strengths.min().item(),
            "max_strength": self.strengths.max().item(),
            "pct_fuzzy": (self.strengths < self.config.fuzzy_threshold).float().mean().item(),
            "pct_forgotten": (self.strengths < self.config.forget_threshold).float().mean().item(),
            "mean_access_count": self.access_counts.mean().item(),
        }

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Forward pass - just returns current strengths.

        Args:
            seq_len: Sequence length to return

        Returns:
            Strength weights [batch, seq_len]
        """
        return self.get_strengths(seq_len)
