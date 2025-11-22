"""
Degradation Module for FADE.

Implements mechanisms to degrade representations based on memory strength.
Weaker memories become "fuzzier" - noisier, less precise representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import DegradationConfig


class DegradationModule(nn.Module):
    """
    Degrades representations based on memory strength.

    Three degradation methods:
    1. Noise injection: Add Gaussian noise proportional to (1 - strength)
    2. Attention masking: Soften attention to weak memories
    3. Quantization: Reduce precision of weak memory representations

    Can use any combination of these methods.
    """

    def __init__(self, config: DegradationConfig):
        super().__init__()
        self.config = config

    def add_noise(
        self,
        x: torch.Tensor,
        strengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to representations inversely proportional to strength.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            strengths: Strength values [batch, seq_len]

        Returns:
            Degraded tensor [batch, seq_len, d_model]
        """
        if not self.training and not self.config.method in ["noise", "combined"]:
            return x

        # Weakness = 1 - strength (high weakness = more noise)
        weakness = (1 - strengths).unsqueeze(-1)  # [batch, seq_len, 1]

        # Generate noise scaled by weakness
        noise = torch.randn_like(x) * self.config.noise_scale * weakness

        return x + noise

    def apply_attention_masking(
        self,
        attention_scores: torch.Tensor,
        strengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply strength-based temperature scaling to attention.

        Weak memories get higher temperature (softer attention distribution),
        making them less precise in what they attend to.

        Args:
            attention_scores: Raw attention scores [batch, heads, seq, seq]
            strengths: Strength values [batch, seq_len]

        Returns:
            Modified attention scores
        """
        # Compute temperature based on weakness
        # Weak memories -> high temperature -> softer attention
        weakness = 1 - strengths  # [batch, seq_len]
        temperature = 1.0 + (self.config.mask_temperature - 1.0) * weakness

        # Apply to both query and key dimensions
        # Temperature for queries (rows)
        temp_q = temperature.unsqueeze(1).unsqueeze(-1)  # [batch, 1, seq, 1]
        # Temperature for keys (columns)
        temp_k = temperature.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq]

        # Combined temperature effect
        combined_temp = (temp_q + temp_k) / 2

        return attention_scores / combined_temp

    def quantize(
        self,
        x: torch.Tensor,
        strengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Quantize representations based on strength.

        Weak memories get coarser quantization (fewer levels),
        representing loss of precision in the memory.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            strengths: Strength values [batch, seq_len]

        Returns:
            Quantized tensor
        """
        # Number of quantization levels varies with strength
        # Strong memory: many levels (high precision)
        # Weak memory: few levels (low precision)

        weakness = (1 - strengths).unsqueeze(-1)  # [batch, seq_len, 1]

        # Effective levels: from config.quantization_levels (weak) to 256 (strong)
        min_levels = self.config.quantization_levels
        max_levels = 256

        # Compute per-position quantization level
        levels = max_levels - weakness * (max_levels - min_levels)
        levels = levels.clamp(min=min_levels, max=max_levels)

        # Normalize to [0, 1] range for quantization
        x_min = x.min(dim=-1, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0]
        x_range = (x_max - x_min).clamp(min=1e-6)
        x_norm = (x - x_min) / x_range

        # Quantize
        x_quantized = torch.round(x_norm * levels) / levels

        # Denormalize
        x_out = x_quantized * x_range + x_min

        # Blend based on training mode (straight-through estimator for gradients)
        if self.training:
            # Use straight-through: forward uses quantized, backward uses original
            return x + (x_out - x).detach()
        else:
            return x_out

    def forward(
        self,
        x: torch.Tensor,
        strengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply degradation to representations.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            strengths: Strength values [batch, seq_len]

        Returns:
            Degraded tensor [batch, seq_len, d_model]
        """
        method = self.config.method

        if method == "noise":
            return self.add_noise(x, strengths)
        elif method == "quantize":
            return self.quantize(x, strengths)
        elif method == "mask":
            # Masking is applied to attention, not embeddings
            # Return input unchanged; masking happens in attention
            return x
        elif method == "combined":
            # Apply both noise and quantization
            x = self.add_noise(x, strengths)
            x = self.quantize(x, strengths)
            return x
        else:
            raise ValueError(f"Unknown degradation method: {method}")


class GradualDegradation(nn.Module):
    """
    Applies degradation gradually over multiple time steps.

    Useful for simulating memory decay during inference.
    """

    def __init__(self, config: DegradationConfig):
        super().__init__()
        self.degradation = DegradationModule(config)

    def forward(
        self,
        x: torch.Tensor,
        initial_strengths: torch.Tensor,
        time_steps: int,
        decay_rate: float = 0.1,
    ) -> tuple:
        """
        Apply gradual degradation over multiple time steps.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            initial_strengths: Starting strength values [batch, seq_len]
            time_steps: Number of time steps to simulate
            decay_rate: Decay rate per step

        Returns:
            Tuple of (final_x, final_strengths, trajectory)
        """
        current_x = x
        current_strengths = initial_strengths
        trajectory = [(current_x.clone(), current_strengths.clone())]

        for _ in range(time_steps):
            # Decay strengths
            current_strengths = current_strengths * (1 - decay_rate)
            current_strengths = current_strengths.clamp(min=0.01)

            # Apply degradation
            current_x = self.degradation(current_x, current_strengths)

            trajectory.append((current_x.clone(), current_strengths.clone()))

        return current_x, current_strengths, trajectory
