"""
Tiny Transformer model for FADE proof of concept.

A minimal transformer architecture designed to run on CPU
while still demonstrating the core FADE mechanisms.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class MultiHeadAttention(nn.Module):
    """Multi-head attention with access to attention weights."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(self.d_head)

        # Store attention weights for fuzziness detection
        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        strength_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional strength-based attention modulation.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Attention mask [batch, seq_len, seq_len]
            strength_weights: Memory strength [batch, seq_len] - used to modulate attention

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)

        # Transpose for attention: [batch, heads, seq, d_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply strength-based modulation if provided
        # Weaker memories get softer attention (higher temperature)
        if strength_weights is not None:
            # strength_weights: [batch, seq_len]
            # Expand for broadcasting: [batch, 1, 1, seq_len]
            strength_expanded = strength_weights.unsqueeze(1).unsqueeze(2)
            # Reduce attention to weak memories by adding negative bias
            weakness_penalty = (1 - strength_expanded) * -2.0
            scores = scores + weakness_penalty

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Store for fuzziness detection
        self.last_attention_weights = attention_weights.detach()

        # Apply attention to values
        output = torch.matmul(attention_weights, v)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        return output


class FeedForward(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        strength_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x), mask, strength_weights)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x


class TinyTransformer(nn.Module):
    """
    Tiny transformer for FADE proof of concept.

    Designed to be small enough to run on CPU while demonstrating
    the core memory degradation and fuzziness detection mechanisms.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Output projection
        self.norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

        # Store intermediate activations for fuzziness detection
        self.last_hidden_states: Optional[torch.Tensor] = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        strength_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            mask: Attention mask [batch, seq_len, seq_len]
            strength_weights: Memory strength per position [batch, seq_len]

        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            hidden_states: Final hidden states [batch, seq_len, d_model]
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask, strength_weights)

        # Store hidden states for fuzziness detection
        self.last_hidden_states = x.detach()

        # Output projection
        x = self.norm(x)
        logits = self.output_proj(x)

        return logits, self.last_hidden_states

    def get_attention_weights(self) -> list:
        """Get attention weights from all layers for fuzziness detection."""
        return [block.attention.last_attention_weights for block in self.blocks]
