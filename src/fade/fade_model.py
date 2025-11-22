"""
FADE Model - Combining all components.

Integrates:
- TinyTransformer (base model)
- StrengthTracker (memory strength)
- DegradationModule (memory decay)
- FuzzinessDetector (uncertainty detection)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .config import FADEConfig, ModelConfig
from .model import TinyTransformer
from .strength import StrengthTracker
from .degradation import DegradationModule
from .fuzziness import FuzzinessDetector


class FADEModel(nn.Module):
    """
    FADE: Fuzzy Associative Degradation Engine

    A transformer model with built-in memory degradation and
    fuzziness detection for improved confidence calibration.
    """

    def __init__(self, config: FADEConfig):
        super().__init__()
        self.config = config

        # Core transformer
        self.transformer = TinyTransformer(config.model)

        # FADE components
        self.strength_tracker = StrengthTracker(config.strength, config.model.max_seq_len)
        self.degradation = DegradationModule(config.degradation)
        self.fuzziness_detector = FuzzinessDetector(config.fuzziness, config.model.d_model)

        # Store original embeddings for reconstruction loss
        self.original_embeddings: Optional[torch.Tensor] = None

    def reset_memory(self, batch_size: int = 1):
        """Reset memory tracking for new sequences."""
        self.strength_tracker.reset(batch_size)

    def advance_time(self, time_delta: float = 1.0):
        """Advance time and decay memories."""
        self.strength_tracker.step(time_delta)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        apply_degradation: bool = True,
        return_fuzziness: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with FADE mechanics.

        Args:
            input_ids: Token IDs [batch, seq_len]
            mask: Attention mask
            apply_degradation: Whether to apply memory degradation
            return_fuzziness: Whether to compute and return fuzziness

        Returns:
            Dict containing:
                - logits: Output logits [batch, seq_len, vocab_size]
                - hidden_states: Final hidden states
                - strengths: Memory strength values
                - fuzziness: Fuzziness scores (if return_fuzziness)
                - should_retrieve: Retrieval mask (if return_fuzziness)
        """
        batch_size, seq_len = input_ids.shape

        # Ensure strength tracker has correct batch size
        if self.strength_tracker.strengths.shape[0] != batch_size:
            self.reset_memory(batch_size)

        # Get current strengths
        strengths = self.strength_tracker.get_strengths(seq_len)

        # Get embeddings (for degradation and reconstruction)
        embeddings = self.transformer.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        embeddings = embeddings + self.transformer.position_embedding(positions)

        # Store original for reconstruction loss
        self.original_embeddings = embeddings.detach()

        # Apply degradation to embeddings if enabled
        if apply_degradation:
            embeddings = self.degradation(embeddings, strengths)

        # Pass through transformer with strength-modulated attention
        # We need to modify the transformer to use our degraded embeddings
        x = embeddings
        for block in self.transformer.blocks:
            x = block(x, mask, strengths)

        # Store hidden states
        hidden_states = x.detach()
        self.transformer.last_hidden_states = hidden_states

        # Output projection
        x = self.transformer.norm(x)
        logits = self.transformer.output_proj(x)

        # Update strengths based on attention patterns
        attention_weights = self.transformer.get_attention_weights()
        if attention_weights and attention_weights[0] is not None:
            # Average attention across layers
            avg_attention = torch.stack([w for w in attention_weights if w is not None]).mean(0)
            self.strength_tracker.update_from_attention(avg_attention, seq_len)

        # Prepare output
        output = {
            "logits": logits,
            "hidden_states": hidden_states,
            "strengths": strengths,
        }

        # Compute fuzziness if requested
        if return_fuzziness:
            fuzziness, should_retrieve = self.fuzziness_detector(
                hidden_states,
                attention_weights,
                self.original_embeddings,
            )
            output["fuzziness"] = fuzziness
            output["should_retrieve"] = should_retrieve

        return output

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        fuzziness: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.

        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            targets: Target token IDs [batch, seq_len]
            fuzziness: Fuzziness scores [batch, seq_len]
            mask: Loss mask [batch, seq_len]

        Returns:
            Dict of loss components
        """
        # Main prediction loss
        pred_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none",
        ).view(targets.shape)

        if mask is not None:
            pred_loss = pred_loss * mask
            pred_loss = pred_loss.sum() / mask.sum().clamp(min=1)
        else:
            pred_loss = pred_loss.mean()

        losses = {"prediction_loss": pred_loss}

        # Calibration loss: fuzziness should predict errors
        if fuzziness is not None:
            # Compute per-position error (1 if wrong, 0 if correct)
            predictions = logits.argmax(dim=-1)
            errors = (predictions != targets).float()

            # Fuzziness should correlate with errors
            # Loss: encourage fuzziness to match error rate
            calibration_loss = F.mse_loss(fuzziness, errors)
            losses["calibration_loss"] = calibration_loss

            # Also compute correlation for monitoring
            if fuzziness.numel() > 1:
                fuzz_flat = fuzziness.view(-1)
                err_flat = errors.view(-1)
                correlation = torch.corrcoef(torch.stack([fuzz_flat, err_flat]))[0, 1]
                if not torch.isnan(correlation):
                    losses["fuzziness_error_correlation"] = correlation

        # Total loss
        total_loss = pred_loss
        if "calibration_loss" in losses:
            total_loss = total_loss + 0.1 * losses["calibration_loss"]

        losses["total_loss"] = total_loss

        return losses

    def get_memory_statistics(self) -> Dict[str, float]:
        """Get current memory strength statistics."""
        return self.strength_tracker.get_statistics()

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FADEWithRetrieval(nn.Module):
    """
    FADE model with simulated retrieval augmentation.

    When fuzziness is high, retrieves from a simple key-value store
    and incorporates the retrieved information.
    """

    def __init__(self, config: FADEConfig):
        super().__init__()
        self.fade = FADEModel(config)

        # Simple retrieval store (key-value memory)
        self.retrieval_store: Dict[str, torch.Tensor] = {}

        # Projection for incorporating retrieved info
        self.retrieval_proj = nn.Linear(config.model.d_model * 2, config.model.d_model)

    def store(self, keys: List[str], values: torch.Tensor):
        """Store key-value pairs for retrieval."""
        for key, value in zip(keys, values):
            self.retrieval_store[key] = value.detach()

    def retrieve(self, query_embeddings: torch.Tensor, k: int = 1) -> torch.Tensor:
        """
        Retrieve most similar values from store.

        Simple cosine similarity retrieval.
        """
        if not self.retrieval_store:
            return torch.zeros_like(query_embeddings)

        # Stack all stored values
        keys = list(self.retrieval_store.keys())
        values = torch.stack([self.retrieval_store[k] for k in keys])  # [num_stored, d_model]

        # Compute similarities
        query_norm = F.normalize(query_embeddings, dim=-1)
        value_norm = F.normalize(values, dim=-1)
        similarities = torch.matmul(query_norm, value_norm.T)  # [batch, seq, num_stored]

        # Get top-k
        top_k_indices = similarities.topk(min(k, len(keys)), dim=-1).indices
        retrieved = values[top_k_indices].mean(dim=-2)  # Average top-k

        return retrieved

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_retrieval: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with optional retrieval augmentation."""
        # First pass through FADE
        output = self.fade(input_ids, mask, apply_degradation=True, return_fuzziness=True)

        # If retrieval is enabled and we have a store
        if use_retrieval and self.retrieval_store and "should_retrieve" in output:
            should_retrieve = output["should_retrieve"]

            if should_retrieve.any():
                # Retrieve for positions that need it
                hidden_states = output["hidden_states"]
                retrieved = self.retrieve(hidden_states)

                # Blend retrieved information where needed
                retrieve_mask = should_retrieve.unsqueeze(-1).float()
                combined = torch.cat([hidden_states, retrieved], dim=-1)
                blended = self.retrieval_proj(combined)

                # Only update positions that needed retrieval
                augmented_hidden = hidden_states * (1 - retrieve_mask) + blended * retrieve_mask

                # Recompute logits with augmented hidden states
                augmented_logits = self.fade.transformer.output_proj(
                    self.fade.transformer.norm(augmented_hidden)
                )
                output["augmented_logits"] = augmented_logits
                output["retrieval_count"] = should_retrieve.sum()

        return output
