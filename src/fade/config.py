"""
Configuration for FADE proof of concept.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the tiny transformer model."""

    # Model architecture
    vocab_size: int = 256  # Small vocab for synthetic task
    d_model: int = 128  # Embedding dimension
    n_heads: int = 4  # Number of attention heads
    n_layers: int = 2  # Number of transformer layers
    d_ff: int = 512  # Feed-forward dimension
    max_seq_len: int = 64  # Maximum sequence length
    dropout: float = 0.1

    # Derived
    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class StrengthConfig:
    """Configuration for strength tracking."""

    # Decay parameters
    base_strength: float = 1.0  # Initial strength for new information
    decay_rate: float = 0.1  # Exponential decay rate
    min_strength: float = 0.01  # Minimum strength (never fully forget)

    # Boost parameters
    attention_boost: float = 0.3  # Boost from attention
    access_boost: float = 0.1  # Boost from repeated access

    # Strength thresholds
    fuzzy_threshold: float = 0.5  # Below this, consider "fuzzy"
    forget_threshold: float = 0.1  # Below this, consider "forgotten"


@dataclass
class DegradationConfig:
    """Configuration for degradation mechanism."""

    # Degradation method: "noise", "mask", "quantize", or "combined"
    method: str = "combined"

    # Noise injection
    noise_scale: float = 0.5  # Max noise magnitude (scaled by 1-strength)

    # Attention masking
    mask_temperature: float = 2.0  # Softmax temperature for weak memories

    # Quantization
    quantization_levels: int = 8  # Number of discrete levels for weak memories


@dataclass
class FuzzinessConfig:
    """Configuration for fuzziness detection."""

    # Which signals to use
    use_attention_entropy: bool = True
    use_reconstruction_error: bool = True
    use_activation_variance: bool = True

    # Weights for combining signals
    entropy_weight: float = 0.4
    reconstruction_weight: float = 0.3
    variance_weight: float = 0.3

    # Thresholds
    retrieval_threshold: float = 0.6  # Above this fuzziness, trigger retrieval


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Basic training params
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    num_epochs: int = 50

    # Task configuration
    num_key_value_pairs: int = 100  # Number of KV pairs to memorize
    key_length: int = 4  # Length of keys
    value_length: int = 4  # Length of values

    # Time simulation
    time_steps_per_epoch: int = 10  # Simulated time steps for decay

    # Evaluation
    eval_every: int = 5  # Evaluate every N epochs

    # Logging
    log_every: int = 10  # Log every N batches


@dataclass
class FADEConfig:
    """Master configuration combining all sub-configs."""

    model: ModelConfig = None
    strength: StrengthConfig = None
    degradation: DegradationConfig = None
    fuzziness: FuzzinessConfig = None
    training: TrainingConfig = None

    # Random seed
    seed: int = 42

    # Device
    device: str = "cpu"

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.strength is None:
            self.strength = StrengthConfig()
        if self.degradation is None:
            self.degradation = DegradationConfig()
        if self.fuzziness is None:
            self.fuzziness = FuzzinessConfig()
        if self.training is None:
            self.training = TrainingConfig()


def get_default_config() -> FADEConfig:
    """Return default configuration."""
    return FADEConfig()
