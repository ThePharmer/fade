"""
Configuration for FADE proof of concept.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
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

    # Weight initialization
    weight_init_std: float = 0.02  # Standard deviation for weight initialization

    # Attention modulation
    weakness_penalty: float = -2.0  # Penalty applied to weak memory attention scores

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")

    # Derived
    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


@dataclass(frozen=True)
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

    # Attention threshold for determining significant attention
    significant_attention_threshold: float = 0.1

    def __post_init__(self):
        if self.base_strength <= 0:
            raise ValueError(f"base_strength must be positive, got {self.base_strength}")
        if self.decay_rate < 0:
            raise ValueError(f"decay_rate must be non-negative, got {self.decay_rate}")
        if self.min_strength < 0:
            raise ValueError(f"min_strength must be non-negative, got {self.min_strength}")
        if self.min_strength > self.base_strength:
            raise ValueError(
                f"min_strength ({self.min_strength}) cannot exceed base_strength ({self.base_strength})"
            )
        if self.attention_boost < 0:
            raise ValueError(f"attention_boost must be non-negative, got {self.attention_boost}")
        if self.access_boost < 0:
            raise ValueError(f"access_boost must be non-negative, got {self.access_boost}")
        if not 0.0 <= self.fuzzy_threshold <= 1.0:
            raise ValueError(f"fuzzy_threshold must be in [0.0, 1.0], got {self.fuzzy_threshold}")
        if not 0.0 <= self.forget_threshold <= 1.0:
            raise ValueError(f"forget_threshold must be in [0.0, 1.0], got {self.forget_threshold}")
        if self.forget_threshold > self.fuzzy_threshold:
            raise ValueError(
                f"forget_threshold ({self.forget_threshold}) should not exceed fuzzy_threshold ({self.fuzzy_threshold})"
            )


_VALID_DEGRADATION_METHODS = {"noise", "mask", "quantize", "combined"}


@dataclass(frozen=True)
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

    def __post_init__(self):
        if self.method not in _VALID_DEGRADATION_METHODS:
            raise ValueError(
                f"method must be one of {_VALID_DEGRADATION_METHODS}, got '{self.method}'"
            )
        if self.noise_scale < 0:
            raise ValueError(f"noise_scale must be non-negative, got {self.noise_scale}")
        if self.mask_temperature <= 0:
            raise ValueError(f"mask_temperature must be positive, got {self.mask_temperature}")
        if self.quantization_levels < 2:
            raise ValueError(f"quantization_levels must be at least 2, got {self.quantization_levels}")


@dataclass(frozen=True)
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

    # EMA momentum for running statistics
    ema_momentum: float = 0.1

    def __post_init__(self):
        if self.entropy_weight < 0:
            raise ValueError(f"entropy_weight must be non-negative, got {self.entropy_weight}")
        if self.reconstruction_weight < 0:
            raise ValueError(f"reconstruction_weight must be non-negative, got {self.reconstruction_weight}")
        if self.variance_weight < 0:
            raise ValueError(f"variance_weight must be non-negative, got {self.variance_weight}")
        weight_sum = self.entropy_weight + self.reconstruction_weight + self.variance_weight
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Fuzziness weights must sum to 1.0, got {weight_sum} "
                f"(entropy={self.entropy_weight}, reconstruction={self.reconstruction_weight}, variance={self.variance_weight})"
            )
        if not 0.0 <= self.retrieval_threshold <= 1.0:
            raise ValueError(f"retrieval_threshold must be in [0.0, 1.0], got {self.retrieval_threshold}")


@dataclass(frozen=True)
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

    # Gradient clipping
    gradient_clip_norm: float = 1.0  # Maximum gradient norm for clipping

    # Loss weights
    calibration_loss_weight: float = 0.1  # Weight for calibration loss term

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.num_key_value_pairs <= 0:
            raise ValueError(f"num_key_value_pairs must be positive, got {self.num_key_value_pairs}")
        if self.key_length <= 0:
            raise ValueError(f"key_length must be positive, got {self.key_length}")
        if self.value_length <= 0:
            raise ValueError(f"value_length must be positive, got {self.value_length}")
        if self.time_steps_per_epoch <= 0:
            raise ValueError(f"time_steps_per_epoch must be positive, got {self.time_steps_per_epoch}")
        if self.eval_every <= 0:
            raise ValueError(f"eval_every must be positive, got {self.eval_every}")
        if self.log_every <= 0:
            raise ValueError(f"log_every must be positive, got {self.log_every}")


@dataclass(frozen=True)
class FADEConfig:
    """Master configuration combining all sub-configs."""

    model: ModelConfig | None = None
    strength: StrengthConfig | None = None
    degradation: DegradationConfig | None = None
    fuzziness: FuzzinessConfig | None = None
    training: TrainingConfig | None = None

    # Random seed
    seed: int = 42

    # Device
    device: str = "cpu"

    def __post_init__(self):
        # Initialize sub-configs with defaults if not provided
        # Use object.__setattr__ since this is a frozen dataclass
        if self.model is None:
            object.__setattr__(self, 'model', ModelConfig())
        if self.strength is None:
            object.__setattr__(self, 'strength', StrengthConfig())
        if self.degradation is None:
            object.__setattr__(self, 'degradation', DegradationConfig())
        if self.fuzziness is None:
            object.__setattr__(self, 'fuzziness', FuzzinessConfig())
        if self.training is None:
            object.__setattr__(self, 'training', TrainingConfig())

        # Validate FADEConfig-specific parameters
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        valid_devices = {"cpu", "cuda", "mps"}
        # Allow cuda:N format for multi-GPU
        device_base = self.device.split(":")[0] if ":" in self.device else self.device
        if device_base not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices} (or cuda:N), got '{self.device}'")


def get_default_config() -> FADEConfig:
    """Return default configuration."""
    return FADEConfig()
