"""
Tests for FADE model components.
"""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fade.config import FADEConfig, get_default_config, ModelConfig
from fade.model import TinyTransformer
from fade.strength import StrengthTracker
from fade.degradation import DegradationModule
from fade.fuzziness import FuzzinessDetector
from fade.fade_model import FADEModel


class TestTinyTransformer:
    """Tests for the tiny transformer model."""

    def test_forward_pass(self):
        """Test basic forward pass."""
        config = ModelConfig(vocab_size=100, d_model=64, n_heads=2, n_layers=2)
        model = TinyTransformer(config)

        batch_size, seq_len = 4, 16
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        logits, hidden_states = model(input_ids)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert hidden_states.shape == (batch_size, seq_len, config.d_model)

    def test_attention_weights_stored(self):
        """Test that attention weights are accessible."""
        config = ModelConfig(vocab_size=100, d_model=64, n_heads=2, n_layers=2)
        model = TinyTransformer(config)

        input_ids = torch.randint(0, 100, (2, 8))
        _ = model(input_ids)

        attention_weights = model.get_attention_weights()
        assert len(attention_weights) == config.n_layers
        assert attention_weights[0].shape == (2, config.n_heads, 8, 8)

    def test_strength_weights_affect_attention(self):
        """Test that strength weights modify attention."""
        config = ModelConfig(vocab_size=100, d_model=64, n_heads=2, n_layers=2)
        model = TinyTransformer(config)

        input_ids = torch.randint(0, 100, (2, 8))

        # Run without strength weights
        logits1, _ = model(input_ids, strength_weights=None)

        # Run with strength weights (some positions weak)
        strength_weights = torch.ones(2, 8)
        strength_weights[:, 4:] = 0.1  # Weaken later positions

        logits2, _ = model(input_ids, strength_weights=strength_weights)

        # Outputs should differ
        assert not torch.allclose(logits1, logits2)

    def test_parameter_count(self):
        """Test parameter counting."""
        config = ModelConfig(vocab_size=100, d_model=64, n_heads=2, n_layers=2)
        model = TinyTransformer(config)

        num_params = model.count_parameters()
        assert num_params > 0
        assert num_params < 1_000_000  # Should be a "tiny" model


class TestStrengthTracker:
    """Tests for strength tracking."""

    def test_initialization(self):
        """Test initial strengths are set correctly."""
        from fade.config import StrengthConfig
        config = StrengthConfig(base_strength=1.0)
        tracker = StrengthTracker(config, max_seq_len=32)

        strengths = tracker.get_strengths(16)
        assert torch.allclose(strengths, torch.ones(1, 16))

    def test_decay(self):
        """Test that strengths decay over time."""
        from fade.config import StrengthConfig
        config = StrengthConfig(decay_rate=0.5)
        tracker = StrengthTracker(config, max_seq_len=32)

        initial = tracker.get_strengths(16).clone()
        tracker.step(time_delta=1.0)
        after_decay = tracker.get_strengths(16)

        # Strengths should be lower after decay
        assert (after_decay < initial).all()

    def test_attention_boost(self):
        """Test that attention boosts strength."""
        from fade.config import StrengthConfig
        config = StrengthConfig(attention_boost=0.5, decay_rate=0.1)
        tracker = StrengthTracker(config, max_seq_len=32)

        # Decay first
        tracker.step(time_delta=2.0)
        before = tracker.get_strengths(8).clone()

        # Apply attention
        attn = torch.ones(1, 2, 8, 8) / 8  # Uniform attention
        tracker.update_from_attention(attn, seq_len=8)
        after = tracker.get_strengths(8)

        # Strengths should increase
        assert (after > before).all()

    def test_minimum_strength(self):
        """Test that strength doesn't go below minimum."""
        from fade.config import StrengthConfig
        config = StrengthConfig(decay_rate=1.0, min_strength=0.01)
        tracker = StrengthTracker(config, max_seq_len=32)

        # Decay many times
        for _ in range(100):
            tracker.step(time_delta=1.0)

        strengths = tracker.get_strengths(16)
        assert (strengths >= config.min_strength).all()


class TestDegradation:
    """Tests for degradation mechanism."""

    def test_noise_injection(self):
        """Test noise injection increases with weakness."""
        from fade.config import DegradationConfig
        config = DegradationConfig(method="noise", noise_scale=1.0)
        degradation = DegradationModule(config)
        degradation.train()

        x = torch.randn(2, 8, 64)

        # Strong memory (low noise)
        strong = torch.ones(2, 8)
        x_strong = degradation(x, strong)

        # Weak memory (high noise)
        weak = torch.full((2, 8), 0.1)
        x_weak = degradation(x, weak)

        # Weak should deviate more from original
        strong_diff = (x_strong - x).abs().mean()
        weak_diff = (x_weak - x).abs().mean()
        assert weak_diff > strong_diff

    def test_quantization(self):
        """Test quantization reduces precision for weak memories."""
        from fade.config import DegradationConfig
        config = DegradationConfig(method="quantize", quantization_levels=4)
        degradation = DegradationModule(config)

        x = torch.randn(2, 8, 64)
        weak = torch.full((2, 8), 0.1)

        x_quantized = degradation(x, weak)

        # Quantized values should be different
        assert not torch.allclose(x, x_quantized)


class TestFuzzinessDetector:
    """Tests for fuzziness detection."""

    def test_entropy_computation(self):
        """Test attention entropy computation."""
        from fade.config import FuzzinessConfig
        config = FuzzinessConfig()
        detector = FuzzinessDetector(config, d_model=64)

        # Uniform attention = high entropy
        uniform_attn = torch.ones(2, 4, 8, 8) / 8
        entropy_uniform = detector.compute_attention_entropy([uniform_attn])

        # Peaked attention = low entropy
        peaked_attn = torch.zeros(2, 4, 8, 8)
        peaked_attn[:, :, :, 0] = 1.0  # All attention on first position
        entropy_peaked = detector.compute_attention_entropy([peaked_attn])

        assert entropy_uniform.mean() > entropy_peaked.mean()

    def test_fuzziness_output_range(self):
        """Test that fuzziness scores are in valid range."""
        from fade.config import FuzzinessConfig
        config = FuzzinessConfig()
        detector = FuzzinessDetector(config, d_model=64)

        hidden_states = torch.randn(2, 8, 64)
        attn = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)

        fuzziness, components = detector.compute_fuzziness(hidden_states, [attn])

        # Fuzziness should be bounded
        assert (fuzziness >= 0).all()
        assert (fuzziness <= 1).all()


class TestFADEModel:
    """Tests for the full FADE model."""

    def test_forward_pass(self):
        """Test full forward pass with all components."""
        config = get_default_config()
        config.model.vocab_size = 100
        config.model.d_model = 64
        config.model.n_layers = 2

        model = FADEModel(config)

        input_ids = torch.randint(0, 100, (4, 16))
        output = model(input_ids)

        assert "logits" in output
        assert "strengths" in output
        assert "fuzziness" in output
        assert "should_retrieve" in output

    def test_time_advance_affects_strength(self):
        """Test that advancing time reduces strength."""
        config = get_default_config()
        config.model.vocab_size = 100
        model = FADEModel(config)

        input_ids = torch.randint(0, 100, (2, 8))

        # First forward pass
        model.reset_memory(2)
        output1 = model(input_ids)
        strength1 = output1["strengths"].clone()

        # Advance time
        model.advance_time(5.0)

        # Second forward pass
        output2 = model(input_ids)
        strength2 = output2["strengths"]

        # Strength should have decayed
        assert (strength2 < strength1).all()

    def test_loss_computation(self):
        """Test loss computation."""
        config = get_default_config()
        config.model.vocab_size = 100
        model = FADEModel(config)

        input_ids = torch.randint(0, 100, (4, 16))
        targets = torch.randint(0, 100, (4, 16))

        output = model(input_ids)
        losses = model.compute_loss(
            output["logits"],
            targets,
            output.get("fuzziness"),
        )

        assert "total_loss" in losses
        assert "prediction_loss" in losses
        assert losses["total_loss"].item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
