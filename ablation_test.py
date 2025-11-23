"""
Ablation test to distinguish FADE mechanism from pure regularization.

Tests:
1. FADE with computed strengths (original)
2. FADE with random strengths (same noise, random targeting)
3. FADE with constant strengths (uniform noise, no targeting)
4. Baseline (no degradation)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
from fade.config import get_default_config
from fade.fade_model import FADEModel
from fade.data import create_data_loaders
from fade.trainer import FADETrainer, BaselineTrainer
from fade.strength import StrengthTracker

def run_ablation():
    config = get_default_config()
    config.training.num_epochs = 50
    config.training.eval_every = 50
    config.seed = 42
    
    # Create dataset
    train_loader, eval_loader, dataset = create_data_loaders(
        num_pairs=config.training.num_key_value_pairs,
        key_length=config.training.key_length,
        value_length=config.training.value_length,
        vocab_size=config.model.vocab_size,
        batch_size=config.training.batch_size,
        seed=config.seed,
    )
    
    results = {}
    
    # Store original method
    original_get_strengths = StrengthTracker.get_strengths
    
    print("=" * 60)
    print("ABLATION TEST: FADE Mechanism vs Regularization")
    print("=" * 60)
    
    # Test 1: FADE with computed strengths (original)
    print("\n[1/4] FADE with COMPUTED strengths (original)...")
    torch.manual_seed(42); np.random.seed(42)
    StrengthTracker.get_strengths = original_get_strengths
    
    model1 = FADEModel(config)
    trainer1 = FADETrainer(model1, config, train_loader, eval_loader, dataset, config.device)
    trainer1.train()
    results['computed'] = {
        'ece': trainer1.state.best_ece,
        'corr': trainer1.state.best_correlation
    }
    print(f"      ECE: {results['computed']['ece']:.4f}, Corr: {results['computed']['corr']:.4f}")
    
    # Test 2: FADE with random strengths
    print("\n[2/4] FADE with RANDOM strengths...")
    torch.manual_seed(42); np.random.seed(42)
    
    def random_strengths(self, seq_len):
        batch_size = self.strengths.shape[0]
        # Random strengths with same mean/std as typical computed strengths
        return torch.rand(batch_size, seq_len, device=self.strengths.device) * 0.8 + 0.1
    
    StrengthTracker.get_strengths = random_strengths
    
    model2 = FADEModel(config)
    trainer2 = FADETrainer(model2, config, train_loader, eval_loader, dataset, config.device)
    trainer2.train()
    results['random'] = {
        'ece': trainer2.state.best_ece,
        'corr': trainer2.state.best_correlation
    }
    print(f"      ECE: {results['random']['ece']:.4f}, Corr: {results['random']['corr']:.4f}")
    
    # Test 3: FADE with constant strengths (uniform degradation)
    print("\n[3/4] FADE with CONSTANT strengths (uniform noise)...")
    torch.manual_seed(42); np.random.seed(42)
    
    def constant_strengths(self, seq_len):
        batch_size = self.strengths.shape[0]
        # Constant strength = 0.5 (middle value, uniform degradation everywhere)
        return torch.ones(batch_size, seq_len, device=self.strengths.device) * 0.5
    
    StrengthTracker.get_strengths = constant_strengths
    
    model3 = FADEModel(config)
    trainer3 = FADETrainer(model3, config, train_loader, eval_loader, dataset, config.device)
    trainer3.train()
    results['constant'] = {
        'ece': trainer3.state.best_ece,
        'corr': trainer3.state.best_correlation
    }
    print(f"      ECE: {results['constant']['ece']:.4f}, Corr: {results['constant']['corr']:.4f}")
    
    # Restore original method
    StrengthTracker.get_strengths = original_get_strengths
    
    # Test 4: Baseline (no degradation)
    print("\n[4/4] Baseline (no degradation)...")
    torch.manual_seed(42); np.random.seed(42)
    
    model4 = FADEModel(config)
    trainer4 = BaselineTrainer(model4, config, train_loader, eval_loader, dataset, config.device)
    trainer4.train()
    results['baseline'] = {
        'ece': trainer4.state.best_ece,
        'corr': trainer4.state.best_correlation
    }
    print(f"      ECE: {results['baseline']['ece']:.4f}, Corr: {results['baseline']['corr']:.4f}")
    
    # Analysis
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Method':<25} {'ECE':<10} {'Fuzz-Err Corr':<15}")
    print("-" * 50)
    for name, r in results.items():
        print(f"{name:<25} {r['ece']:<10.4f} {r['corr']:<15.4f}")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    computed_ece = results['computed']['ece']
    random_ece = results['random']['ece']
    constant_ece = results['constant']['ece']
    baseline_ece = results['baseline']['ece']
    
    # Key comparison: computed vs random/constant
    if computed_ece < random_ece * 0.85 and computed_ece < constant_ece * 0.85:
        print("✓ COMPUTED strengths significantly outperform RANDOM and CONSTANT")
        print("  -> The strength computation mechanism provides SPECIFIC benefit")
        print("  -> FADE is NOT just noise regularization!")
        mechanism_works = True
    elif computed_ece < baseline_ece * 0.5:
        if abs(computed_ece - random_ece) / computed_ece < 0.15:
            print("~ COMPUTED and RANDOM strengths perform similarly")
            print("  -> Both beat baseline, but targeting doesn't matter much")
            print("  -> FADE may primarily be noise regularization")
            mechanism_works = False
        else:
            print("? Mixed results - needs more investigation")
            mechanism_works = None
    else:
        print("✗ Minimal improvement over baseline")
        print("  -> FADE may not be working as intended")
        mechanism_works = False
    
    # Additional insight: check if constant beats random
    if constant_ece < random_ece * 0.9:
        print("\n  Note: CONSTANT strengths beat RANDOM")
        print("  -> Uniform noise is more effective than random targeting")
    
    return results, mechanism_works

if __name__ == "__main__":
    results, mechanism_works = run_ablation()
