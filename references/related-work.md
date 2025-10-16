# Related Work

This document tracks research related to FADE, organized by area.

## Confidence-Based RAG

### Direct Approaches
- **ConfRAG (2024)**: Fine-tuning models to output "I am unsure" responses
- **UncertaintyRAG (2024)**: SNR-based span uncertainty from attention patterns
- **Activation-based confidence (2025)**: Using raw FFN activation patterns as confidence signals

### Indirect Inference Methods
- Token-probability based confidence estimation
- Self-consistency across multiple generations
- Ensemble-based uncertainty quantification

**FADE's distinction**: These use proxy signals to infer confidence post-hoc. FADE makes uncertainty intrinsic through retrieval difficulty.

## Memory Architectures

### Explicit Memory Systems
- **Memory Networks (2015)**: Explicit memory module with attention-based read/write
- **Neural Turing Machines (2014)**: Differentiable memory access with content and location-based addressing
- **Differentiable Neural Computers (2016)**: Extended NTMs with more sophisticated memory operations

**FADE's distinction**: These add explicit memory modules. FADE modifies working memory behavior directly within the transformer architecture.

## Active Forgetting in ML

### Efficiency-Focused Forgetting
- **Expire-Span (2021)**: Learning to forget by predicting span expiration times for efficiency
- **Forgetting Transformer (2024)**: Forget gates in attention mechanisms for context management
- **Learning by Active Forgetting (2021)**: Using inhibitory neurons to prevent catastrophic forgetting

**FADE's distinction**: These use forgetting for computational efficiency or preventing catastrophic forgetting. FADE uses forgetting as information about confidence.

## Catastrophic Forgetting & Continual Learning

- **Elastic Weight Consolidation (2017)**: Protecting important weights from change
- **Progressive Neural Networks (2016)**: Separate columns for different tasks
- **Sleep-like replay for consolidation (2020-2022)**: Periodic replay of important information

**FADE's relationship**: FADE's degradation mechanism could help or hurt continual learning depending on implementation. Requires investigation.

## Calibration & Uncertainty Quantification

### Confidence Calibration
- **Temperature Scaling**: Post-hoc calibration of neural network outputs
- **Expected Calibration Error (ECE)**: Metric for measuring calibration quality
- **Platt Scaling**: Logistic regression for probability calibration

### Bayesian Approaches
- **Monte Carlo Dropout**: Dropout at test time for uncertainty estimation
- **Deep Ensembles**: Multiple models for uncertainty quantification
- **Variational Inference**: Bayesian neural networks

**FADE's distinction**: These calibrate existing confidence measures. FADE provides a different source of confidence signal.

## Biological Memory Systems

### Cognitive Science
- **Forgetting curves (Ebbinghaus)**: Mathematical models of memory decay
- **Spacing effect**: Spaced repetition improves retention
- **Context-dependent memory**: Environmental context affects recall

### Neuroscience
- **Synaptic plasticity**: Strengthening/weakening of neural connections
- **Memory consolidation**: Transfer from short-term to long-term memory
- **Hippocampal indexing theory**: Hippocampus stores pointers to cortical representations

**FADE's inspiration**: Human memory naturally degrades and this degradation provides information about certainty.

## RAG Systems

### General RAG
- **Dense Passage Retrieval (DPR)**: Neural retrieval for QA
- **RAG (Lewis et al. 2020)**: Original retrieval-augmented generation
- **REALM**: Pre-training with retrieval

### Adaptive RAG
- **Self-RAG**: Model decides when to retrieve
- **Adaptive retrieval**: Dynamic retrieval based on confidence
- **FiD (Fusion-in-Decoder)**: Efficient processing of retrieved passages

**FADE's contribution**: Novel confidence signal for triggering retrieval.

## To Explore

### Potential Connections
- Attention mechanisms and their relationship to memory strength
- Meta-learning for learning decay parameters
- Hierarchical memory systems (multiple timescales)
- Cross-modal memory degradation

### Related but Unexplored
- Working memory capacity limits in cognitive science
- Information theory perspectives on forgetting
- Optimal forgetting for different task types

## Contributing

Found relevant work? Please add it! Categories to expand:
- Recent confidence calibration methods
- New RAG approaches
- Memory architecture innovations
- Cognitive science findings on forgetting

---

**Note**: This is a living document. As research progresses, we'll update with new relevant work and refine the distinctions.
