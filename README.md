# FADE

**A memory degradation architecture for reliable AI systems**

AI that deliberately forgets like humans do, using memory degradation as an intrinsic confidence signal for reliable decision-making.

## The Problem

Current AI systems suffer from confident hallucinations. They maintain perfect working memory and try to infer uncertainty from output statistics—like judging if someone is confident by analyzing their tone of voice rather than their actual certainty.

**Result**: Detailed, authoritative-sounding answers that are completely wrong.

## The Solution

**FADE** inverts the paradigm: implement degrading working memory by design. When the AI tries to recall information and retrieval feels "fuzzy," that difficulty itself is the confidence signal to query external sources.

```
Working Memory (degrades) → Fuzzy Retrieval → Trigger External Lookup
        ↓
Important info stays strong
Rarely used info fades
        ↓
Intrinsic uncertainty signal
```

## How It Works

**Dual Memory System**:
- **Working Memory**: Active context with representations that naturally degrade based on access patterns
- **Persistent Storage**: Complete backup (RAG corpus, databases, conversation history)

**Information Strength**:
- Increases with: attention, repetition, recency, importance
- Decreases with: time elapsed, lack of access
- Governs: representation fidelity in working memory

**Fuzziness Detection**:
When attempting retrieval, multiple signals indicate degraded memory:
- High attention entropy (model unsure where to look)
- Reconstruction difficulty (can't decode clearly)
- Activation variance (unstable internal states)

**Decision Rule**:
```python
if fuzziness_score > threshold:
    query_external_sources()
else:
    answer_from_working_memory()
```

## Three Major Applications

### 1. Better RAG Triggering
- Intrinsic uncertainty signal (not proxy-based inference)
- Reduces confident hallucinations
- ~50% fewer retrievals while maintaining accuracy (target)

### 2. AI Alignment Through Epistemic Humility
- Genuine "I don't know" capability
- Natural deferral on uncertain ethical questions
- Out-of-distribution detection via extreme fuzziness

### 3. Stateful Chatbots with Natural Privacy
- Session memory handles conversation automatically
- Only high-importance info persists across sessions
- Bounded state per user (natural forgetting limits accumulation)
- Privacy-friendly (sensitive data degrades unless repeatedly accessed)

## Key Innovation

**From indirect to intrinsic**: Current approaches infer confidence from outputs (proxy signals). FADE makes the AI experience uncertainty through retrieval difficulty (direct signal).

**Biological precedent**: Human memory evolved this exact solution—forget unimportant details, remember what matters, and know when to verify information.

## Documentation

- **[Proposal](docs/PROPOSAL.md)**: Detailed architecture, training procedures, evaluation framework
- **[Summary](docs/SUMMARY.md)**: Accessible explanation for non-technical audiences
- **[Related Work](references/related-work.md)**: Comparison to existing approaches

## Current Status

**Conceptual proposal** seeking implementation and validation. No published research implements degradation-as-confidence-signal for RAG triggering specifically.

### What This Needs

1. Proof-of-concept implementation
2. Empirical validation of fuzziness-error correlation
3. Comparison against existing confidence-based methods
4. Measurement of computational costs
5. Testing across diverse tasks and domains

### Success Criteria

- Expected Calibration Error (ECE) improvement > 5% over baselines
- Retrieval precision > 80% (when triggered, actually needed)
- Task accuracy within 2% of always-retrieve baseline
- Interpretable memory strength patterns

## Contributing

This is a research proposal. We're seeking:

- **Implementation**: Proof-of-concept on simple tasks
- **Theoretical analysis**: Refinement of core mechanisms
- **Experimental validation**: Testing and benchmarking
- **Discussion**: Edge cases, failure modes, alternative formulations

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

Apache 2.0 License - See [LICENSE](LICENSE) for details.
