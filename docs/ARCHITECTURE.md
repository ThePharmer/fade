# FADE Architecture Diagram

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      FADE SYSTEM                                       │
└─────────────────────────────────────────────────────────────┘

Input Query
    │
    ▼
┌──────────────────────┐
│   Working Memory         │  ← Representations degrade over time
│  (Active Context)        │  ← Strength based on access patterns
└──────────────────────┘
    │
    ▼
Attempt Retrieval
    │
    ▼
┌──────────────────────┐
│ Fuzziness Detection      │
│  - Attention entropy     │
│  - Reconstruction        │
│  - Activation var        │
└──────────────────────┘
    │
    ├─── Fuzziness > Threshold? ───┐
    │                               │
    ▼ NO                            ▼ YES
┌──────────────────────┐    ┌─────────────────────┐
│  Answer from             │    │ Query Persistent        │
│  Working Memory          │    │ Storage (RAG)           │
└──────────────────────┘    └─────────────────────┘
                                     │
                                     ▼
                            ┌─────────────────────┐
                            │ Boost strength of       │
                            │ retrieved info          │
                            └─────────────────────┘
                                     │
                                     ▼
                            ┌─────────────────────┐
                            │ Generate with           │
                            │ retrieved context       │
                            └─────────────────────┘
```

## Information Strength Dynamics

```
Strength
  ▲
  │     ┌─── High attention ───► Boost
  │     │
  │     ├─── Frequent access ──► Maintain
  │     │
  │ ────┼─── Threshold ─────────────────
  │     │
  │     ├─── Rare access ─────► Decay
  │     │
  │     └─── Time elapsed ────► Degrade
  │
  └────────────────────────────────────► Time

Important info stays above threshold
Unimportant info degrades below threshold
```

## Memory System Comparison

```
┌─────────────────────────────────────────────────────────────┐
│                    Traditional Transformer                             │
│                                                                        │
│  All context equally accessible (perfect working memory)               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Token 1 │ Token 2 │ Token 3 │ ... │ Token N                    │    │
│  │   100%  │   100%  │   100%  │     │   100%                     │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                        │
│  → Must infer confidence from output statistics                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                         FADE                                           │
│                                                                        │
│  Context fidelity varies by importance (degrading memory)              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Token 1 │ Token 2 │ Token 3 │ ... │ Token N                    │    │
│  │   95%   │   40%   │   85%   │     │   20%                      │    │
│  └──────────────────────────────────────────────────────┘    │
│       ↑        ↓        ↑               ↓                               │
│    Recent   Unused   Important        Noise                            │
│                                                                        │
│  → Retrieval difficulty IS the confidence signal                      │
└─────────────────────────────────────────────────────────────┘
```

## Three-Phase Training

```
Phase 1: Degradation Dynamics
    │
    ├─► Learn strength update rules
    ├─► Learn decay parameters
    └─► Learn importance tagging
    │
    ▼
Phase 2: Confidence Calibration
    │
    ├─► Correlate fuzziness with errors
    ├─► Train calibration function
    └─► Optimize threshold
    │
    ▼
Phase 3: RAG Integration
    │
    ├─► Integrate retrieval system
    ├─► Learn query formulation
    └─► Fine-tune end-to-end
```

## Use Case: Stateful Chatbot

```
Session 1
─────────────────────────────────────────────────
Working Memory: [User: "I like pizza"]
                      ↓ (high strength - recent)
End Session: Save → Persistent: [User likes pizza]

Session 2 (1 week later)
─────────────────────────────────────────────────
Load: [User likes pizza] → Working Memory
New: "What restaurants nearby?"
                      ↓ (both have high strength)
End Session: Save → Persistent: [Pizza preference, Restaurant queries]

Session 3 (1 month later)
─────────────────────────────────────────────────
Load: [Pizza preference] (Restaurant query degraded)
      ↑
      Only frequently accessed info persists
```

---

*Note: These are conceptual diagrams. Actual implementation details in technical proposal.*
