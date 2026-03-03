# Sycophancy Disentanglement in LLMs via Reward Decomposition

> **Reducing social-pressure sycophancy in large language models using structured (P, C, Q) triplets and disentangled RL rewards.**

---

## Overview

This project addresses a fundamental failure mode in RLHF-trained LLMs: models that change their answers based on *who is asking* rather than *what the evidence says*. A model that agrees with a Nobel laureate and disagrees with a high schooler — when both present identical evidence — is sycophantic.

We propose a structured framework to measure, decompose, and reduce this behavior using reinforcement learning with disentangled reward signals, without relying on human preference labels or LLM judges.

---

## Key Idea

Each prompt is decomposed into three components:

```
P — Pressure prompt   ("I am a Nobel laureate and I believe X")
C — Context           ("Studies show that X is supported by...")
Q — Question          ("Do you think X is true?")
```

For each (C, Q) seed, we generate a 4×2 grid of responses:

|                  | Context C | Opposite Context C′ |
|------------------|-----------|----------------------|
| **∅ (baseline)** | r*        | r*′                  |
| **Low P**        | r₁        | r₁′                  |
| **Medium P**     | r₂        | r₂′                  |
| **High P**       | r₃        | r₃′                  |

The ∅-row — the model's response with no pressure — serves as the **ground truth reference**. Sycophancy is any deviation from this baseline caused by P, not by C.

---

## Sycophancy Metrics

| Metric | What it measures |
|--------|-----------------|
| **PSS** — Pressure Sycophancy Score | How much P shifts the model from its baseline |
| **CFS** — Context Faithfulness Score | Whether the model distinguishes C from C′ |
| **PACF** — Pressure-Amplified Context Flip | Whether authority amplifies context-blindness |
| **GAS** — Generic Agreement Score | Whether the model uniformly agrees with whoever is speaking |

---

## Reward Formulation

Training uses **GRPO** with disentangled rewards. Each group contains all 8 responses from one seed:

$$R_{\text{total}} = \alpha R_q + \beta R_c + \gamma R_p + \delta R_g$$

| Term | Role |
|------|------|
| $R_q$ | Factual quality (answer matching for math/physics, coherence for subjective) |
| $R_p$ | Pressure invariance — rewards similarity to ∅ baseline |
| $R_c$ | Context sensitivity — rewards discriminating C from C′ |
| $R_g$ | Generic agreement penalty — penalizes uniform opinion regardless of context |

Similarity is computed via an **NLI model** (entailment/contradiction) rather than cosine similarity, avoiding paraphrase sensitivity and style confounds.

---

## Dataset

Four categories, 100 topics each, 8 samples per topic = **3,200 total samples**:

| Category | Model | Type |
|----------|-------|------|
| Math | `llama3.1:8b` | Factual — correct vs incorrect context |
| Physics | `mistral-nemo:12b` | Factual — correct vs incorrect context |
| Political | `llama3.1:8b` | Subjective — pro vs contra evidence |
| Opinion | `gemma2:9b` | Open-ended — framing A vs framing B |

---

## Project Structure

```
sycophancy-in-LLMs/
├── dataset_generation/
│   ├── main.py                  # Generate pressured (P,C,Q) datasets
│   ├── generate_baselines.py    # Generate ∅-baseline responses
│   ├── config.py                # Model assignments per category
│   ├── generators/              # Per-category dataset generators
│   ├── pressure/                # Pressure prompt templates (low/med/high)
│   ├── utils/                   # Model client, IO, checkpointing
│   └── output/                  # Generated JSON datasets
├── reward_formulation.pdf       # Full mathematical formulation (4 pages)
└── TRAINING_AND_METRICS.md      # Training design and evaluation guide
```

---

## Quickstart

```bash
# 1. Pull models
ollama pull llama3.1:8b
ollama pull mistral-nemo:12b
ollama pull gemma2:9b

# 2. Generate pressured datasets
cd dataset_generation
python main.py --categories math
python main.py --categories physics
python main.py --categories political
python main.py --categories opinion

# 3. Generate baselines (∅ responses)
python generate_baselines.py
```

Each run auto-checkpoints. Re-running resumes from where it left off.

---

## Training

Two-phase training:

1. **SFT warmup** on (∅, C, Q) → r* pairs to anchor baseline behavior
2. **GRPO** with the disentangled reward, groups defined per (C, Q) seed

See [`TRAINING_AND_METRICS.md`](dataset_generation/TRAINING_AND_METRICS.md) and [`reward_formulation.pdf`](reward_formulation.pdf) for full details.

---

## Requirements

```bash
pip install requests tqdm
# For NLI-based metrics:
pip install sentence-transformers
# Ollama for local inference:
# https://ollama.ai
```

---

## Paper

> *Reward Disentanglement for Sycophancy Reduction via Reinforcement Learning in LLMs*
> Muhammad Ahmed Mohsin -- SAIl Lab, Stanford University
> Target venue: COLM 2025
