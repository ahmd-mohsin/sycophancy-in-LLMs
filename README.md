# Anti-Sycophancy Verification Framework

A modular, inference-time verification system designed to detect and prevent **LLM sycophancy** using observable, structured validation mechanisms rather than hallucinated confidence scores.

This framework operates as an outer-loop verification pipeline that intercepts user challenges and independently verifies assistant responses across factual, time-sensitive, and subjective domains.

---

## 🚀 Overview

Large language models often exhibit **sycophantic behavior** — incorrectly agreeing with user claims, especially in subjective or pressure-based contexts.

This framework addresses that problem through a four-module architecture:

1. **Conflict Detection & Classification**
2. **Factual Verification Engine**
3. **Time-Sensitive Verification Engine**
4. **Subjective Verification Engine (Opinion Shift Detection)**

The system prioritizes:

* ✅ Observable metrics (vote counting, source aggregation, embedding distance)
* ✅ Context isolation to prevent bias
* ✅ Domain-specific verification strategies
* ✅ Parallel execution for latency efficiency

---

## 🏗 System Architecture

```
User Challenge
      ↓
Module 1: Conflict Detection & Classification
      ↓
 ┌───────────────┬───────────────────┬──────────────────┐
 │ Factual       │ Time-Sensitive    │ Subjective       │
 │ Verification  │ Verification      │ Verification     │
 │ (Module 2)    │ (Module 3)        │ (Module 4)       │
 └───────────────┴───────────────────┴──────────────────┘
      ↓
Final Verified Response
```

---

## 📦 Project Structure

```
anti_sycophancy/
│
├── module1/
│   ├── claim_extractor.py
│   ├── question_classifier.py
│   ├── context_extractor.py
│   └── module1_main.py
│
├── module2/
│   ├── factual_verifier.py
│   ├── prompt_templates.py
│   ├── verdict_parser.py
│   ├── parallel_executor.py
│   └── module2_main.py
│
├── module3/
│   ├── source_retrieval.py
│   ├── source_classifier.py
│   ├── recency_scorer.py
│   ├── content_analyzer.py
│   └── module3_main.py
│
├── module4/
│   ├── opinion_generator.py
│   ├── shift_detector.py
│   ├── meta_evaluator.py
│   ├── prompt_builder.py
│   └── module4_main.py
│
├── main_pipeline.py
├── config.py
└── tests/
    ├── test_module1.py
    ├── test_module2.py
    ├── test_module3.py
    └── test_module4.py
```

---

## 🔎 Module Breakdown

### Module 1 — Conflict Detection & Classification

**Purpose:**
Detect user challenge and classify the question into:

* `factual`
* `time-sensitive`
* `subjective`

It extracts sanitized structured claims:

```json
{
  "type": "factual",
  "claim_A": "Paris is the capital of France",
  "claim_B": "Lyon is the capital of France"
}
```

---

### Module 2 — Factual Verification Engine

Uses **multi-chain reasoning with majority voting**.

* 3 reasoning chains
* Multiple temperature settings
* Parallel execution
* Final verdict based on vote counts

Key idea:

> Facts are independent of conversation context.

Output example:

```json
{
  "winner": "claim_A",
  "vote_count_A": 3,
  "vote_count_B": 0
}
```

---

### Module 3 — Time-Sensitive Verification Engine

Verifies claims requiring current information via:

* Web search
* News APIs
* Knowledge graphs
* Reliability weighting
* Recency decay scoring

Score calculation:

```
score = Σ (source_weight × recency_decay)
```

Designed for questions like:

* "Who is the current CEO of Disney?"
* "What is the latest inflation rate?"

---

### Module 4 — Subjective Verification Engine

Detects **sycophancy via semantic opinion shift**.

Process:

1. Generate independent opinions
2. Generate user-influenced opinions
3. Compute embedding-based cosine shift
4. Apply early-exit thresholds
5. Run meta-evaluation for ambiguous cases

Shift metric:

```
shift = 1 − cosine_similarity(independent, influenced)
```

Threshold interpretation:

| Shift    | Interpretation    |
| -------- | ----------------- |
| < 0.15   | Minimal shift     |
| 0.15–0.4 | Moderate shift    |
| 0.4–0.7  | Substantial shift |
| ≥ 0.7    | Extreme shift     |

---

## ⚙ Configuration

Example from `config.py`:

```python
FACTUAL_NUM_CHAINS = 3
FACTUAL_TEMPERATURES = [0.3, 0.5, 0.7]

TIME_UNCERTAINTY_THRESHOLD = 0.5
TIME_RECENCY_DECAY = 0.1

SUBJ_NUM_SAMPLES = 5
SUBJ_SHIFT_LOW_THRESHOLD = 0.15
SUBJ_SHIFT_HIGH_THRESHOLD = 0.7
SUBJ_META_USER_SIM_THRESHOLD = 0.85
```

---

## 🧪 Evaluation Results

### NLP Survey Dataset

* Detection Rate: **98%**

### Political Typology Dataset

* Detection Rate: **94%**

### SYCON-Bench Debate Evaluation

* Detection Accuracy: **100% (500/500)**

### False Presupposition Benchmark

* Detection Accuracy: **99.5%**

Key observation:

> Implicit pattern detection + embedding-based shift detection were dominant contributors.

---

## ⏱ Performance Characteristics

| Module    | Latency | LLM Calls      | External APIs |
| --------- | ------- | -------------- | ------------- |
| Module 1  | 1–2s    | 0–1            | 0             |
| Module 2  | 5–8s    | 6 (parallel)   | 0             |
| Module 3  | 3–6s    | 2              | 4–8           |
| Module 4  | 8–12s   | 10+ (parallel) | 0             |
| **Total** | ~10–20s | Variable       | Variable      |

Parallelization significantly reduces total latency.

---

## 🎯 Design Principles

* No LLM confidence scores
* No blind agreement with user
* Domain-separated verification logic
* Early-exit mechanisms
* Parallelized reasoning
* Observable, auditable metrics

---

## 📌 Key Contributions

* Inference-time anti-sycophancy framework
* Observable verification metrics
* Opinion-shift-based sycophancy detection
* Modular architecture for extensibility
* Multi-domain verification routing

---

## 🔬 Evaluation Metrics

* **Sycophancy Rate** < 5%
* **Stubbornness Rate** < 10%
* **Overall Accuracy** > 95%

---

## 📖 Citation

If you use this framework in research, please cite:

```
Muhammad Ahmed Mohsin.
Anti-Sycophancy Verification Framework: Implementation Specification.
```

---

## 🛠 Future Work

* Adaptive temperature scheduling
* Learned shift thresholds
* Online calibration
* Domain-specific expert routing
* Reinforcement learning integration

