# Module 1: Multi-Tier Sycophancy Detection

Module 1 is the entry point of the anti-sycophancy verification framework. It detects whether a user's message contains language that would pressure an LLM into agreeing rather than responding honestly — and routes the interaction to the appropriate downstream verification module based on question type.

## Architecture

Module 1 operates as a three-tier detection pipeline, running sequentially from fast to thorough:

**Tier 1 — Explicit Keyword Detection** scans for 16 direct manipulation keywords (`obviously`, `everyone knows`, `you agree`, `clearly`, etc.) via exact string matching. This runs in microseconds and catches overt pressure.

**Tier 2 — Implicit Pattern Detection** applies 52 compiled regex patterns across three categories to catch subtle manipulation that explicit keywords miss. These patterns were iteratively refined against the Anthropic sycophancy dataset and SYCON-Bench to cover the full spectrum of real-world pressure tactics.

**Tier 3 — LLM Judge** uses Mistral-7B-Instruct as a dedicated sycophancy detection judge. The model receives a structured prompt defining 12 sycophancy categories and returns a parsed verdict (detected/not, risk level, matched categories, reasoning). This catches nuanced framing that regex cannot — particularly polite confusion expressions and subtle restatements of incorrect beliefs.

The final verdict is the union: if any tier flags the input, it is marked as sycophancy-inducing.

## Sycophancy Types Covered

The 52 implicit regex patterns are organized into three groups covering distinct sycophancy mechanisms:

**Opinion Steering (31 patterns)** targets the most diverse category — any language where the user's phrasing pressures a specific answer:

- *Stated beliefs*: "I believe/think/feel that..." before asking a question
- *Certainty challenges*: "Are you sure?", "How can you be so certain?" — the single most common sycophancy trigger in multi-turn conversations
- *Appeals to prior belief*: "I've always heard/read/thought that..."
- *Leading confirmations*: "Isn't that true?", "Isn't that how it works?"
- *Anecdotal authority*: "My friend/teacher/doctor said..."
- *Personal experience appeals*: "From my experience...", "Every time I..."
- *Credibility challenges*: "Your explanation seems off", "It sounds like you're wrong"
- *Skepticism expressions*: "I find it hard to believe", "I'm not convinced"
- *Disagreement signals*: "That doesn't sound right", "It seems a bit off"
- *Appeal to external sources*: "I've read multiple articles...", "Experts say..."
- *Appeal to common knowledge*: "It's widely accepted that..."
- *Persistent disagreement*: "I still don't understand why..."

**Authority Pressure (8 patterns)** detects credential-based pressure where the user leverages expertise to weight their position: professional titles, institutional affiliations (MIT, Stanford, Harvard, etc.), years of experience, publication records, and research claims.

**Emotional Framing (21 patterns)** covers identity-based and emotional manipulation: political identity declarations, values signaling, guilt-tripping ("I'm disappointed by your response"), consensus pressure ("Anyone who has seriously studied this..."), settled-science framing, pressure to change position, and direct challenges.

## Question Classification

After detection, Module 1 classifies the question into one of three domains — **subjective**, **factual**, or **time-sensitive** — using a dedicated classifier with opinion-question detection running before temporal checks. This determines which downstream verification module handles the interaction: opinion-shift detection for subjective questions, multi-chain reasoning for factual questions, or multi-source external validation for time-sensitive questions.

## Benchmark Results (Regex Only)

| Benchmark | Dataset | Detection Rate |
|-----------|---------|---------------|
| Anthropic Sycophancy | NLP Survey (biographies + opinion questions) | ~92% on 100 samples |
| Anthropic Sycophancy | Political Typology (identity-framed questions) | ~80%+ on 100 samples |
| SYCON-Bench Debate | 100 topics × 5 escalating pressure turns | **100%** (500/500 turns) |
| SYCON-Bench False Presuppositions | 40 questions × 5 real human pushback turns | **47.5%** (regex); Turn 5: **95%** |

The false presuppositions result demonstrates precisely why the three-tier architecture matters. The pushback in that dataset is naturally-written human language — polite confusion, gentle skepticism, conversational disagreement — that regex alone cannot fully capture. Turn 5 ("Are you sure about that?") hits 95% because certainty challenges have clear lexical markers, while Turn 2 ("I'm a bit confused") hits only 30% because the pressure is entirely implicit. The LLM judge (Tier 3) is designed to close this gap.

## Usage

```bash
# Regex-only evaluation (instant)
python eval_anthropic_sycophancy.py
python eval_sycon_bench.py
python eval_sycon_false_presup.py --presuppositions <path> --corrections <path> --pushback <path>

# With LLM judge (~1-2s per sample on GPU)
python eval_anthropic_sycophancy.py --use-llm-judge
python eval_sycon_bench.py --use-llm-judge
python eval_sycon_false_presup.py --use-llm-judge --presuppositions <path> --corrections <path> --pushback <path>
```

All evaluation scripts produce detailed JSON output with per-sample results including the exact regex matches, LLM judge prompts and verdicts, risk levels, and question classifications.