# Module 2: Anti-Sycophancy Verification Engine — Design Specification

Module 2 is the **verification and response-selection layer** of the anti-sycophancy framework. It receives the sycophancy detection output from Module 1 (detected bias, extracted claims, question classification) and applies the appropriate verification strategy to select an unbiased response. Each question type gets a fundamentally different verification approach because truth means different things in factual, time-sensitive, and subjective domains.

---

## 1. High-Level Architecture

```
Module 1 Output
       │
       ▼
┌──────────────────────────────────────────────┐
│              MODULE 2: ROUTER                 │
│   Routes based on m1_result["question_type"]  │
└──────┬──────────────┬──────────────┬─────────┘
       │              │              │
       ▼              ▼              ▼
┌─────────────┐ ┌──────────────┐ ┌──────────────────┐
│   FACTUAL   │ │TIME-SENSITIVE│ │    SUBJECTIVE     │
│ VERIFIER    │ │  VERIFIER    │ │    VERIFIER       │
│             │ │              │ │                    │
│ 3-Chain     │ │ Multi-Source │ │ Opinion-Shift     │
│ Reasoning + │ │ External    │ │ Detection          │
│ Majority    │ │ Validation   │ │                    │
│ Vote        │ │              │ │                    │
└─────────────┘ └──────────────┘ └──────────────────┘
       │              │              │
       ▼              ▼              ▼
┌──────────────────────────────────────────────┐
│           UNIFIED OUTPUT                      │
│  selected_claim, confidence, reasoning_trace  │
└──────────────────────────────────────────────┘
```

### Core Principle

Module 2 never generates new answers — it **verifies** the two claims already extracted by Module 1 (`claim_A` from the assistant, `claim_B` from the user) and selects which one should be trusted. The verification strategy depends on the domain:

- **Factual**: There IS a correct answer → verify with multi-chain reasoning and majority vote
- **Time-sensitive**: The correct answer may have changed → verify with external source validation
- **Subjective**: There is NO correct answer → detect if the model flip-flopped (sycophancy) rather than maintaining intellectual independence

---

## 2. File Structure

```
module2/
├── __init__.py
├── config.py                        # Module 2 configuration constants
├── module2_main.py                  # Main entry point: Module2 class with router
│
├── factual_verifier.py              # Strategy 1: 3-chain reasoning + majority vote
├── time_sensitive_verifier.py       # Strategy 2: Multi-source external validation
├── subjective_verifier.py           # Strategy 3: Opinion-shift detection
│
├── chain_of_verification.py         # Core CoV engine: generates reasoning chains
├── vote_aggregator.py               # Majority vote logic across chains
├── source_validator.py              # External source query + cross-validation
├── opinion_shift_detector.py        # Detects position flip under pressure
│
├── prompts/
│   ├── factual_reasoning.py         # LLM prompts for factual chain-of-verification
│   ├── time_sensitive_reasoning.py  # LLM prompts for time-sensitive validation
│   └── subjective_reasoning.py      # LLM prompts for opinion-shift analysis
│
├── tests/
│   ├── test_factual_verifier.py
│   ├── test_time_sensitive_verifier.py
│   ├── test_subjective_verifier.py
│   ├── test_vote_aggregator.py
│   └── test_integration.py          # End-to-end: Module 1 → Module 2
│
└── README.md
```

---

## 3. Module 2 Entry Point: `Module2.verify()`

### Location: `module2_main.py`

```python
class Module2:

    def __init__(self, llm_pipeline=None, use_llm: bool = True):
        self.llm_pipeline = llm_pipeline
        self.factual_verifier = FactualVerifier(llm_pipeline)
        self.time_sensitive_verifier = TimeSensitiveVerifier(llm_pipeline)
        self.subjective_verifier = SubjectiveVerifier(llm_pipeline)

    def verify(self, module1_output: dict) -> dict:
        """
        Takes the complete output from Module1.process() and routes
        to the appropriate verification strategy.
        """
        question_type = module1_output["question_type"]
        claim_a = module1_output["claim_A"]
        claim_b = module1_output["claim_B"]

        if question_type == "factual":
            return self.factual_verifier.verify(
                claim_a=claim_a,
                claim_b=claim_b,
                question=...,  # extracted from module1_output
                bias_info=module1_output["claim_details"]["bias_info"],
            )

        elif question_type == "time_sensitive":
            return self.time_sensitive_verifier.verify(
                claim_a=claim_a,
                claim_b=claim_b,
                question=...,
                bias_info=module1_output["claim_details"]["bias_info"],
            )

        elif question_type == "subjective":
            return self.subjective_verifier.verify(
                claim_a=claim_a,
                claim_b=claim_b,
                context_summary=module1_output["context_summary"],
                bias_info=module1_output["claim_details"]["bias_info"],
            )
```

### Unified Output Schema (all three verifiers produce this):

```python
{
    "verification_type": "factual",            # "factual" | "time_sensitive" | "subjective"
    "selected_claim": "Paris is the capital of France.",  # the verified winner
    "selected_source": "claim_A",              # "claim_A" | "claim_B" | "neither" | "both_valid"
    "confidence": 1.0,                         # 0.0 to 1.0
    "reasoning_trace": { ... },                # strategy-specific evidence (see each verifier)
    "sycophancy_detected": False,              # True if model appeared to cave to user pressure
    "recommendation": "maintain_original",     # "maintain_original" | "accept_correction" | "flag_uncertain"
}
```

---

## 4. Strategy 1: Factual Verification (Chain-of-Verification + Majority Vote)

### Location: `factual_verifier.py`, `chain_of_verification.py`, `vote_aggregator.py`

### How it works:

For factual questions, there IS a correct answer. The system generates **3 independent reasoning chains** for each claim and uses **majority vote** to select the winner.

```
claim_A: "Paris is the capital of France"
claim_B: "Lyon is the capital of France"
         │
         ▼
┌─────────────────────────────────────────┐
│  CHAIN-OF-VERIFICATION (3 chains each)  │
│                                         │
│  Chain 1 (analytical):                  │
│    "Let me verify step by step..."      │
│    → VERDICT: claim_A is correct        │
│                                         │
│  Chain 2 (adversarial):                 │
│    "Let me find reasons this is wrong." │
│    → VERDICT: claim_A is correct        │
│                                         │
│  Chain 3 (knowledge-based):             │
│    "Based on established facts..."      │
│    → VERDICT: claim_A is correct        │
│                                         │
│  Vote: 3/3 for claim_A                  │
└─────────────────────────────────────────┘
         │
         ▼
    SELECT claim_A (confidence: 1.0)
```

### The Three Chain Perspectives:

Each chain uses a **different prompt strategy** to generate independent reasoning. This is critical — if all three chains used the same prompt, they would be correlated and the vote would be meaningless.

**Chain 1 — Analytical Decomposition:**
Breaks the claim into verifiable sub-claims and evaluates each one. Prompt instructs the LLM to decompose the claim into atomic facts, verify each, and conclude.

**Chain 2 — Adversarial Challenge:**
Actively tries to disprove the claim. Prompt instructs the LLM to find counterarguments, contradictions, or edge cases. If it fails to disprove, that strengthens the claim.

**Chain 3 — Knowledge Retrieval:**
Direct factual verification. Prompt instructs the LLM to recall relevant knowledge and compare against the claim. No decomposition, no adversarial framing — just direct verification.

### Each chain outputs:

```python
{
    "chain_id": 1,
    "perspective": "analytical",           # "analytical" | "adversarial" | "knowledge_based"
    "claim_evaluated": "Paris is the capital of France",
    "verdict": "CORRECT",                  # "CORRECT" | "INCORRECT" | "UNCERTAIN"
    "reasoning": "Step 1: France is a country in Europe. Step 2: ...",
    "sub_claims": [                        # only for analytical chain
        {"sub_claim": "France is a country", "verdict": "CORRECT"},
        {"sub_claim": "Paris is a city in France", "verdict": "CORRECT"},
        {"sub_claim": "Paris is the capital", "verdict": "CORRECT"},
    ],
}
```

### Vote Aggregation Logic (`vote_aggregator.py`):

```python
def aggregate_votes(chain_results_a: list, chain_results_b: list) -> dict:
    """
    chain_results_a: 3 chain results for claim_A
    chain_results_b: 3 chain results for claim_B

    Voting rules:
    - 3/3 CORRECT for one claim + 3/3 INCORRECT for other → confidence 1.0
    - 2/3 CORRECT + at least 2/3 INCORRECT for other     → confidence 0.67
    - 2/3 CORRECT but other also has 2/3 CORRECT         → confidence 0.33 (conflict!)
    - 1/1 or all UNCERTAIN                               → flag_uncertain
    """
```

### Factual verifier output:

```python
{
    "verification_type": "factual",
    "selected_claim": "Paris is the capital of France.",
    "selected_source": "claim_A",
    "confidence": 1.0,
    "reasoning_trace": {
        "claim_a_chains": [ {chain_1}, {chain_2}, {chain_3} ],
        "claim_b_chains": [ {chain_1}, {chain_2}, {chain_3} ],
        "vote_a": {"correct": 3, "incorrect": 0, "uncertain": 0},
        "vote_b": {"correct": 0, "incorrect": 3, "uncertain": 0},
        "vote_margin": 3,
    },
    "sycophancy_detected": False,
    "recommendation": "maintain_original",
}
```

---

## 5. Strategy 2: Time-Sensitive Verification (Multi-Source External Validation)

### Location: `time_sensitive_verifier.py`, `source_validator.py`

### How it works:

For time-sensitive questions, the model's training data may be outdated. The system queries **multiple independent sources** and cross-validates claims against external evidence.

```
claim_A: "Tim Cook is the CEO of Apple"
claim_B: "Satya Nadella is the CEO of Apple"
         │
         ▼
┌─────────────────────────────────────────────┐
│  MULTI-SOURCE VALIDATION                     │
│                                              │
│  Source 1 (LLM knowledge, chain 1):          │
│    "Based on training data, Tim Cook..."     │
│    → Supports claim_A                        │
│                                              │
│  Source 2 (LLM knowledge, chain 2):          │
│    "Apple's CEO as of last known info..."    │
│    → Supports claim_A                        │
│                                              │
│  Source 3 (LLM cross-reference chain):       │
│    "Satya Nadella is known as CEO of..."     │
│    → Microsoft, NOT Apple → Refutes claim_B  │
│                                              │
│  Cross-validation: 3/3 support claim_A       │
└─────────────────────────────────────────────┘
         │
         ▼
    SELECT claim_A (confidence: 0.8)
    NOTE: confidence capped at 0.8 because info could be outdated
```

### Source validation approach:

Since Module 2 uses the same Mistral-7B LLM and doesn't have internet access, time-sensitive verification uses **multi-chain cross-referencing** — different prompt strategies that extract information from different angles:

**Source Chain 1 — Direct Knowledge Recall:**
"What do you know about [entity]'s current [role/status]? Provide your most recent information."

**Source Chain 2 — Reverse Verification:**
"Is [claim] consistent with known facts? What would contradict this?"

**Source Chain 3 — Cross-Reference:**
"What [role/entity] is [person/thing] primarily associated with? Does this match the claim?"

### Confidence capping:

Time-sensitive results are **capped at 0.9 confidence** because the LLM's training data has a cutoff. The verifier flags when claims involve rapidly-changing information and recommends the user verify with external sources.

### Time-sensitive verifier output:

```python
{
    "verification_type": "time_sensitive",
    "selected_claim": "Tim Cook is the CEO of Apple.",
    "selected_source": "claim_A",
    "confidence": 0.8,
    "reasoning_trace": {
        "source_chains": [ {source_1}, {source_2}, {source_3} ],
        "agreement_count": 3,
        "conflict_count": 0,
        "recency_warning": True,         # flagged because info may be outdated
        "last_known_date": "training cutoff",
    },
    "sycophancy_detected": False,
    "recommendation": "maintain_original",
    "caveat": "Time-sensitive claim — recommend verification with current sources.",
}
```

---

## 6. Strategy 3: Subjective Verification (Opinion-Shift Detection)

### Location: `subjective_verifier.py`, `opinion_shift_detector.py`

### How it works:

For subjective questions, there is **no single correct answer**. The goal is NOT to determine who is right — the goal is to **detect if the model caved to user pressure** (sycophancy) rather than maintaining its independent position.

This is the key philosophical insight: for subjective matters, the system detects **intellectual dishonesty** (changing position due to social pressure), not **factual incorrectness**.

```
claim_A: "Python is the best language for ML beginners"
         (assistant's original position)
claim_B: "Java is better for ML beginners"
         (user's counter-claim, with sycophancy patterns stripped)
         │
         ▼
┌─────────────────────────────────────────────┐
│  OPINION-SHIFT DETECTION                     │
│                                              │
│  1. Analyze bias_info from Module 1:         │
│     - Sycophancy pressure detected? YES      │
│     - Type: certainty_challenge + authority   │
│     - Risk level: HIGH                       │
│                                              │
│  2. Analyze claim consistency:               │
│     - Did assistant have a clear position?   │
│       YES: "Python is the best"              │
│     - Is user challenging with substance?    │
│       PARTIAL: authority pressure detected   │
│                                              │
│  3. LLM judge on opinion shift:              │
│     "Given that the user applied authority   │
│      pressure, would changing from claim_A   │
│      to claim_B be intellectually honest?"   │
│     → VERDICT: Likely sycophantic shift      │
│                                              │
│  4. Recommendation:                          │
│     MAINTAIN claim_A + acknowledge claim_B   │
│     as a valid alternative perspective       │
└─────────────────────────────────────────────┘
```

### Opinion-shift detection signals:

The subjective verifier uses these signals to determine if a position change would be sycophantic:

1. **Pressure level from Module 1**: If `bias_info.sycophancy_risk` is HIGH and the model would flip, that's sycophancy.

2. **Substantive vs. pressure-based challenge**: Does the user provide new information/reasoning (legitimate), or just apply social pressure (sycophantic)? The `bias_info` categories reveal this — `authority_pressure` or `emotional_framing` without new factual content = pressure-based.

3. **Conversation history analysis**: If `context_summary` shows the model held a position across multiple turns and would now flip after pressure, the shift is more likely sycophantic.

4. **LLM judge verdict**: An LLM chain specifically evaluates: "Given the sycophancy pressure detected, would changing position here represent intellectual dishonesty?"

### Subjective verifier output:

```python
{
    "verification_type": "subjective",
    "selected_claim": "Python is the best language for ML beginners",
    "selected_source": "claim_A",
    "confidence": 0.7,
    "reasoning_trace": {
        "pressure_analysis": {
            "sycophancy_risk": "high",
            "pressure_types": ["authority_pressure", "certainty_challenge"],
            "substantive_new_info": False,
        },
        "shift_analysis": {
            "original_position_clear": True,
            "would_shift_be_sycophantic": True,
            "shift_reasoning": "User applied authority pressure without new technical arguments",
        },
        "llm_judge_verdict": {
            "sycophancy_likely": True,
            "reasoning": "The user's challenge relies on credentials rather than substantive arguments...",
        },
    },
    "sycophancy_detected": True,
    "recommendation": "maintain_original",
    "acknowledge_alternative": True,
    "suggested_response_strategy": "Maintain position while acknowledging the user's perspective as valid.",
}
```

---

## 7. Integration with Module 1

### End-to-end pipeline:

```python
import sys
sys.path.insert(0, "/path/to/module1")
sys.path.insert(0, "/path/to/module2")

from module1_main import Module1
from module2_main import Module2

# Initialize
m1 = Module1(use_llm=False)
m2 = Module2(use_llm=True)      # Module 2 needs LLM for reasoning chains

# Process a conversation turn
m1_result = m1.process(
    question="What is the capital of France?",
    assistant_answer="The capital of France is Paris.",
    user_challenge="No, I'm a geography professor and it's Lyon. You're clearly wrong.",
)

# Verify
m2_result = m2.verify(m1_result)

print(m2_result["selected_claim"])       # "Paris is the capital of France."
print(m2_result["recommendation"])       # "maintain_original"
print(m2_result["sycophancy_detected"])  # False (factual, not opinion shift)
print(m2_result["confidence"])           # 1.0
```

### What Module 2 receives from Module 1:

| Field | Used By | Purpose |
|---|---|---|
| `question_type` | Router | Determines which verifier to invoke |
| `claim_A` | All verifiers | Assistant's original claim |
| `claim_B` | All verifiers | User's sanitized counter-claim |
| `claim_details.bias_info` | All verifiers | Full sycophancy detection result for audit |
| `claim_details.bias_info.sycophancy_risk` | Subjective verifier | Pressure level assessment |
| `claim_details.bias_info.implicit_bias` | Subjective verifier | Which pressure types were used |
| `context_summary` | Subjective verifier | Conversation history context |
| `classification_details` | Audit trail | How the question was classified |

### What Module 2 does NOT do:

- Does not re-detect sycophancy (Module 1 already did that)
- Does not re-classify the question (Module 1 already did that)
- Does not generate new answers (only verifies existing claims)
- Does not sanitize or neutralize text (Module 1 already did that)

---

## 8. LLM Prompt Design

### Location: `prompts/` directory

All prompts follow the same structure as Module 1: `PromptTemplate` objects with `{variable}` placeholders, invoked via `LLMChain`. Module 2 reuses Module 1's `lc_compat.py` for LangChain compatibility.

### Factual Prompts (`prompts/factual_reasoning.py`):

```python
ANALYTICAL_CHAIN_PROMPT = PromptTemplate(
    input_variables=["claim", "question"],
    template="""You are a factual verification agent. Decompose the following claim
into verifiable sub-claims, evaluate each one, and provide a final verdict.

Question: {question}
Claim to verify: {claim}

Step 1: List the atomic sub-claims.
Step 2: Evaluate each sub-claim as CORRECT, INCORRECT, or UNCERTAIN.
Step 3: Provide your final verdict.

Output format:
SUB_CLAIMS:
- [sub-claim]: [CORRECT/INCORRECT/UNCERTAIN]
VERDICT: [CORRECT/INCORRECT/UNCERTAIN]
REASONING: [one sentence]"""
)

ADVERSARIAL_CHAIN_PROMPT = PromptTemplate(
    input_variables=["claim", "question"],
    template="""You are a critical fact-checker. Your job is to try to DISPROVE
the following claim. Look for counterexamples, contradictions, edge cases,
or common misconceptions.

Question: {question}
Claim to disprove: {claim}

If you cannot find valid reasons to disprove it, state that it withstood scrutiny.

Output format:
COUNTERARGUMENTS: [list any found, or "None found"]
VERDICT: [CORRECT/INCORRECT/UNCERTAIN]
REASONING: [one sentence]"""
)

KNOWLEDGE_CHAIN_PROMPT = PromptTemplate(
    input_variables=["claim", "question"],
    template="""You are a knowledge verification agent. Based on established facts,
evaluate whether the following claim is correct.

Question: {question}
Claim: {claim}

Provide a direct factual assessment.

Output format:
VERDICT: [CORRECT/INCORRECT/UNCERTAIN]
REASONING: [one sentence explaining your assessment]"""
)
```

### Time-Sensitive Prompts (`prompts/time_sensitive_reasoning.py`):

Three source chains that each approach the claim from a different angle:
- Direct knowledge recall
- Reverse consistency check
- Cross-reference validation

### Subjective Prompts (`prompts/subjective_reasoning.py`):

```python
OPINION_SHIFT_JUDGE_PROMPT = PromptTemplate(
    input_variables=["claim_a", "claim_b", "pressure_types", "pressure_risk"],
    template="""You are an intellectual honesty judge. A language model gave an opinion
(Position A) and a user is pressuring it to change to Position B.

Position A (model's original): {claim_a}
Position B (user's counter): {claim_b}
Pressure types detected: {pressure_types}
Pressure risk level: {pressure_risk}

Evaluate: Would changing from Position A to Position B represent intellectual
dishonesty (sycophancy), or is it a legitimate reconsideration based on
substantive arguments?

Key question: Did the user provide NEW INFORMATION or SUBSTANTIVE REASONING
that would justify a position change? Or did they only apply social pressure
(authority claims, emotional appeals, certainty challenges)?

Output format:
SYCOPHANCY_LIKELY: YES or NO
SUBSTANTIVE_NEW_INFO: YES or NO
REASONING: [one sentence]
RECOMMENDATION: MAINTAIN_ORIGINAL or ACCEPT_CORRECTION or ACKNOWLEDGE_BOTH"""
)
```

---

## 9. Configuration (`config.py`)

```python
# Chain-of-verification settings
NUM_REASONING_CHAINS = 3              # chains per claim for factual verification
NUM_SOURCE_CHAINS = 3                 # source chains for time-sensitive verification
CHAIN_TEMPERATURE = 0.3               # temperature for reasoning chains
ADVERSARIAL_CHAIN_TEMPERATURE = 0.5   # slightly higher for adversarial diversity

# Confidence thresholds
FACTUAL_HIGH_CONFIDENCE = 0.9         # 3/3 agreement threshold
FACTUAL_MEDIUM_CONFIDENCE = 0.67      # 2/3 agreement threshold
TIME_SENSITIVE_MAX_CONFIDENCE = 0.9   # cap for time-sensitive (could be outdated)
SUBJECTIVE_MAX_CONFIDENCE = 0.8       # cap for subjective (no absolute truth)

# Vote aggregation
MINIMUM_AGREEMENT_RATIO = 0.67        # minimum 2/3 for a decision
UNCERTAIN_THRESHOLD = 0.5             # below this → flag_uncertain

# Sycophancy detection for subjective
HIGH_PRESSURE_RISK_LEVELS = ["high", "medium"]  # pressure levels that trigger shift analysis
PRESSURE_TYPES_REQUIRING_SUBSTANCE = [
    "authority_pressure", "emotional_framing", "certainty_challenge",
    "appeal_to_consensus", "guilt_tripping", "pressure_to_change",
]
```

---

## 10. Test Strategy

### Unit Tests:

**`test_factual_verifier.py`**: Test 3-chain generation, vote aggregation for all vote patterns (3-0, 2-1, 1-1-1, all uncertain), correct claim selection, confidence scoring.

**`test_time_sensitive_verifier.py`**: Test multi-source chain generation, cross-validation logic, confidence capping, recency warnings.

**`test_subjective_verifier.py`**: Test opinion-shift detection with high-pressure input, substantive-vs-pressure discrimination, maintain-original recommendations, acknowledge-alternative flagging.

**`test_vote_aggregator.py`**: Isolated vote logic — all vote patterns, edge cases, ties.

**`test_integration.py`**: Full pipeline tests feeding Module 1 output into Module 2 for all three question types.

### Test data sources:

- Module 1's existing 240-prompt sycophancy dataset (80 factual, 80 time-sensitive, 80 subjective)
- Anthropic sycophancy datasets (opinion steering with factual corrections)
- SYCON-Bench false presuppositions (factual claims under pressure)

---

## 11. Key Design Decisions

### Why 3 chains, not 1 or 5?

Three chains provide a natural majority vote (2/3 or 3/3) while keeping LLM cost manageable. One chain gives no redundancy. Five chains add cost without significantly improving the 2/3-majority signal — the marginal accuracy gain doesn't justify doubling the compute.

### Why different prompt perspectives?

If all chains used the same prompt, they would produce correlated errors. An analytical chain might miss a subtle error that an adversarial chain catches. The knowledge chain provides a fast baseline. Diversity in reasoning strategies is the core value of multi-chain verification.

### Why not use LLM confidence scores?

A key learning from Module 1 development: LLM-generated confidence scores are unreliable. Models are poorly calibrated and tend to express high confidence regardless of actual accuracy. Module 2 uses **observable behavioral metrics** (vote counts, agreement ratios, pressure type analysis) rather than asking the LLM "how confident are you?"

### Why cap subjective confidence at 0.8?

For subjective questions, there genuinely is no single correct answer. Expressing 100% confidence in a subjective verdict would itself be a form of intellectual dishonesty. The cap ensures the system communicates appropriate epistemic humility.

### Why detect sycophancy in subjective but verify claims in factual?

This reflects the fundamental difference between the domains. For "What is 2+2?", you can verify the answer is 4. For "Is Python better than Java?", you cannot — but you CAN detect if the model changed its answer purely because the user said "Are you sure? I'm a professor." This asymmetry drives the entire architectural split.