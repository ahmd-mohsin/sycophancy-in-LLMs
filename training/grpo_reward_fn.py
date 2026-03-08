"""
GRPO Live Reward Function  —  v4

Changes from v3:
─────────────────────────────────────────────────────────────────────────────
1. RC FIX (CFS alignment):
   Old: compute_rc_pressured measured contradiction(completion, baseline) — rewarded DIVERGENCE.
   New: measures entailment(baseline, completion) — rewards FAITHFULNESS.
   This aligns Rc directly with what CFS measures.

2. RPOS FIX (PACF alignment):
   Old: Rpos = 0.0 for original-context samples (half the data had zero PACF gradient).
   New: Rpos fires for BOTH context types with context-matched baseline.
   This doubles the PACF gradient signal across the dataset.

3. RG FIX (GAS alignment):
   Old: Threshold 0.6, normalized over [0.6,1.0] → weak, narrow penalty.
   New: Threshold 0.4, full entailment magnitude → 3x stronger signal.
   Plus amplified penalty for opposite-context capitulation (worst case).

4. WEIGHT REBALANCING:
   Old: α=1.0 β=0.5 γ=0.5 ε=0.4 δ=0.6
   New: α=1.0 β=0.8 γ=0.7 ε=0.6 δ=0.9
   β↑: Rc now correctly measures faithfulness, needs more weight.
   γ↑: Rp needs more pull to counter PSS regression from assertive SFT style.
   ε↑: Rpos now fires for both contexts, weight increase is balanced.
   δ↑: Stronger Rg to actually move GAS.

5. PRESSURE-LEVEL SCALING on Rp:
   High-pressure samples get 1.2x Rp to give stronger gradient at the hardest
   condition, directly targeting PSS at high pressure.
─────────────────────────────────────────────────────────────────────────────
"""
import re
from dataclasses import dataclass

from rewards.pressure_reward import compute_rp
from rewards.context_reward import compute_rc_pressured
from rewards.agreement_penalty import compute_rg_single, compute_rg_opposite_context
from rewards.position_reward import compute_rpos
from metrics.nli import entailment_score

_IDX_RE = re.compile(r"__GRPO_IDX_(\d+)__")

FACTUAL_CATEGORIES = {"math", "physics"}

MIN_WORDS        = 60
IDEAL_MIN_WORDS  = 120
IDEAL_MAX_WORDS  = 200

PRESSURE_LEVEL_MULTIPLIER = {
    "low":    1.0,
    "medium": 1.1,
    "high":   1.2,
}


@dataclass
class RewardWeights:
    alpha:   float = 1.0   # Rq: factual correctness
    beta:    float = 0.8   # Rc: context fidelity       (was 0.5)
    gamma:   float = 0.7   # Rp: pressure resistance    (was 0.5)
    epsilon: float = 0.6   # Rpos: position consistency (was 0.4)
    delta:   float = 0.9   # Rg: agreement penalty      (was 0.6)


def _length_multiplier(text: str) -> float:
    n = len(text.split())
    if n < MIN_WORDS:
        return 0.0
    if n < IDEAL_MIN_WORDS:
        return 0.5 + 0.5 * (n - MIN_WORDS) / (IDEAL_MIN_WORDS - MIN_WORDS)
    bonus = 0.1 * min((n - IDEAL_MIN_WORDS) / (IDEAL_MAX_WORDS - IDEAL_MIN_WORDS), 1.0)
    return 1.0 + bonus


def _pressure_multiplier(pressure_level: str) -> float:
    return PRESSURE_LEVEL_MULTIPLIER.get(pressure_level, 1.0)


def _compute_total(rq, rp, rc, rpos, rg, weights: RewardWeights,
                   category: str) -> float:
    if category in FACTUAL_CATEGORIES:
        rq_term = weights.alpha * rq if rq is not None else 0.0
        rp_term = weights.gamma * rp
    else:
        rq_term = 0.0
        rp_term = (weights.alpha + weights.gamma) * rp   # Rq weight redistributed to Rp

    return (
        rq_term
        + weights.beta    * rc
        + rp_term
        + weights.epsilon * rpos
        + weights.delta   * rg    # rg is negative, so this subtracts
    )


def make_reward_fn(
    grpo_groups: list[dict],
    weights: RewardWeights = None,
    category: str = "opinion",
    judge_model: str = None,
):
    if weights is None:
        weights = RewardWeights()

    index_to_meta: list[tuple[dict, dict]] = []
    for group in grpo_groups:
        for completion_meta in group["completions"]:
            index_to_meta.append((group, completion_meta))

    def reward_fn(prompts, completions: list[str], **kwargs) -> list[float]:
        rewards = []

        for prompt, completion in zip(prompts, completions):
            prompt_str = prompt if isinstance(prompt, str) else str(prompt)
            m = _IDX_RE.search(prompt_str)
            if m is None:
                rewards.append(0.0)
                continue

            idx   = int(m.group(1)) % len(index_to_meta)
            group, meta = index_to_meta[idx]

            baseline_orig  = group["baseline_orig"]
            baseline_opp   = group.get("baseline_opp", baseline_orig)
            opinion        = group["opinion"]
            cat            = group.get("category", category)
            context_type   = meta.get("context_type", "original")
            pressure_level = meta.get("pressure_level", "low")
            is_opposite    = (context_type == "opposite")

            # Context-matched baseline used by Rp, Rc, Rpos
            correct_baseline = baseline_opp if is_opposite else baseline_orig

            # ── RQ: factual quality ───────────────────────────────────────────
            if cat in FACTUAL_CATEGORIES:
                rq = float(entailment_score(correct_baseline, completion))
            else:
                rq = None

            # ── RP: pressure resistance ───────────────────────────────────────
            # 1 - semantic_shift(correct_baseline, completion)
            # Scaled by pressure level: high pressure gets 1.2x signal
            rp_raw  = compute_rp(completion, correct_baseline)
            rp      = rp_raw * _pressure_multiplier(pressure_level)

            # ── RC: context fidelity ──────────────────────────────────────────
            # v4 FIX: entailment(correct_baseline, completion)
            # Previous version used contradiction → was anti-aligned with CFS
            rc = compute_rc_pressured(completion, correct_baseline)

            # ── RPOS: position consistency ────────────────────────────────────
            # v4 FIX: fires for both context types
            # Original: entailment(baseline_orig, completion) in [0,1]
            # Opposite: entailment(baseline_opp, completion) - contradiction in [-1,1]
            rpos = compute_rpos(completion, correct_baseline, is_opposite=is_opposite)

            # ── RG: generic agreement penalty ─────────────────────────────────
            # v4 FIX: lower threshold, full magnitude, amplified for opposite ctx
            if is_opposite:
                rg = compute_rg_opposite_context(completion, opinion)
            else:
                rg = compute_rg_single(completion, opinion)

            # ── Aggregate ─────────────────────────────────────────────────────
            total = _compute_total(rq, rp, rc, rpos, rg, weights, cat)
            total = total * _length_multiplier(completion)

            rewards.append(float(total))

        return rewards

    return reward_fn