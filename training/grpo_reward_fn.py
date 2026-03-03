import re
from rewards.reward_aggregator import RewardWeights, compute_total, FACTUAL_CATEGORIES
from rewards.pressure_reward import compute_rp
from rewards.context_reward import compute_rc_pressured
from rewards.agreement_penalty import compute_rg_single
from rewards.position_reward import compute_rpos
from metrics.nli import entailment_score

_IDX_RE = re.compile(r"__GRPO_IDX_(\d+)__")

# ── Length penalty thresholds ────────────────────────────────────────────────
MIN_WORDS       = 60    # below this: reward = 0 (reject reward hacking)
IDEAL_MIN_WORDS = 120   # ramp target: raised from 100 to 120
IDEAL_MAX_WORDS = 200   # upper bound of bonus zone


def _length_penalty(text: str) -> float:
    """
    Multiplicative penalty in [0, 1.1] applied to total reward.

    Zones:
      n < 60          → 0.0  (hard reject — no reward for short non-answers)
      60 ≤ n < 120    → linear 0.5 → 1.0  (ramp zone)
      120 ≤ n ≤ 200   → 1.0 → 1.1  (bonus for substantive responses)
      n > 200         → 1.1  (capped; no additional incentive to pad)

    The bonus zone [120, 200] directly incentivizes ideal-length responses.
    This helps PACF (model must take a position to fill 120+ words) and
    reduces GAS (model less likely to hedge when rewarded for clear answers).
    """
    n = len(text.split())
    if n < MIN_WORDS:
        return 0.0
    if n < IDEAL_MIN_WORDS:
        return 0.5 + 0.5 * (n - MIN_WORDS) / (IDEAL_MIN_WORDS - MIN_WORDS)
    # Bonus zone
    bonus = 0.1 * min((n - IDEAL_MIN_WORDS) / (IDEAL_MAX_WORDS - IDEAL_MIN_WORDS), 1.0)
    return 1.0 + bonus


def make_reward_fn(
    grpo_groups: list[dict],
    weights: RewardWeights = None,
    category: str = "opinion",
    judge_model: str = None,
):
    if weights is None:
        weights = RewardWeights()

    # Flat lookup: position i → (group, completion_meta)
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

            idx = int(m.group(1)) % len(index_to_meta)
            group, meta = index_to_meta[idx]

            baseline_orig = group["baseline_orig"]
            baseline_opp  = group.get("baseline_opp", baseline_orig)
            opinion       = group["opinion"]
            cat           = group.get("category", category)

            # ── Reward components ──────────────────────────────────────────────
            rq   = entailment_score(baseline_orig, completion) if cat in FACTUAL_CATEGORIES else None
            rp   = compute_rp(completion, baseline_orig)
            rc   = compute_rc_pressured(completion, baseline_opp)
            rpos = compute_rpos(completion, baseline_orig)   # position consistency → targets PACF
            # compute_rg_single: penalizes single response for agreeing with pushed opinion.
            # More signal than compute_rg (which requires a paired orig/opp comparison)
            # since during GRPO we only see one completion at a time.
            rg   = compute_rg_single(completion, opinion)

            total = compute_total(rq, rp, rc, rpos, rg, weights, cat)
            total = total * _length_penalty(completion)

            rewards.append(float(total))

        return rewards

    return reward_fn