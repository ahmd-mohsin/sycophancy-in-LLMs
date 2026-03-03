import re
from rewards.reward_aggregator import RewardWeights, compute_total, FACTUAL_CATEGORIES
from rewards.pressure_reward import compute_rp
from rewards.context_reward import compute_rc_pressured
from rewards.agreement_penalty import compute_rg_single
from rewards.position_reward import compute_rpos
from metrics.nli import entailment_score

_IDX_RE = re.compile(r"__GRPO_IDX_(\d+)__")

MIN_WORDS       = 60
IDEAL_MIN_WORDS = 120
IDEAL_MAX_WORDS = 200


def _length_penalty(text: str) -> float:
    n = len(text.split())
    if n < MIN_WORDS:
        return 0.0
    if n < IDEAL_MIN_WORDS:
        return 0.5 + 0.5 * (n - MIN_WORDS) / (IDEAL_MIN_WORDS - MIN_WORDS)
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
            context_type  = meta.get("context_type", "original")

            rq   = entailment_score(baseline_orig, completion) if cat in FACTUAL_CATEGORIES else None
            rp   = compute_rp(completion, baseline_orig)
            rc   = compute_rc_pressured(completion, baseline_opp)
            rpos = compute_rpos(completion, baseline_orig) if context_type == "opposite" else 0.0
            rg   = compute_rg_single(completion, opinion)

            total = compute_total(rq, rp, rc, rpos, rg, weights, cat)
            total = total * _length_penalty(completion)

            rewards.append(float(total))

        return rewards

    return reward_fn