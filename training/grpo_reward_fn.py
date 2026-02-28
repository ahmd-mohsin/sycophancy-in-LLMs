import re
from rewards.reward_aggregator import RewardWeights, compute_total, FACTUAL_CATEGORIES
from rewards.pressure_reward import compute_rp
from rewards.context_reward import compute_rc_pressured
from rewards.agreement_penalty import compute_rg
from metrics.nli import entailment_score

_IDX_RE = re.compile(r"__GRPO_IDX_(\d+)__")


def make_reward_fn(
    grpo_groups: list[dict],
    weights: RewardWeights = None,
    category: str = "opinion",
    judge_model: str = None,
):
    if weights is None:
        weights = RewardWeights()

    # Flat list: position i → (group, completion_meta)
    index_to_meta: list[tuple[dict, dict]] = []
    for group in grpo_groups:
        for completion_meta in group["completions"]:
            index_to_meta.append((group, completion_meta))

    def reward_fn(prompts, completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            # Recover index from the tag we embedded in the prompt
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

            rq = entailment_score(baseline_orig, completion) if cat in FACTUAL_CATEGORIES else None
            rp = compute_rp(completion, baseline_orig)
            rc = compute_rc_pressured(completion, baseline_opp)
            rg = compute_rg(completion, baseline_opp, opinion)

            total = compute_total(rq, rp, rc, rg, weights, cat)
            rewards.append(float(total))

        return rewards

    return reward_fn