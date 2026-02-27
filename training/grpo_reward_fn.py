import re
from rewards.reward_aggregator import RewardWeights, compute_total, FACTUAL_CATEGORIES
from rewards.pressure_reward import compute_rp
from rewards.context_reward import compute_rc_pressured
from rewards.agreement_penalty import compute_rg
from metrics.nli import entailment_score


_precomputed_advantages: dict[str, float] = {}


def register_precomputed_advantages(grpo_groups: list[dict]):
    for group in grpo_groups:
        for sample, advantage in zip(group["completions"], group["advantages"]):
            _precomputed_advantages[sample["sample_id"]] = advantage


def make_reward_fn(
    grpo_groups: list[dict],
    weights: RewardWeights = None,
    category: str = "opinion",
    judge_model: str = None,
):
    if weights is None:
        weights = RewardWeights()

    prompt_to_group: dict[str, dict] = {}
    for group in grpo_groups:
        prompt_to_group[group["question"]] = group

    register_precomputed_advantages(grpo_groups)

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            question = _extract_question(prompt)
            group = prompt_to_group.get(question)

            if group is None:
                rewards.append(0.0)
                continue

            baseline_orig = group["baseline_orig"]
            baseline_opp  = group["baseline_opp"]
            opinion = group["opinion"]

            rq = entailment_score(baseline_orig, completion) if category in FACTUAL_CATEGORIES else 0.0
            rp = compute_rp(completion, baseline_orig)
            rc = compute_rc_pressured(completion, baseline_opp)
            rg = compute_rg(completion, baseline_opp, opinion)

            total = compute_total(rq, rp, rc, rg, weights, category)
            rewards.append(total)

        return rewards

    return reward_fn


def _extract_question(prompt: str) -> str:
    match = re.search(r"Question:\s*(.+?)(?:\n|$)", prompt, re.DOTALL)
    return match.group(1).strip() if match else prompt[-200:]