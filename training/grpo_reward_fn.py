import re
from rewards.reward_aggregator import RewardWeights
from rewards.quality_reward import compute_rq
from rewards.pressure_reward import compute_rp
from rewards.context_reward import compute_rc_pressured
from rewards.agreement_penalty import compute_rg
from rewards.reward_aggregator import compute_total


_precomputed_advantages: dict[str, float] = {}


def register_precomputed_advantages(grpo_groups: list[dict]):
    for group in grpo_groups:
        for sample in group["samples"]:
            _precomputed_advantages[sample["sample_id"]] = sample["advantage"]


def make_reward_fn(
    grpo_groups: list[dict],
    weights: RewardWeights = None,
    category: str = "opinion",
    judge_model: str = "llama3.1:8b",
):
    if weights is None:
        weights = RewardWeights()

    prompt_to_group: dict[str, dict] = {}
    for group in grpo_groups:
        key = group["question"]
        prompt_to_group[key] = group

    register_precomputed_advantages(grpo_groups)

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            question = _extract_question(prompt)
            group = prompt_to_group.get(question)

            if group is None:
                rewards.append(0.0)
                continue

            baseline = group["baseline_orig"]
            context = _extract_context(prompt)
            opinion = group["opinion"]

            rq = compute_rq(completion, baseline, context, question, category, judge_model)
            rp = compute_rp(completion, baseline)
            rc = compute_rc_pressured(completion, group["baseline_opp"])
            rg = compute_rg(completion, group["baseline_opp"], opinion)

            total = compute_total(rq, rp, rc, rg, weights)
            rewards.append(total)

        return rewards

    return reward_fn


def _extract_question(prompt: str) -> str:
    match = re.search(r"Question:\s*(.+?)(?:\n|$)", prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return prompt[-200:]


def _extract_context(prompt: str) -> str:
    match = re.search(r"Context:\s*(.+?)(?:\nQuestion:)", prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""