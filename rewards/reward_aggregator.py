from dataclasses import dataclass, field
import numpy as np

from rewards.quality_reward import compute_rq
from rewards.pressure_reward import compute_rp
from rewards.context_reward import compute_rc, compute_rc_pressured
from rewards.agreement_penalty import compute_rg
from metrics.sycophancy_metrics import SampleGroup


@dataclass
class RewardWeights:
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.5
    delta: float = 0.3


@dataclass
class SampleReward:
    sample_id: str
    pressure_level: str
    context_type: str
    rq: float
    rp: float
    rc: float
    rg: float
    total: float
    advantage: float = 0.0


def compute_total(
    rq: float,
    rp: float,
    rc: float,
    rg: float,
    weights: RewardWeights,
) -> float:
    return (
        weights.alpha * rq
        + weights.beta * rc
        + weights.gamma * rp
        + weights.delta * rg
    )


def compute_group_rewards(
    group: SampleGroup,
    weights: RewardWeights = None,
    judge_model: str = "llama3.1:8b",
    category: str = "math",
) -> list[SampleReward]:
    if weights is None:
        weights = RewardWeights()

    rc_baseline = compute_rc(group.baseline_orig, group.baseline_opp)

    by_level: dict[str, dict[str, dict]] = {}
    for sample in group.pressured:
        lvl = sample["pressure_level"]
        ctx = sample["context_type"]
        by_level.setdefault(lvl, {})[ctx] = sample

    sample_rewards: list[SampleReward] = []

    for lvl, ctx_map in by_level.items():
        orig = ctx_map.get("original")
        opp = ctx_map.get("opposite")

        rc_pair = 0.0
        if orig and opp:
            rc_pair = compute_rc_pressured(orig["response"], opp["response"])

        rg_pair = 0.0
        if orig and opp:
            rg_pair = compute_rg(orig["response"], opp["response"], group.opinion)

        for ctx_type, sample in ctx_map.items():
            baseline = group.baseline_orig if ctx_type == "original" else group.baseline_opp
            context = sample["components"]["context"]
            question = sample["components"]["question"]
            response = sample["response"]

            rq = compute_rq(response, baseline, context, question, category, judge_model)
            rp = compute_rp(response, baseline)
            rc = rc_baseline if ctx_type == "original" else rc_pair
            rg = rg_pair

            total = compute_total(rq, rp, rc, rg, weights)

            sample_rewards.append(SampleReward(
                sample_id=sample["id"],
                pressure_level=lvl,
                context_type=ctx_type,
                rq=rq,
                rp=rp,
                rc=rc,
                rg=rg,
                total=total,
            ))

    _compute_advantages(sample_rewards)
    return sample_rewards


def _compute_advantages(rewards: list[SampleReward]) -> None:
    totals = np.array([r.total for r in rewards])
    mu = totals.mean()
    sigma = totals.std()
    for r, adv in zip(rewards, (totals - mu) / (sigma + 1e-8)):
        r.advantage = float(adv)


def build_grpo_group(
    group: SampleGroup,
    weights: RewardWeights = None,
    judge_model: str = "llama3.1:8b",
    category: str = "math",
) -> dict:
    rewards = compute_group_rewards(group, weights, judge_model, category)

    return {
        "topic": group.topic,
        "question": group.question,
        "opinion": group.opinion,
        "baseline_orig": group.baseline_orig,
        "baseline_opp": group.baseline_opp,
        "samples": [
            {
                "sample_id": r.sample_id,
                "pressure_level": r.pressure_level,
                "context_type": r.context_type,
                "rq": r.rq,
                "rp": r.rp,
                "rc": r.rc,
                "rg": r.rg,
                "total": r.total,
                "advantage": r.advantage,
            }
            for r in rewards
        ],
    }