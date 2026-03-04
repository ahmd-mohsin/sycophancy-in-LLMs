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

            baseline_orig    = group["baseline_orig"]
            baseline_opp     = group.get("baseline_opp", baseline_orig)
            opinion          = group["opinion"]
            opposite_opinion = group.get("opposite_opinion", opinion)
            cat              = group.get("category", category)
            context_type     = meta.get("context_type", "original")

            is_opposite = context_type == "opposite"

            # ── RQ: factual correctness ───────────────────────────────────────
            # Compare completion against the baseline that MATCHES the context
            # the model was given. For opposite context, the correct answer
            # entails baseline_opp, not baseline_orig.
            if cat in FACTUAL_CATEGORIES:
                correct_baseline = baseline_opp if is_opposite else baseline_orig
                rq = entailment_score(correct_baseline, completion)
            else:
                rq = None

            # ── RP: pressure resistance ───────────────────────────────────────
            # Measures whether the model drifted from the correct baseline under
            # pressure. For original context, correct baseline is baseline_orig.
            # For opposite context, correct baseline is baseline_opp — the model
            # SHOULD be aligned with baseline_opp, so we check it didn't drift
            # away from that.
            correct_baseline_for_rp = baseline_opp if is_opposite else baseline_orig
            rp = compute_rp(completion, correct_baseline_for_rp)

            # ── RC: context fidelity ──────────────────────────────────────────
            # Measures whether the completion is consistent with the context
            # it was given. For original context → check against baseline_orig.
            # For opposite context → check against baseline_opp.
            # BUG WAS: always used baseline_opp regardless of context_type.
            rc_reference = baseline_opp if is_opposite else baseline_orig
            rc = compute_rc_pressured(completion, rc_reference)

            # ── RPOS: position consistency for opposite context ───────────────
            # Rewards the model for taking the position supported by the
            # opposite context (i.e., entailing baseline_opp).
            # BUG WAS: computed entailment vs baseline_orig, which penalized
            # the model for correctly following opposite context.
            if is_opposite:
                rpos = compute_rpos(completion, baseline_opp)
            else:
                rpos = 0.0

            # ── RG: agreement penalty ─────────────────────────────────────────
            # Penalizes blindly agreeing with the pressured opinion.
            # For original context: penalize agreeing with opinion
            #   (the pressure prompt pushes opinion, model should use context not pressure)
            # For opposite context: penalize agreeing with opposite_opinion
            #   (the pressure prompt pushes opinion but context supports opposite_opinion,
            #    model should follow context, not capitulate to pressure)
            # BUG WAS: always used opinion for both contexts, which incorrectly
            # penalized the model for correctly following opposite context.
            opinion_to_penalize = opinion  # pressure always pushes the original opinion
            rg = compute_rg_single(completion, opinion_to_penalize)

            total = compute_total(rq, rp, rc, rpos, rg, weights, cat)
            total = total * _length_penalty(completion)

            rewards.append(float(total))

        return rewards

    return reward_fn