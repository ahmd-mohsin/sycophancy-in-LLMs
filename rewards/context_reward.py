"""
Context Fidelity Reward (Rc)  —  v3

v1/v2 BUG: compute_rc_pressured() measured contradiction_score(completion, baseline).
This rewarded the completion for DIVERGING from the baseline, which is the
opposite of what CFS measures.

CFS metric (sycophancy_metrics.py compute_cfs) measures:
    entailment_score(orig_response, opp_response)
i.e. how SIMILAR the two responses are at the same pressure level.
A high CFS means the model gives consistent, evidence-faithful responses
regardless of context — its original-context answer entails its opposite-context answer.

v3 FIX: Rc now measures entailment between the completion and the
context-matched baseline, NOT contradiction. This directly optimises what
CFS measures: completions that entail the correct baseline will be consistent
across context types, driving CFS up.

Additionally we add a cross-context consistency bonus: for original-context
completions we reward entailment with baseline_orig; for opposite-context
completions we reward entailment with baseline_opp. Both push toward the
same goal — the model producing responses that are grounded in the provided
evidence rather than in the pressure direction.
"""
from metrics.nli import entailment_score, nli_score_batch


def compute_rc_single(
    completion: str,
    correct_baseline: str,
) -> float:
    """
    Single-completion Rc: entailment of completion with context-matched baseline.

    This is the reward that gets called in grpo_reward_fn for each completion.
    correct_baseline is baseline_orig for original context, baseline_opp for
    opposite context — the caller resolves this.

    Returns float in [0, 1]. Higher = more faithful to evidence context.
    """
    if not completion or not correct_baseline:
        return 0.0
    return float(entailment_score(correct_baseline, completion))


def compute_rc_pressured(
    completion: str,
    rc_reference: str,
) -> float:
    """
    Drop-in replacement for the old compute_rc_pressured.
    rc_reference should already be the context-matched baseline
    (caller in grpo_reward_fn resolves original vs opposite).

    Old signature was (response_orig, response_opp) — contradicted the metric.
    New signature is (completion, correct_baseline) — directly optimises CFS.
    """
    return compute_rc_single(completion, rc_reference)


def compute_rc(
    baseline_orig: str,
    baseline_opp: str,
) -> float:
    """
    Legacy function kept for backward compatibility with reward_dataset.py.
    Measures contradiction between the two baselines — used to validate
    dataset quality, not as a training reward.
    """
    from metrics.nli import contradiction_score
    if not baseline_orig or not baseline_opp:
        return 0.0
    return float(contradiction_score(baseline_orig, baseline_opp))


def compute_rc_batch(
    completions: list[str],
    correct_baseline: str,
) -> list[float]:
    """
    Batch version for efficiency when scoring many completions at once.
    """
    if not completions or not correct_baseline:
        return [0.0] * len(completions)
    pairs = [(correct_baseline, c) if c else ("", "") for c in completions]
    results = nli_score_batch(pairs)
    out = []
    for c, scores in zip(completions, results):
        if not c:
            out.append(0.0)
        else:
            out.append(float(scores.get("entailment", 0.0)))
    return out