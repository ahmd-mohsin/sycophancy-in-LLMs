"""
Position Consistency Reward (Rpos)  —  v3

v1/v2 PROBLEMS:
1. Rpos only fired for opposite-context samples (is_opposite=True in grpo_reward_fn).
   For original-context samples it was hardcoded to 0.0.
   This means for half the training data, PACF received zero gradient signal.

2. PACF metric measures Pearson correlation between pressure level and CFS score
   across levels. To push PACF positive, the model needs to maintain high
   position consistency ACROSS all pressure levels, not just opposite context.

3. The entailment - contradiction formula is correct but the reference was
   sometimes wrong (see grpo_reward_fn bug history). v3 ensures the reference
   is always context-matched.

v3 FIX:
1. Rpos fires for BOTH context types:
   - Original context: reward for entailing baseline_orig (maintain correct position)
   - Opposite context: reward for entailing baseline_opp (adopt correct opposite position)
   Both push PACF in the same direction.

2. For original context, Rpos acts as a pressure-level-aware bonus:
   the gradient tells the model "you should sound like your baseline even under pressure."
   This directly reduces PSS as well as helping PACF.

3. Rpos range extended: original context uses [0,1] entailment only (no contradiction
   penalty — we don't want to penalize the model for acknowledging the other side).
   Opposite context keeps the full entailment - contradiction formula since there
   the model SHOULD be contradicting the original opinion.
"""
from metrics.nli import nli_score_batch


def compute_rpos(response: str, correct_baseline: str,
                 is_opposite: bool = False) -> float:
    """
    Position consistency reward.

    Args:
        response:         The model's completion under pressure.
        correct_baseline: Context-matched baseline:
                          baseline_orig for original context,
                          baseline_opp  for opposite context.
                          Caller (grpo_reward_fn) resolves this.
        is_opposite:      Whether this is an opposite-context sample.
                          Controls whether contradiction is penalized.

    Returns:
        Float reward:
          Original context: entailment(correct_baseline, response) in [0, 1]
          Opposite context: entailment - contradiction in [-1, 1]

    Note: the old signature was compute_rpos(response, baseline_orig) with
    no is_opposite flag. The new signature is backward-compatible since
    is_opposite defaults to False.
    """
    if not response or not correct_baseline:
        return 0.0

    results = nli_score_batch([(correct_baseline, response)])
    scores  = results[0]
    entailment    = float(scores.get("entailment",    0.0))
    contradiction = float(scores.get("contradiction", 0.0))

    if is_opposite:
        # Opposite context: model should fully adopt the opposite position.
        # Reward positive alignment, penalize contradiction.
        return entailment - contradiction
    else:
        # Original context: model should maintain its original position.
        # Only reward entailment; do not penalize contradiction since the
        # model may legitimately acknowledge the pressure argument.
        return entailment


def compute_rpos_batch(
    responses: list[str],
    correct_baseline: str,
    is_opposite: bool = False,
) -> list[float]:
    """
    Batch version for efficiency.
    """
    if not responses or not correct_baseline:
        return [0.0] * len(responses)

    pairs   = [(correct_baseline, r) if r else ("", "") for r in responses]
    results = nli_score_batch(pairs)

    output = []
    for r, scores in zip(responses, results):
        if not r:
            output.append(0.0)
            continue
        e = float(scores.get("entailment",    0.0))
        c = float(scores.get("contradiction", 0.0))
        if is_opposite:
            output.append(e - c)
        else:
            output.append(e)

    return output