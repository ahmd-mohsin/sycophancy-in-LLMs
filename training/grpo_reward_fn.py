"""
GRPO Live Reward Function  —  v6

Root cause analysis of PSS regression (v2→v4/v5):
─────────────────────────────────────────────────────────────────────────────
v2 had PSS=0.0128, GAS=0.0000.
v4/v5 have PSS≈0.11, GAS≈0.05.

The v4 RC fix (contradiction→entailment) is theoretically correct but
creates a gradient conflict on original-context samples:

  baseline_orig already entails opinion o (it is the committed pro-o response).
  RC_orig = entailment(baseline_orig, completion) rewards completions that
  agree with o. But Rg penalises agreement with o. On original-context samples
  RC and Rg are in direct opposition, producing noisy gradients and driving
  GAS up.

  On opposite-context samples this conflict does not exist: baseline_opp
  entails ¬o, so entailment(baseline_opp, completion) rewards disagreeing
  with o, which is consistent with Rg.

FIX (v6):
  RC is now context-aware:
    - Original context: RC = entailment(baseline_orig, completion)
      BUT capped: only applied when the completion does NOT strongly entail o.
      Concretely: RC_orig = entailment(baseline_orig, c) * (1 - entailment(o, c))
      This rewards following the baseline style/reasoning without rewarding
      sycophantic agreement with the pushed opinion.

    - Opposite context: RC = entailment(baseline_opp, completion) [unchanged from v4]
      No conflict here — following baseline_opp means disagreeing with o.

  RPOS: reverted to opposite-context only.
    Firing Rpos on original-context samples (v4 change) was redundant with RC
    on original context and added noise. Opposite-context Rpos is kept — it
    provides the clearest gradient signal for PACF.

  Weights: α=1.0, β=0.8, γ=0.5, ε=0.6, δ=0.6 (same as v5).
  Epochs: 1.
  Pressure multiplier: removed (confirmed harmful in v4 diagnosis).
─────────────────────────────────────────────────────────────────────────────
"""
import re
from dataclasses import dataclass, field
from collections import defaultdict

from rewards.pressure_reward import compute_rp
from rewards.agreement_penalty import compute_rg_single, compute_rg_opposite_context
from rewards.position_reward import compute_rpos
from metrics.nli import entailment_score, nli_score_batch

_IDX_RE = re.compile(r"__GRPO_IDX_(\d+)__")

FACTUAL_CATEGORIES = {"math", "physics"}

MIN_WORDS       = 60
IDEAL_MIN_WORDS = 120
IDEAL_MAX_WORDS = 200


@dataclass
class RewardWeights:
    alpha:   float = 1.0   # Rq: factual correctness
    beta:    float = 0.8   # Rc: context fidelity
    gamma:   float = 0.5   # Rp: pressure resistance
    epsilon: float = 0.6   # Rpos: position consistency (opposite context only)
    delta:   float = 0.6   # Rg: agreement penalty


@dataclass
class _RewardLog:
    counts:  dict = field(default_factory=lambda: defaultdict(int))
    sums:    dict = field(default_factory=lambda: defaultdict(float))
    sq_sums: dict = field(default_factory=lambda: defaultdict(float))

    def record(self, components: dict):
        for k, v in components.items():
            if v is not None:
                self.counts[k]  += 1
                self.sums[k]    += v
                self.sq_sums[k] += v * v

    def stats(self) -> dict:
        out = {}
        for k in self.counts:
            n    = self.counts[k]
            mean = self.sums[k] / n if n else 0.0
            var  = (self.sq_sums[k] / n - mean ** 2) if n > 1 else 0.0
            out[k] = {"mean": round(mean, 4), "std": round(var ** 0.5, 4), "n": n}
        return out


def _length_multiplier(text: str) -> float:
    n = len(text.split())
    if n < MIN_WORDS:
        return 0.0
    if n < IDEAL_MIN_WORDS:
        return 0.5 + 0.5 * (n - MIN_WORDS) / (IDEAL_MIN_WORDS - MIN_WORDS)
    bonus = 0.1 * min((n - IDEAL_MIN_WORDS) / (IDEAL_MAX_WORDS - IDEAL_MIN_WORDS), 1.0)
    return 1.0 + bonus


def _compute_rc_v6(completion: str, correct_baseline: str,
                   opinion: str, is_opposite: bool) -> float:
    """
    Context-aware RC.

    Opposite context: entailment(baseline_opp, completion).
        Unambiguous — following baseline_opp means disagreeing with o.

    Original context: entailment(baseline_orig, completion) dampened by
        how much the completion agrees with the pushed opinion o.
        RC_orig = entail(baseline_orig, c) * (1 - entail(o, c))
        This rewards baseline-faithful responses that are NOT sycophantic.
        A completion that perfectly follows baseline_orig while also agreeing
        with o (sycophantic shortcut) gets dampened toward zero.
    """
    if not completion or not correct_baseline:
        return 0.0

    if is_opposite:
        return float(entailment_score(correct_baseline, completion))

    # Original context: compute both scores in one batch call
    pairs = [(correct_baseline, completion), (opinion, completion)]
    results = nli_score_batch(pairs)
    e_baseline = float(results[0].get("entailment", 0.0))
    e_opinion  = float(results[1].get("entailment", 0.0))
    return e_baseline * (1.0 - e_opinion)


def _compute_total(rq, rp, rc, rpos, rg, weights: RewardWeights,
                   category: str) -> float:
    if category in FACTUAL_CATEGORIES:
        rq_term = weights.alpha * rq if rq is not None else 0.0
        rp_term = weights.gamma * rp
    else:
        rq_term = 0.0
        rp_term = (weights.alpha + weights.gamma) * rp

    return (
        rq_term
        + weights.beta    * rc
        + rp_term
        + weights.epsilon * rpos
        + weights.delta   * rg   # rg is negative
    )


def make_reward_fn(
    grpo_groups: list[dict],
    weights: RewardWeights = None,
    category: str = "opinion",
    judge_model: str = None,
):
    if weights is None:
        weights = RewardWeights()

    _log = _RewardLog()

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
            is_opposite    = (context_type == "opposite")

            correct_baseline = baseline_opp if is_opposite else baseline_orig

            # ── RQ: factual correctness ───────────────────────────────────────
            if cat in FACTUAL_CATEGORIES:
                rq = float(entailment_score(correct_baseline, completion))
            else:
                rq = None

            # ── RP: pressure resistance ───────────────────────────────────────
            rp = compute_rp(completion, correct_baseline)

            # ── RC: context fidelity (v6 context-aware) ───────────────────────
            # Original: entail(baseline_orig, c) * (1 - entail(o, c))
            #   Rewards baseline fidelity but dampens sycophantic shortcuts.
            # Opposite: entail(baseline_opp, c) [unchanged from v4]
            rc = _compute_rc_v6(completion, correct_baseline, opinion, is_opposite)

            # ── RPOS: position consistency (opposite context only) ─────────────
            # Reverted to opposite-only (v4's both-context firing was redundant
            # with RC on original context and added noise to PSS gradient).
            if is_opposite:
                rpos = compute_rpos(completion, correct_baseline, is_opposite=True)
            else:
                rpos = 0.0

            # ── RG: generic agreement penalty ─────────────────────────────────
            if is_opposite:
                rg = compute_rg_opposite_context(completion, opinion)
            else:
                rg = compute_rg_single(completion, opinion)

            # ── Aggregate ─────────────────────────────────────────────────────
            total = _compute_total(rq, rp, rc, rpos, rg, weights, cat)
            total = total * _length_multiplier(completion)

            _log.record({
                "rq":    rq,
                "rp":    rp,
                "rc":    rc,
                "rpos":  rpos if is_opposite else None,
                "rg":    rg,
                "total": total,
            })

            rewards.append(float(total))

        return rewards

    reward_fn.get_stats = _log.stats
    return reward_fn