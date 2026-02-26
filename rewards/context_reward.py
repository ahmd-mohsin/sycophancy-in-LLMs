from metrics.nli import contradiction_score, entailment_score


def compute_rc(
    baseline_orig: str,
    baseline_opp: str,
) -> float:
    return float(contradiction_score(baseline_orig, baseline_opp))


def compute_rc_pressured(
    response_orig: str,
    response_opp: str,
) -> float:
    return float(contradiction_score(response_orig, response_opp))