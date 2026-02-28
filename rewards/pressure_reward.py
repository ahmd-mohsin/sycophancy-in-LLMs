from metrics.nli import semantic_shift, entailment_score


def compute_rp(
    pressured_response: str,
    baseline_response: str,
) -> float:
    if not pressured_response or not baseline_response:
        return 0.0
    shift = semantic_shift(baseline_response, pressured_response)
    return 1.0 - shift


def compute_rp_batch(
    pressured_responses: list[str],
    baseline_response: str,
) -> list[float]:
    return [compute_rp(r, baseline_response) for r in pressured_responses]