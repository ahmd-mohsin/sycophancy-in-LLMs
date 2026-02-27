from module2.config import (
    FACTUAL_HIGH_CONFIDENCE,
    FACTUAL_MEDIUM_CONFIDENCE,
    FACTUAL_LOW_CONFIDENCE,
    MINIMUM_AGREEMENT_RATIO,
)


class VoteAggregator:

    @staticmethod
    def count_verdicts(chain_results: list) -> dict:
        counts = {"CORRECT": 0, "INCORRECT": 0, "UNCERTAIN": 0}
        for result in chain_results:
            verdict = result.get("verdict", "UNCERTAIN").upper()
            if verdict in counts:
                counts[verdict] += 1
            else:
                counts["UNCERTAIN"] += 1
        return counts

    @staticmethod
    def aggregate_votes(chain_results_a: list, chain_results_b: list) -> dict:
        vote_a = VoteAggregator.count_verdicts(chain_results_a)
        vote_b = VoteAggregator.count_verdicts(chain_results_b)

        total_chains = len(chain_results_a)

        a_correct = vote_a["CORRECT"]
        a_incorrect = vote_a["INCORRECT"]
        b_correct = vote_b["CORRECT"]
        b_incorrect = vote_b["INCORRECT"]

        if a_correct == total_chains and b_incorrect >= total_chains - 1:
            selected = "claim_A"
            confidence = FACTUAL_HIGH_CONFIDENCE
        elif b_correct == total_chains and a_incorrect >= total_chains - 1:
            selected = "claim_B"
            confidence = FACTUAL_HIGH_CONFIDENCE

        elif a_correct >= 2 and b_incorrect >= 2:
            selected = "claim_A"
            confidence = FACTUAL_MEDIUM_CONFIDENCE
        elif b_correct >= 2 and a_incorrect >= 2:
            selected = "claim_B"
            confidence = FACTUAL_MEDIUM_CONFIDENCE

        elif a_correct >= 2 and b_correct >= 2:
            selected = "claim_A" if a_correct > b_correct else "claim_B" if b_correct > a_correct else "neither"
            confidence = FACTUAL_LOW_CONFIDENCE

        elif a_correct > b_correct:
            selected = "claim_A"
            confidence = FACTUAL_MEDIUM_CONFIDENCE if a_correct >= 2 else FACTUAL_LOW_CONFIDENCE
        elif b_correct > a_correct:
            selected = "claim_B"
            confidence = FACTUAL_MEDIUM_CONFIDENCE if b_correct >= 2 else FACTUAL_LOW_CONFIDENCE

        else:
            selected = "neither"
            confidence = 0.0

        vote_margin = abs(a_correct - b_correct)

        return {
            "selected_source": selected,
            "confidence": confidence,
            "vote_a": vote_a,
            "vote_b": vote_b,
            "vote_margin": vote_margin,
            "a_agreement_ratio": a_correct / max(total_chains, 1),
            "b_agreement_ratio": b_correct / max(total_chains, 1),
        }