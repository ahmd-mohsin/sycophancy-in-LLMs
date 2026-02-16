from module2.chain_of_verification import ChainOfVerification
from module2.vote_aggregator import VoteAggregator


class FactualVerifier:

    def __init__(self, llm_pipeline=None):
        self.llm_pipeline = llm_pipeline
        self.cov = ChainOfVerification(llm_pipeline)
        self.aggregator = VoteAggregator()

    def verify(self, claim_a: str, claim_b: str, question: str, bias_info: dict) -> dict:
        chains_a = self.cov.run_all_chains(claim_a, question)
        chains_b = self.cov.run_all_chains(claim_b, question)

        vote_result = self.aggregator.aggregate_votes(chains_a, chains_b)

        selected_source = vote_result["selected_source"]

        if selected_source == "claim_A":
            selected_claim = claim_a
            recommendation = "maintain_original"
        elif selected_source == "claim_B":
            selected_claim = claim_b
            recommendation = "accept_correction"
        else:
            selected_claim = claim_a
            recommendation = "flag_uncertain"

        sycophancy_detected = (
            selected_source == "claim_A"
            and bias_info.get("has_bias", False)
            and bias_info.get("sycophancy_risk", "none") in ("high", "medium")
        )

        return {
            "verification_type": "factual",
            "selected_claim": selected_claim,
            "selected_source": selected_source,
            "confidence": vote_result["confidence"],
            "reasoning_trace": {
                "claim_a_chains": chains_a,
                "claim_b_chains": chains_b,
                "vote_a": vote_result["vote_a"],
                "vote_b": vote_result["vote_b"],
                "vote_margin": vote_result["vote_margin"],
            },
            "sycophancy_detected": sycophancy_detected,
            "recommendation": recommendation,
        }