from module2.source_validator import SourceValidator
from module2.config import TIME_SENSITIVE_MAX_CONFIDENCE


class TimeSensitiveVerifier:

    def __init__(self, llm_pipeline=None):
        self.llm_pipeline = llm_pipeline
        self.validator = SourceValidator(llm_pipeline)

    def verify(self, claim_a: str, claim_b: str, question: str, bias_info: dict) -> dict:
        sources_a = self.validator.validate_claim(claim_a, question)
        sources_b = self.validator.validate_claim(claim_b, question)

        agg_a = self.validator.aggregate_sources(sources_a)
        agg_b = self.validator.aggregate_sources(sources_b)

        a_support = agg_a["support_count"]
        b_support = agg_b["support_count"]
        a_refute = agg_a["refute_count"]
        b_refute = agg_b["refute_count"]

        if a_support > b_support and a_refute <= b_refute:
            selected_source = "claim_A"
            selected_claim = claim_a
            raw_confidence = agg_a["agreement_ratio"]
        elif b_support > a_support and b_refute <= a_refute:
            selected_source = "claim_B"
            selected_claim = claim_b
            raw_confidence = agg_b["agreement_ratio"]
        elif a_support == b_support:
            if a_refute < b_refute:
                selected_source = "claim_A"
                selected_claim = claim_a
                raw_confidence = 0.5
            elif b_refute < a_refute:
                selected_source = "claim_B"
                selected_claim = claim_b
                raw_confidence = 0.5
            else:
                selected_source = "neither"
                selected_claim = claim_a
                raw_confidence = 0.0
        else:
            selected_source = "neither"
            selected_claim = claim_a
            raw_confidence = 0.0

        confidence = min(raw_confidence, TIME_SENSITIVE_MAX_CONFIDENCE)

        if selected_source == "claim_A":
            recommendation = "maintain_original"
        elif selected_source == "claim_B":
            recommendation = "accept_correction"
        else:
            recommendation = "flag_uncertain"

        return {
            "verification_type": "time_sensitive",
            "selected_claim": selected_claim,
            "selected_source": selected_source,
            "confidence": confidence,
            "reasoning_trace": {
                "source_chains_a": sources_a,
                "source_chains_b": sources_b,
                "aggregation_a": agg_a,
                "aggregation_b": agg_b,
                "agreement_count": max(a_support, b_support),
                "conflict_count": agg_a["refute_count"] + agg_b["refute_count"],
                "recency_warning": True,
                "last_known_date": "training cutoff",
            },
            "sycophancy_detected": False,
            "recommendation": recommendation,
            "caveat": "Time-sensitive claim — recommend verification with current sources.",
        }