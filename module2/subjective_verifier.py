from module2.opinion_shift_detector import OpinionShiftDetector
from module2.conversation_summarizer import ConversationSummarizer
from module2.config import SUBJECTIVE_MAX_CONFIDENCE


class SubjectiveVerifier:

    def __init__(self, llm_pipeline=None):
        self.llm_pipeline = llm_pipeline
        self.shift_detector = OpinionShiftDetector(llm_pipeline)
        self.summarizer = ConversationSummarizer(llm_pipeline)

    def verify(
        self,
        claim_a: str,
        claim_b: str,
        bias_info: dict,
        context_summary: dict = None,
        raw_challenge: str = "",
        conversation_history: list = None,
        use_llm: bool = True,
    ) -> dict:
        conv_summary = {}
        if conversation_history and len(conversation_history) > 0:
            conv_summary = self.summarizer.summarize(conversation_history, use_llm=use_llm)
        elif context_summary:
            conv_summary = context_summary

        shift_result = self.shift_detector.detect_shift(
            claim_a=claim_a,
            claim_b=claim_b,
            bias_info=bias_info,
            raw_challenge=raw_challenge,
            use_llm=use_llm and self.llm_pipeline is not None,
        )

        shift_judgment = shift_result["shift_judgment"]
        pressure_analysis = shift_result["pressure_analysis"]

        sycophancy_likely = shift_judgment["sycophancy_likely"]
        recommendation = shift_judgment["recommendation"]

        if conv_summary.get("position_shifted", False) and pressure_analysis["is_high_pressure"]:
            sycophancy_likely = True
            if recommendation != "maintain_original":
                recommendation = "maintain_original"

        if sycophancy_likely:
            selected_source = "claim_A"
            selected_claim = claim_a
        elif recommendation == "accept_correction":
            selected_source = "claim_B"
            selected_claim = claim_b
        elif recommendation == "acknowledge_both":
            selected_source = "both_valid"
            selected_claim = claim_a
        else:
            selected_source = "claim_A"
            selected_claim = claim_a

        raw_confidence = 0.7 if sycophancy_likely else 0.5
        if pressure_analysis["is_high_pressure"] and pressure_analysis["is_pressure_only"]:
            raw_confidence = 0.8
        confidence = min(raw_confidence, SUBJECTIVE_MAX_CONFIDENCE)

        acknowledge_alternative = (
            not sycophancy_likely
            or recommendation == "acknowledge_both"
            or shift_result["substance_analysis"].get("has_substance", False)
        )

        return {
            "verification_type": "subjective",
            "selected_claim": selected_claim,
            "selected_source": selected_source,
            "confidence": confidence,
            "reasoning_trace": {
                "pressure_analysis": {
                    "sycophancy_risk": pressure_analysis["sycophancy_risk"],
                    "pressure_types": pressure_analysis["pressure_types"],
                    "substantive_new_info": shift_result["substance_analysis"].get("has_substance", False),
                },
                "shift_analysis": {
                    "original_position_clear": True,
                    "would_shift_be_sycophantic": sycophancy_likely,
                    "shift_reasoning": shift_judgment["reasoning"],
                },
                "llm_judge_verdict": {
                    "sycophancy_likely": sycophancy_likely,
                    "reasoning": shift_judgment["reasoning"],
                },
                "conversation_summary": conv_summary,
            },
            "sycophancy_detected": sycophancy_likely,
            "recommendation": recommendation,
            "acknowledge_alternative": acknowledge_alternative,
            "suggested_response_strategy": (
                "Maintain position while acknowledging the user's perspective as valid."
                if sycophancy_likely
                else "Consider both perspectives and provide a balanced response."
            ),
        }