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
        question: str = "",
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
            question=question,
            use_llm=use_llm and self.llm_pipeline is not None,
        )

        shift_judgment = shift_result["shift_judgment"]
        pressure_analysis = shift_result["pressure_analysis"]
        dual_eval = shift_result["dual_evaluation"]
        unbiased_output = shift_result["unbiased_output"]

        sycophancy_detected = shift_judgment["sycophancy_likely"]
        recommendation = shift_judgment["recommendation"]

        if conv_summary.get("position_shifted", False) and pressure_analysis["is_high_pressure"]:
            sycophancy_detected = True
            if recommendation not in ("generate_unbiased",):
                recommendation = "generate_unbiased"

        if sycophancy_detected and unbiased_output is not None:
            unbiased_rec = unbiased_output.get("recommendation", "BALANCED")
            if unbiased_rec == "FAVOR_A":
                selected_source = "claim_A"
                selected_claim = claim_a
            elif unbiased_rec == "FAVOR_B":
                selected_source = "claim_B"
                selected_claim = claim_b
            else:
                selected_source = "both_valid"
                selected_claim = unbiased_output.get("unbiased_position", claim_a)
        elif sycophancy_detected and unbiased_output is None:
            selected_source = "both_valid"
            selected_claim = f"Both perspectives have merit. Position A: {claim_a[:80]}. Position B: {claim_b[:80]}."
        elif recommendation == "accept_correction":
            selected_source = "claim_B"
            selected_claim = claim_b
        elif recommendation == "acknowledge_both":
            selected_source = "both_valid"
            selected_claim = claim_a
        else:
            selected_source = "both_valid"
            selected_claim = claim_a

        if sycophancy_detected:
            raw_confidence = 0.7
            if pressure_analysis["is_high_pressure"] and dual_eval.get("sycophancy_signal"):
                raw_confidence = 0.8
        else:
            raw_confidence = 0.5
        confidence = min(raw_confidence, SUBJECTIVE_MAX_CONFIDENCE)

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
                "dual_evaluation": {
                    "clean_assessment": dual_eval.get("clean_evaluation", {}).get("assessment", ""),
                    "pressured_assessment": dual_eval.get("pressured_evaluation", {}).get("assessment", ""),
                    "outputs_differ": dual_eval.get("outputs_differ", False),
                    "sycophancy_signal": dual_eval.get("sycophancy_signal", False),
                },
                "shift_analysis": {
                    "sycophancy_detected": sycophancy_detected,
                    "shift_reasoning": shift_judgment["reasoning"],
                },
                "unbiased_output": unbiased_output,
                "conversation_summary": conv_summary,
            },
            "sycophancy_detected": sycophancy_detected,
            "recommendation": recommendation,
            "unbiased_response": unbiased_output,
            "suggested_response_strategy": (
                "Sycophancy detected. Generated unbiased opinion considering both perspectives."
                if sycophancy_detected
                else "No sycophancy detected. Consider both perspectives and provide a balanced response."
            ),
        }