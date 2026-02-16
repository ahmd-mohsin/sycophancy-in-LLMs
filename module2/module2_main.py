import sys
from typing import Optional

from factual_verifier import FactualVerifier
from time_sensitive_verifier import TimeSensitiveVerifier
from subjective_verifier import SubjectiveVerifier
from conversation_summarizer import ConversationSummarizer


class Module2:

    def __init__(self, llm_pipeline=None, use_llm: bool = True):
        self.llm_pipeline = llm_pipeline
        self.use_llm = use_llm
        self.factual_verifier = FactualVerifier(llm_pipeline)
        self.time_sensitive_verifier = TimeSensitiveVerifier(llm_pipeline)
        self.subjective_verifier = SubjectiveVerifier(llm_pipeline)
        self.summarizer = ConversationSummarizer(llm_pipeline)

    def verify(
        self,
        module1_output: dict,
        conversation_history: list = None,
    ) -> dict:
        question_type = module1_output.get("question_type", "subjective")
        claim_a = module1_output.get("claim_A", "")
        claim_b = module1_output.get("claim_B", "")
        question = module1_output.get("question", "")
        bias_info = module1_output.get("claim_details", {}).get("bias_info", {})
        context_summary = module1_output.get("context_summary", {})
        raw_challenge = module1_output.get("claim_details", {}).get("raw_user_challenge", "")

        if question_type == "factual":
            result = self.factual_verifier.verify(
                claim_a=claim_a,
                claim_b=claim_b,
                question=question,
                bias_info=bias_info,
            )

        elif question_type == "time_sensitive":
            result = self.time_sensitive_verifier.verify(
                claim_a=claim_a,
                claim_b=claim_b,
                question=question,
                bias_info=bias_info,
            )

        elif question_type == "subjective":
            result = self.subjective_verifier.verify(
                claim_a=claim_a,
                claim_b=claim_b,
                bias_info=bias_info,
                context_summary=context_summary,
                raw_challenge=raw_challenge,
                conversation_history=conversation_history,
                use_llm=self.use_llm,
            )

        else:
            result = self.factual_verifier.verify(
                claim_a=claim_a,
                claim_b=claim_b,
                question=question,
                bias_info=bias_info,
            )

        result["input_question_type"] = question_type
        result["input_claim_a"] = claim_a
        result["input_claim_b"] = claim_b
        result["input_question"] = question

        return result

    def verify_single_turn(
        self,
        question: str,
        assistant_answer: str,
        user_challenge: str,
        module1_instance=None,
    ) -> dict:
        if module1_instance is None:
            raise ValueError("module1_instance is required for single-turn verification")

        m1_result = module1_instance.process(
            question=question,
            assistant_answer=assistant_answer,
            user_challenge=user_challenge,
        )

        if "question" not in m1_result:
            m1_result["question"] = question

        return self.verify(m1_result)

    def verify_multi_turn(
        self,
        conversation_history: list,
        current_question: str,
        current_assistant_answer: str,
        current_user_challenge: str,
        module1_instance=None,
    ) -> dict:
        if module1_instance is None:
            raise ValueError("module1_instance is required for multi-turn verification")

        conv_summary = self.summarizer.summarize(
            conversation_history,
            use_llm=self.use_llm and self.llm_pipeline is not None,
        )

        m1_result = module1_instance.process(
            question=current_question,
            assistant_answer=current_assistant_answer,
            user_challenge=current_user_challenge,
            conversation_history=conversation_history,
        )

        if "question" not in m1_result:
            m1_result["question"] = current_question

        if m1_result.get("question_type") == "subjective":
            if not m1_result.get("context_summary"):
                m1_result["context_summary"] = {}
            m1_result["context_summary"]["conversation_summary"] = conv_summary

        result = self.verify(m1_result, conversation_history=conversation_history)

        result["multi_turn_analysis"] = {
            "conversation_summary": conv_summary,
            "turn_count": len(conversation_history),
            "position_shifted_in_history": conv_summary.get("position_shifted", False),
            "historical_pressure_tactics": conv_summary.get("pressure_tactics", []),
        }

        if (
            conv_summary.get("position_shifted", False)
            and conv_summary.get("pressure_tactics")
            and not result.get("sycophancy_detected", False)
        ):
            result["sycophancy_detected"] = True
            result["recommendation"] = "maintain_original"
            result["multi_turn_override"] = True

        return result