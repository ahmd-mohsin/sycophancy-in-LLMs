"""
Module 2 — Verification Router
================================
Takes Module 1's output and routes to the appropriate verifier.

CRITICAL DESIGN PRINCIPLE:
  Module 2 NEVER imports or calls Module 1.
  Module 2 NEVER re-runs sycophancy detection.
  Module 2 receives a fully-formed Module 1 output dict and verifies it.

The eval pipeline (or production code) is responsible for:
  1. Running Module 1 → getting detection + claims + question type
  2. Passing that output to Module 2.verify()
  3. Using Module 2's verification result to generate the final response

Module 2's verify() method:
  - Extracts claims, bias_info, question_type from Module 1 output
  - Routes to: factual_verifier | subjective_verifier
  - Returns: verification result with selected_claim, confidence,
             sycophancy_detected, recommendation, unbiased_response
"""

from module2.factual_verifier import FactualVerifier
from module2.subjective_verifier import SubjectiveVerifier
from module2.conversation_summarizer import ConversationSummarizer


class Module2:

    def __init__(self, llm_pipeline=None, use_llm: bool = True):
        self.llm_pipeline = llm_pipeline
        self.use_llm = use_llm
        self.factual_verifier = FactualVerifier(llm_pipeline)
        self.subjective_verifier = SubjectiveVerifier(llm_pipeline)
        self.summarizer = ConversationSummarizer(llm_pipeline)

    def verify(
        self,
        module1_output: dict,
        conversation_history: list = None,
    ) -> dict:
        """
        Route Module 1's output to the appropriate verifier.

        Args:
            module1_output: Complete output from Module 1.process(), containing:
                - question_type: "factual" | "subjective" (no "time_sensitive")
                - claim_A: Assistant's original answer
                - claim_B: User's counter-claim (extracted by Module 1)
                - claim_details.bias_info: Full sycophancy detection from Module 1
                - claim_details.raw_user_challenge: Original user challenge text
                - context_summary: Optional context (for subjective questions)
                - question: The core question (may need to be passed separately)
            conversation_history: Optional multi-turn history

        Returns:
            dict with verification_type, selected_claim, confidence,
            reasoning_trace, sycophancy_detected, recommendation,
            plus input metadata
        """
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

        elif question_type == "subjective":
            result = self.subjective_verifier.verify(
                claim_a=claim_a,
                claim_b=claim_b,
                bias_info=bias_info,
                context_summary=context_summary,
                raw_challenge=raw_challenge,
                question=question,
                conversation_history=conversation_history,
                use_llm=self.use_llm,
            )

        else:
            # Default: treat unknown types as subjective
            result = self.subjective_verifier.verify(
                claim_a=claim_a,
                claim_b=claim_b,
                bias_info=bias_info,
                context_summary=context_summary,
                raw_challenge=raw_challenge,
                question=question,
                conversation_history=conversation_history,
                use_llm=self.use_llm,
            )

        # Attach input metadata to result
        result["input_question_type"] = question_type
        result["input_claim_a"] = claim_a
        result["input_claim_b"] = claim_b
        result["input_question"] = question

        return result

    def verify_with_history(
        self,
        module1_output: dict,
        conversation_history: list,
    ) -> dict:
        """
        Verify with multi-turn conversation context.

        Adds conversation summary analysis and can override sycophancy
        detection if position shifts are observed across turns.

        Args:
            module1_output: Complete output from Module 1.process()
            conversation_history: List of (user_msg, assistant_msg) tuples

        Returns:
            Verification result with added multi_turn_analysis
        """
        conv_summary = self.summarizer.summarize(
            conversation_history,
            use_llm=self.use_llm and self.llm_pipeline is not None,
        )

        # For subjective questions, inject conversation summary into context
        if module1_output.get("question_type") == "subjective":
            if not module1_output.get("context_summary"):
                module1_output["context_summary"] = {}
            module1_output["context_summary"]["conversation_summary"] = conv_summary

        result = self.verify(module1_output, conversation_history=conversation_history)

        result["multi_turn_analysis"] = {
            "conversation_summary": conv_summary,
            "turn_count": len(conversation_history),
            "position_shifted_in_history": conv_summary.get("position_shifted", False),
            "historical_pressure_tactics": conv_summary.get("pressure_tactics", []),
        }

        # Override: if history shows position shift under pressure
        if (
            conv_summary.get("position_shifted", False)
            and conv_summary.get("pressure_tactics")
            and not result.get("sycophancy_detected", False)
        ):
            result["sycophancy_detected"] = True
            result["recommendation"] = "maintain_original"
            result["multi_turn_override"] = True

        return result