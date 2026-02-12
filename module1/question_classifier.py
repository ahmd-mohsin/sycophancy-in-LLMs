import re
from typing import Optional

from lc_compat import PromptTemplate, LLMChain, HuggingFacePipeline
from config import TEMPORAL_MARKERS, FACTUAL_INDICATOR_KEYWORDS


CLASSIFICATION_TEMPLATE = None
if PromptTemplate is not None:
    CLASSIFICATION_TEMPLATE = PromptTemplate(
        input_variables=["question", "user_challenge"],
        template=(
            "Classify the following question into exactly one category.\n\n"
            "Categories:\n"
            "- factual: Has a single objectively correct answer based on established "
            "facts, math, science, or history\n"
            "- time_sensitive: Requires current or recent information that may change "
            "over time\n"
            "- subjective: Involves opinions, preferences, recommendations, or matters "
            "without a single correct answer\n\n"
            "Question: {question}\n"
            "User's response: {user_challenge}\n\n"
            "Output ONLY one word: factual, time_sensitive, or subjective\n\n"
            "Classification:"
        ),
    )

VALID_TYPES = {"factual", "time_sensitive", "subjective"}


class QuestionClassifier:

    def __init__(self, llm_pipeline=None):
        self.llm_pipeline = llm_pipeline
        self._chain = None

    def _get_chain(self):
        if self._chain is not None:
            return self._chain
        if self.llm_pipeline is not None and LLMChain is not None and CLASSIFICATION_TEMPLATE is not None:
            self._chain = LLMChain(
                llm=self.llm_pipeline,
                prompt=CLASSIFICATION_TEMPLATE,
            )
        return self._chain

    def _check_temporal(self, question: str) -> bool:
        q_lower = question.lower()
        for marker in TEMPORAL_MARKERS:
            if re.search(r"\b" + re.escape(marker.lower()) + r"\b", q_lower):
                return True
        return False

    def _calculate_factual_score(self, question: str) -> int:
        q_lower = question.lower()
        score = 0

        for keyword in FACTUAL_INDICATOR_KEYWORDS["single_answer"]:
            if keyword.lower() in q_lower:
                score += 1
                break

        for keyword in FACTUAL_INDICATOR_KEYWORDS["calculation"]:
            if keyword.lower() in q_lower:
                score += 1
                break

        for keyword in FACTUAL_INDICATOR_KEYWORDS["historical"]:
            if keyword.lower() in q_lower:
                score += 1
                break

        for keyword in FACTUAL_INDICATOR_KEYWORDS["scientific"]:
            if keyword.lower() in q_lower:
                score += 1
                break

        return score

    def classify_rule_based(self, question: str, user_challenge: str = "") -> str:
        combined = f"{question} {user_challenge}"

        if self._check_temporal(combined):
            return "time_sensitive"

        factual_score = self._calculate_factual_score(combined)
        if factual_score >= 2:
            return "factual"

        if factual_score == 1:
            subjective_indicators = [
                "best", "worst", "should", "recommend", "opinion",
                "prefer", "favorite", "think", "feel", "better",
                "which do you", "what do you think", "advice",
            ]
            q_lower = question.lower()
            has_subjective = any(ind in q_lower for ind in subjective_indicators)
            if not has_subjective:
                return "factual"

        return "subjective"

    def classify_llm(self, question: str, user_challenge: str = "") -> str:
        chain = self._get_chain()
        if chain is None:
            return self.classify_rule_based(question, user_challenge)

        try:
            result = chain.invoke({
                "question": question,
                "user_challenge": user_challenge,
            })
            classification = result.get("text", "").strip().lower()
            classification = re.sub(r"[^a-z_]", "", classification)

            if classification in VALID_TYPES:
                return classification

            for valid_type in VALID_TYPES:
                if valid_type in classification:
                    return valid_type

            return self.classify_rule_based(question, user_challenge)
        except Exception:
            return self.classify_rule_based(question, user_challenge)

    def classify(
        self, question: str, user_challenge: str = "", use_llm: bool = True
    ) -> dict:
        rule_based = self.classify_rule_based(question, user_challenge)

        if use_llm and self.llm_pipeline is not None:
            llm_result = self.classify_llm(question, user_challenge)
            if rule_based == llm_result:
                final = rule_based
                method = "consensus"
            else:
                final = rule_based
                method = "rule_based_override"
        else:
            final = rule_based
            llm_result = None
            method = "rule_based_only"

        return {
            "question_type": final,
            "rule_based_result": rule_based,
            "llm_result": llm_result,
            "method": method,
        }