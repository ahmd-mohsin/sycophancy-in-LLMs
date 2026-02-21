import re
from typing import Optional

from .lc_compat import PromptTemplate, LLMChain, HuggingFacePipeline
from .config import SYCOPHANCY_REMOVAL_PATTERNS
from .bias_detector import BiasDetector


CLAIM_EXTRACTION_TEMPLATE = None
if PromptTemplate is not None:
    CLAIM_EXTRACTION_TEMPLATE = PromptTemplate(
        input_variables=["text", "question"],
        template=(
            "Extract the single core factual or opinion claim from the following text "
            "in the context of the given question. Output ONLY the extracted claim as "
            "a short declarative sentence. Do not include any preamble, explanation, "
            "or extra text.\n\n"
            "Question: {question}\n"
            "Text: {text}\n\n"
            "Core claim:"
        ),
    )

# Phrases from the prompt template — if the "claim" contains these, LLM
# extraction failed and we must fall back to regex.
_PROMPT_CONTAMINATION_PHRASES = [
    "Extract the single core",
    "Output ONLY the extracted claim",
    "Do not include any pream",
    "Core claim:",
    "short declarative sentence",
]


class ClaimExtractor:

    def __init__(self, llm_pipeline=None, bias_detector: Optional[BiasDetector] = None):
        self.llm_pipeline = llm_pipeline
        self.bias_detector = bias_detector or BiasDetector(llm_pipeline)
        self._chain = None

    def _get_chain(self):
        if self._chain is not None:
            return self._chain
        if self.llm_pipeline is not None and LLMChain is not None and CLAIM_EXTRACTION_TEMPLATE is not None:
            self._chain = LLMChain(
                llm=self.llm_pipeline,
                prompt=CLAIM_EXTRACTION_TEMPLATE,
            )
        return self._chain

    def _sanitize_user_challenge(self, text: str) -> str:
        cleaned = text.strip()
        for pattern in SYCOPHANCY_REMOVAL_PATTERNS:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)
            cleaned = regex.sub("", cleaned)
        cleaned = re.sub(r"^\s*[,.\-!?]+\s*", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = cleaned.strip()
        return cleaned

    @staticmethod
    def _is_prompt_contaminated(text: str) -> bool:
        """Check if the extracted 'claim' is actually the prompt template."""
        for phrase in _PROMPT_CONTAMINATION_PHRASES:
            if phrase.lower() in text.lower():
                return True
        return False

    def _extract_claim_regex(self, text: str, question: str) -> str:
        cleaned = text.strip()

        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        if not sentences:
            return cleaned

        question_lower = question.lower()
        question_words = set(question_lower.split())

        best_sentence = sentences[0]
        best_score = 0

        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_words = set(sentence_lower.split())
            overlap = len(question_words & sentence_words)

            has_verb = bool(re.search(r"\b(is|are|was|were|has|have|does|do)\b", sentence_lower))
            score = overlap + (2 if has_verb else 0)

            if score > best_score:
                best_score = score
                best_sentence = sentence

        return best_sentence.strip()

    def _extract_claim_llm(self, text: str, question: str) -> str:
        chain = self._get_chain()
        if chain is None:
            return self._extract_claim_regex(text, question)

        try:
            result = chain.invoke({"text": text, "question": question})
            claim = result.get("text", "").strip()

            # Validate: not empty, not too short, not the prompt itself
            if not claim or len(claim) < 3:
                return self._extract_claim_regex(text, question)
            if self._is_prompt_contaminated(claim):
                return self._extract_claim_regex(text, question)

            # Clean up: remove leading "Core claim:" or similar prefixes
            # that the LLM might echo
            for prefix in ["Core claim:", "Claim:", "The claim is:", "Answer:"]:
                if claim.lower().startswith(prefix.lower()):
                    claim = claim[len(prefix):].strip()

            if not claim or len(claim) < 3:
                return self._extract_claim_regex(text, question)

            return claim

        except Exception:
            return self._extract_claim_regex(text, question)

    def extract_claims(
        self,
        assistant_answer: str,
        user_challenge: str,
        question: str,
        use_llm: bool = True,
    ) -> dict:
        sanitized_challenge = self._sanitize_user_challenge(user_challenge)

        bias_result = self.bias_detector.neutralize(sanitized_challenge, use_llm=use_llm)
        sanitized_challenge = bias_result["neutralized"]

        if use_llm and self.llm_pipeline is not None:
            claim_a = self._extract_claim_llm(assistant_answer, question)
            claim_b = self._extract_claim_llm(sanitized_challenge, question)
        else:
            claim_a = self._extract_claim_regex(assistant_answer, question)
            claim_b = self._extract_claim_regex(sanitized_challenge, question)

        # Final safety check: if claims are identical or contaminated, use regex
        if claim_a == claim_b and len(claim_a) > 10:
            claim_a = self._extract_claim_regex(assistant_answer, question)
            claim_b = self._extract_claim_regex(sanitized_challenge, question)
        if self._is_prompt_contaminated(claim_a):
            claim_a = self._extract_claim_regex(assistant_answer, question)
        if self._is_prompt_contaminated(claim_b):
            claim_b = self._extract_claim_regex(sanitized_challenge, question)

        return {
            "claim_A": claim_a,
            "claim_B": claim_b,
            "raw_user_challenge": user_challenge,
            "sanitized_challenge": sanitized_challenge,
            "bias_detected": bias_result["was_modified"],
            "bias_info": bias_result["bias_info"],
        }