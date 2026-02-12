import re
from typing import Optional

from lc_compat import PromptTemplate, LLMChain, HuggingFacePipeline
from config import BIAS_KEYWORDS


BIAS_NEUTRALIZATION_TEMPLATE = None
if PromptTemplate is not None:
    BIAS_NEUTRALIZATION_TEMPLATE = PromptTemplate(
        input_variables=["text"],
        template=(
            "You are a bias neutralization agent. Your task is to rewrite the "
            "following text to remove any biased, leading, or manipulative language "
            "while preserving the factual content and core meaning.\n\n"
            "Rules:\n"
            "1. Remove phrases that attribute agreement or confirmation to others\n"
            "2. Remove absolutist language like 'obviously', 'clearly', 'everyone knows'\n"
            "3. Remove social pressure language\n"
            "4. Keep all factual claims and substantive arguments intact\n"
            "5. Output ONLY the rewritten text, nothing else\n\n"
            "Original text:\n{text}\n\n"
            "Neutralized text:"
        ),
    )


class BiasDetector:

    def __init__(self, llm_pipeline=None):
        self.bias_patterns = self._compile_patterns()
        self.llm_pipeline = llm_pipeline
        self._chain = None

    def _compile_patterns(self) -> list[re.Pattern]:
        patterns = []
        for keyword in BIAS_KEYWORDS:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            patterns.append(pattern)
        return patterns

    def _get_chain(self):
        if self._chain is not None:
            return self._chain
        if self.llm_pipeline is not None and LLMChain is not None and BIAS_NEUTRALIZATION_TEMPLATE is not None:
            self._chain = LLMChain(
                llm=self.llm_pipeline,
                prompt=BIAS_NEUTRALIZATION_TEMPLATE,
            )
        return self._chain

    def detect_bias(self, text: str) -> dict:
        found_keywords = []
        positions = []

        text_lower = text.lower()
        for keyword in BIAS_KEYWORDS:
            start = 0
            while True:
                idx = text_lower.find(keyword.lower(), start)
                if idx == -1:
                    break
                found_keywords.append(keyword)
                positions.append((idx, idx + len(keyword)))
                start = idx + 1

        return {
            "has_bias": len(found_keywords) > 0,
            "keywords_found": found_keywords,
            "positions": positions,
            "bias_density": len(found_keywords) / max(len(text.split()), 1),
        }

    def remove_bias_regex(self, text: str) -> str:
        cleaned = text
        for pattern in self.bias_patterns:
            cleaned = pattern.sub("", cleaned)

        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"\s*,\s*,", ",", cleaned)
        cleaned = cleaned.strip()
        cleaned = cleaned.strip(",").strip()

        return cleaned

    def remove_bias_llm(self, text: str) -> str:
        chain = self._get_chain()
        if chain is None:
            return self.remove_bias_regex(text)

        try:
            result = chain.invoke({"text": text})
            neutralized = result.get("text", "").strip()
            if not neutralized or len(neutralized) < 5:
                return self.remove_bias_regex(text)
            return neutralized
        except Exception:
            return self.remove_bias_regex(text)

    def neutralize(self, text: str, use_llm: bool = True) -> dict:
        bias_info = self.detect_bias(text)

        if not bias_info["has_bias"]:
            return {
                "original": text,
                "neutralized": text,
                "was_modified": False,
                "bias_info": bias_info,
            }

        if use_llm and self.llm_pipeline is not None:
            neutralized = self.remove_bias_llm(text)
        else:
            neutralized = self.remove_bias_regex(text)

        return {
            "original": text,
            "neutralized": neutralized,
            "was_modified": True,
            "bias_info": bias_info,
        }