import re
from typing import Optional

from module2.lc_compat import LLMChain
from module2.prompts.time_sensitive_reasoning import (
    DIRECT_KNOWLEDGE_PROMPT,
    REVERSE_VERIFICATION_PROMPT,
    CROSS_REFERENCE_PROMPT,
)
from module2.config import NUM_SOURCE_CHAINS


class SourceValidator:

    def __init__(self, llm_pipeline=None):
        self.llm_pipeline = llm_pipeline
        self._chains = {}

    def _get_chain(self, name: str, prompt):
        if name in self._chains:
            return self._chains[name]
        if self.llm_pipeline is not None and prompt is not None:
            chain = LLMChain(llm=self.llm_pipeline, prompt=prompt)
            self._chains[name] = chain
            return chain
        return None

    def _extract_entity(self, claim: str, question: str) -> str:
        combined = f"{question} {claim}"

        proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", combined)
        if proper_nouns:
            return max(proper_nouns, key=len)

        words = combined.split()
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
                      "what", "who", "which", "that", "this", "for", "and", "or",
                      "current", "currently", "latest", "now", "today"}
        content_words = [w for w in words if w.lower() not in stopwords and len(w) > 2]
        return " ".join(content_words[:3]) if content_words else claim[:50]

    def _parse_source_output(self, output: str) -> dict:
        supports = "UNCERTAIN"
        reasoning = ""
        info = ""

        for line in output.split("\n"):
            stripped = line.strip()
            upper = stripped.upper()

            if upper.startswith("SUPPORTS_CLAIM:"):
                value = upper.split(":", 1)[1].strip()
                if "YES" in value:
                    supports = "YES"
                elif "NO" in value:
                    supports = "NO"
                else:
                    supports = "UNCERTAIN"
            elif upper.startswith("REASONING:"):
                reasoning = stripped.split(":", 1)[1].strip()
            elif upper.startswith("KNOWN_INFO:") or upper.startswith("CONTRADICTIONS:") or upper.startswith("ENTITY_KNOWN_FOR:"):
                info = stripped.split(":", 1)[1].strip()

        return {
            "supports_claim": supports,
            "reasoning": reasoning,
            "info": info,
        }

    def _run_source_regex(self, claim: str, question: str) -> dict:
        return {
            "supports_claim": "UNCERTAIN",
            "reasoning": "Regex mode: no external validation available for time-sensitive claims",
            "info": "LLM required for time-sensitive verification",
        }

    def run_direct_knowledge(self, claim: str, question: str) -> dict:
        chain = self._get_chain("direct_knowledge", DIRECT_KNOWLEDGE_PROMPT)
        result = {"source_type": "direct_knowledge", "claim_evaluated": claim}

        if chain is None:
            result.update(self._run_source_regex(claim, question))
            result["method"] = "regex_fallback"
            return result

        try:
            output = chain.invoke({"claim": claim, "question": question})
            parsed = self._parse_source_output(output.get("text", ""))
            result.update(parsed)
            result["method"] = "llm"
            result["raw_output"] = output.get("text", "")
            return result
        except Exception:
            result.update(self._run_source_regex(claim, question))
            result["method"] = "regex_fallback"
            return result

    def run_reverse_verification(self, claim: str, question: str) -> dict:
        chain = self._get_chain("reverse_verification", REVERSE_VERIFICATION_PROMPT)
        result = {"source_type": "reverse_verification", "claim_evaluated": claim}

        if chain is None:
            result.update(self._run_source_regex(claim, question))
            result["method"] = "regex_fallback"
            return result

        try:
            output = chain.invoke({"claim": claim, "question": question})
            parsed = self._parse_source_output(output.get("text", ""))
            result.update(parsed)
            result["method"] = "llm"
            result["raw_output"] = output.get("text", "")
            return result
        except Exception:
            result.update(self._run_source_regex(claim, question))
            result["method"] = "regex_fallback"
            return result

    def run_cross_reference(self, claim: str, question: str) -> dict:
        entity = self._extract_entity(claim, question)
        chain = self._get_chain("cross_reference", CROSS_REFERENCE_PROMPT)
        result = {"source_type": "cross_reference", "claim_evaluated": claim, "entity": entity}

        if chain is None:
            result.update(self._run_source_regex(claim, question))
            result["method"] = "regex_fallback"
            return result

        try:
            output = chain.invoke({"claim": claim, "question": question, "entity": entity})
            parsed = self._parse_source_output(output.get("text", ""))
            result.update(parsed)
            result["method"] = "llm"
            result["raw_output"] = output.get("text", "")
            return result
        except Exception:
            result.update(self._run_source_regex(claim, question))
            result["method"] = "regex_fallback"
            return result

    def validate_claim(self, claim: str, question: str) -> list:
        sources = [
            self.run_direct_knowledge(claim, question),
            self.run_reverse_verification(claim, question),
            self.run_cross_reference(claim, question),
        ]
        return sources[:NUM_SOURCE_CHAINS]

    @staticmethod
    def aggregate_sources(source_results: list) -> dict:
        support_count = 0
        refute_count = 0
        uncertain_count = 0

        for source in source_results:
            verdict = source.get("supports_claim", "UNCERTAIN").upper()
            if verdict == "YES":
                support_count += 1
            elif verdict == "NO":
                refute_count += 1
            else:
                uncertain_count += 1

        total = len(source_results)
        return {
            "support_count": support_count,
            "refute_count": refute_count,
            "uncertain_count": uncertain_count,
            "agreement_ratio": support_count / max(total, 1),
            "total_sources": total,
        }