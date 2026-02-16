import re
from typing import Optional

from lc_compat import LLMChain
from prompts.factual_reasoning import (
    ANALYTICAL_CHAIN_PROMPT,
    ADVERSARIAL_CHAIN_PROMPT,
    KNOWLEDGE_CHAIN_PROMPT,
)
from config import NUM_REASONING_CHAINS


CHAIN_CONFIGS = [
    {
        "chain_id": 1,
        "perspective": "analytical",
        "prompt": ANALYTICAL_CHAIN_PROMPT,
    },
    {
        "chain_id": 2,
        "perspective": "adversarial",
        "prompt": ADVERSARIAL_CHAIN_PROMPT,
    },
    {
        "chain_id": 3,
        "perspective": "knowledge_based",
        "prompt": KNOWLEDGE_CHAIN_PROMPT,
    },
]


class ChainOfVerification:

    def __init__(self, llm_pipeline=None):
        self.llm_pipeline = llm_pipeline
        self._chains = {}

    def _get_chain(self, perspective: str, prompt):
        if perspective in self._chains:
            return self._chains[perspective]
        if self.llm_pipeline is not None and prompt is not None:
            chain = LLMChain(llm=self.llm_pipeline, prompt=prompt)
            self._chains[perspective] = chain
            return chain
        return None

    def _parse_analytical_output(self, output: str) -> dict:
        sub_claims = []
        verdict = "UNCERTAIN"
        reasoning = ""

        in_subclaims = False
        for line in output.split("\n"):
            stripped = line.strip()
            upper = stripped.upper()

            if "SUB_CLAIMS" in upper or "SUB-CLAIMS" in upper:
                in_subclaims = True
                continue

            if in_subclaims and stripped.startswith("-"):
                parts = stripped[1:].strip()
                sc_verdict = "UNCERTAIN"
                for v in ["CORRECT", "INCORRECT", "UNCERTAIN"]:
                    if v in parts.upper():
                        sc_verdict = v
                        break
                sc_text = re.sub(r":\s*(?:CORRECT|INCORRECT|UNCERTAIN).*$", "", parts, flags=re.IGNORECASE).strip()
                sub_claims.append({"sub_claim": sc_text, "verdict": sc_verdict})

            if upper.startswith("VERDICT:"):
                in_subclaims = False
                for v in ["CORRECT", "INCORRECT", "UNCERTAIN"]:
                    if v in upper:
                        verdict = v
                        break

            if upper.startswith("REASONING:"):
                reasoning = stripped.split(":", 1)[1].strip()

        return {
            "sub_claims": sub_claims,
            "verdict": verdict,
            "reasoning": reasoning,
        }

    def _parse_adversarial_output(self, output: str) -> dict:
        counterarguments = ""
        verdict = "UNCERTAIN"
        reasoning = ""

        for line in output.split("\n"):
            stripped = line.strip()
            upper = stripped.upper()

            if upper.startswith("COUNTERARGUMENTS:") or upper.startswith("COUNTER ARGUMENTS:"):
                counterarguments = stripped.split(":", 1)[1].strip()
            elif upper.startswith("VERDICT:"):
                for v in ["CORRECT", "INCORRECT", "UNCERTAIN"]:
                    if v in upper:
                        verdict = v
                        break
            elif upper.startswith("REASONING:"):
                reasoning = stripped.split(":", 1)[1].strip()

        return {
            "counterarguments": counterarguments,
            "verdict": verdict,
            "reasoning": reasoning,
        }

    def _parse_knowledge_output(self, output: str) -> dict:
        verdict = "UNCERTAIN"
        reasoning = ""

        for line in output.split("\n"):
            stripped = line.strip()
            upper = stripped.upper()

            if upper.startswith("VERDICT:"):
                for v in ["CORRECT", "INCORRECT", "UNCERTAIN"]:
                    if v in upper:
                        verdict = v
                        break
            elif upper.startswith("REASONING:"):
                reasoning = stripped.split(":", 1)[1].strip()

        return {
            "verdict": verdict,
            "reasoning": reasoning,
        }

    def _run_chain_regex(self, claim: str, question: str, perspective: str) -> dict:
        claim_lower = claim.lower()
        question_lower = question.lower()

        question_words = set(re.findall(r"\b\w+\b", question_lower))
        claim_words = set(re.findall(r"\b\w+\b", claim_lower))
        overlap = len(question_words & claim_words)

        has_factual_verb = bool(re.search(
            r"\b(?:is|are|was|were|has|have|equals?|contains?|consists?)\b",
            claim_lower
        ))

        has_number = bool(re.search(r"\d+", claim))

        score = overlap + (2 if has_factual_verb else 0) + (1 if has_number else 0)

        if score >= 3:
            verdict = "CORRECT"
        elif score >= 1:
            verdict = "UNCERTAIN"
        else:
            verdict = "UNCERTAIN"

        result = {
            "verdict": verdict,
            "reasoning": f"Regex heuristic: overlap={overlap}, factual_verb={has_factual_verb}, score={score}",
        }

        if perspective == "analytical":
            result["sub_claims"] = [{"sub_claim": claim, "verdict": verdict}]
        elif perspective == "adversarial":
            result["counterarguments"] = "Regex mode: no adversarial analysis available"

        return result

    def run_single_chain(self, claim: str, question: str, chain_config: dict) -> dict:
        perspective = chain_config["perspective"]
        prompt = chain_config["prompt"]
        chain_id = chain_config["chain_id"]

        chain = self._get_chain(perspective, prompt)

        base_result = {
            "chain_id": chain_id,
            "perspective": perspective,
            "claim_evaluated": claim,
        }

        if chain is None:
            regex_result = self._run_chain_regex(claim, question, perspective)
            base_result.update(regex_result)
            base_result["method"] = "regex_fallback"
            return base_result

        try:
            result = chain.invoke({"claim": claim, "question": question})
            output = result.get("text", "").strip()

            if perspective == "analytical":
                parsed = self._parse_analytical_output(output)
            elif perspective == "adversarial":
                parsed = self._parse_adversarial_output(output)
            else:
                parsed = self._parse_knowledge_output(output)

            base_result.update(parsed)
            base_result["method"] = "llm"
            base_result["raw_output"] = output
            return base_result

        except Exception:
            regex_result = self._run_chain_regex(claim, question, perspective)
            base_result.update(regex_result)
            base_result["method"] = "regex_fallback"
            return base_result

    def run_all_chains(self, claim: str, question: str) -> list:
        results = []
        for config in CHAIN_CONFIGS[:NUM_REASONING_CHAINS]:
            result = self.run_single_chain(claim, question, config)
            results.append(result)
        return results