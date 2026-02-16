import re
from typing import Optional

from module2.lc_compat import LLMChain
from module2.prompts.subjective_reasoning import (
    OPINION_SHIFT_JUDGE_PROMPT,
    SUBSTANCE_EVALUATOR_PROMPT,
    CLEAN_EVALUATION_PROMPT,
    PRESSURED_EVALUATION_PROMPT,
    UNBIASED_OPINION_PROMPT,
)
from module2.config import (
    HIGH_PRESSURE_RISK_LEVELS,
    PRESSURE_TYPES_REQUIRING_SUBSTANCE,
    SUBSTANTIVE_SIGNAL_TYPES,
)


class OpinionShiftDetector:

    def __init__(self, llm_pipeline=None):
        self.llm_pipeline = llm_pipeline
        self._judge_chain = None
        self._substance_chain = None
        self._clean_eval_chain = None
        self._pressured_eval_chain = None
        self._unbiased_chain = None

    def _get_chain(self, attr_name, prompt_template):
        cached = getattr(self, attr_name, None)
        if cached is not None:
            return cached
        if self.llm_pipeline is not None and prompt_template is not None:
            chain = LLMChain(llm=self.llm_pipeline, prompt=prompt_template)
            setattr(self, attr_name, chain)
            return chain
        return None

    def _get_judge_chain(self):
        return self._get_chain("_judge_chain", OPINION_SHIFT_JUDGE_PROMPT)

    def _get_substance_chain(self):
        return self._get_chain("_substance_chain", SUBSTANCE_EVALUATOR_PROMPT)

    def _get_clean_eval_chain(self):
        return self._get_chain("_clean_eval_chain", CLEAN_EVALUATION_PROMPT)

    def _get_pressured_eval_chain(self):
        return self._get_chain("_pressured_eval_chain", PRESSURED_EVALUATION_PROMPT)

    def _get_unbiased_chain(self):
        return self._get_chain("_unbiased_chain", UNBIASED_OPINION_PROMPT)

    def analyze_pressure(self, bias_info: dict) -> dict:
        risk_level = bias_info.get("sycophancy_risk", "none")
        implicit = bias_info.get("implicit_bias", {})

        pressure_types = []
        for match in implicit.get("opinion_steering", []):
            pressure_types.append(match.get("pattern", "opinion_steering"))
        for match in implicit.get("authority_pressure", []):
            pressure_types.append(match.get("pattern", "authority_pressure"))
        for match in implicit.get("emotional_framing", []):
            pressure_types.append(match.get("pattern", "emotional_framing"))

        llm_judge = bias_info.get("llm_judge", {})
        for key in ("opinion_steering", "authority_pressure", "emotional_pressure", "identity_framing"):
            if llm_judge.get(key) and key not in pressure_types:
                pressure_types.append(key)

        pressure_only_types = [p for p in pressure_types if p in PRESSURE_TYPES_REQUIRING_SUBSTANCE]
        substantive_types = [p for p in pressure_types if p in SUBSTANTIVE_SIGNAL_TYPES]

        is_high_pressure = risk_level in HIGH_PRESSURE_RISK_LEVELS
        is_pressure_only = len(pressure_only_types) > 0 and len(substantive_types) == 0

        return {
            "sycophancy_risk": risk_level,
            "pressure_types": pressure_types,
            "pressure_only_types": pressure_only_types,
            "substantive_types": substantive_types,
            "is_high_pressure": is_high_pressure,
            "is_pressure_only": is_pressure_only,
            "total_signals": implicit.get("total_signals", 0),
        }

    def evaluate_substance_regex(self, raw_challenge: str, pressure_types: list) -> dict:
        authority_number_pattern = re.compile(
            r"(?:\d+\s*(?:years?|decades?)\s*(?:of\s+)?experience)|"
            r"(?:(?:PhD|Ph\.D|doctorate|master|professor|Dr)\b)",
            re.IGNORECASE
        )
        cleaned_for_data = authority_number_pattern.sub("", raw_challenge)
        has_data = bool(re.search(r"\d+(?:\.\d+)?%", cleaned_for_data)) or bool(
            re.search(r"\b\d{2,}\b", cleaned_for_data)
            and not re.search(r"\b\d{4}\b", cleaned_for_data)
        )

        has_citation = bool(re.search(
            r"(?:according to|study|research paper|journal|published in|data shows|statistics show|survey found)",
            raw_challenge,
            re.IGNORECASE
        ))
        has_logical_argument = bool(re.search(
            r"(?:because|therefore|since|given that|considering that|the reason is|this means|it follows that|which leads to|as a result)",
            raw_challenge,
            re.IGNORECASE
        ))
        has_example = bool(re.search(
            r"(?:for example|for instance|such as|like when|consider the case|a concrete example)",
            raw_challenge,
            re.IGNORECASE
        ))

        substance_count = sum([has_data, has_citation, has_logical_argument, has_example])
        has_substance = substance_count >= 1

        pressure_count = len(pressure_types)
        pressure_dominant = pressure_count > substance_count and pressure_count >= 2

        return {
            "has_substance": has_substance,
            "substance_signals": {
                "has_data": has_data,
                "has_citation": has_citation,
                "has_logical_argument": has_logical_argument,
                "has_example": has_example,
            },
            "substance_count": substance_count,
            "pressure_dominant": pressure_dominant,
        }

    def evaluate_substance_llm(self, raw_challenge: str, pressure_types: list) -> dict:
        chain = self._get_substance_chain()
        if chain is None:
            return self.evaluate_substance_regex(raw_challenge, pressure_types)

        try:
            result = chain.invoke({
                "user_challenge": raw_challenge,
                "pressure_types": ", ".join(pressure_types),
            })
            output = result.get("text", "").strip()

            parsed = {
                "has_substance": False,
                "substance_description": "",
                "pressure_only": True,
                "raw_output": output,
            }

            for line in output.split("\n"):
                upper = line.strip().upper()
                if upper.startswith("HAS_SUBSTANCE:"):
                    parsed["has_substance"] = "YES" in upper
                elif upper.startswith("SUBSTANCE_DESCRIPTION:"):
                    parsed["substance_description"] = line.strip().split(":", 1)[1].strip()
                elif upper.startswith("PRESSURE_ONLY:"):
                    parsed["pressure_only"] = "YES" in upper

            regex_result = self.evaluate_substance_regex(raw_challenge, pressure_types)
            parsed["substance_signals"] = regex_result["substance_signals"]
            parsed["substance_count"] = regex_result["substance_count"]
            parsed["pressure_dominant"] = not parsed["has_substance"]

            return parsed
        except Exception:
            return self.evaluate_substance_regex(raw_challenge, pressure_types)

    def _parse_evaluation_output(self, output: str) -> dict:
        parsed = {"assessment": "PARTIAL", "position": "", "reasoning": ""}
        for line in output.split("\n"):
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("ASSESSMENT:"):
                value = upper.split(":", 1)[1].strip()
                for a in ["AGREE", "DISAGREE", "PARTIAL"]:
                    if a in value:
                        parsed["assessment"] = a
                        break
            elif upper.startswith("POSITION:"):
                parsed["position"] = stripped.split(":", 1)[1].strip()
            elif upper.startswith("REASONING:"):
                parsed["reasoning"] = stripped.split(":", 1)[1].strip()
        return parsed

    def dual_evaluate_llm(self, question: str, clean_challenge: str, raw_challenge: str) -> dict:
        clean_chain = self._get_clean_eval_chain()
        pressured_chain = self._get_pressured_eval_chain()

        if clean_chain is None or pressured_chain is None:
            return self.dual_evaluate_regex(question, clean_challenge, raw_challenge)

        try:
            clean_result = clean_chain.invoke({
                "question": question,
                "clean_challenge": clean_challenge,
            })
            clean_parsed = self._parse_evaluation_output(clean_result.get("text", ""))

            pressured_result = pressured_chain.invoke({
                "question": question,
                "raw_challenge": raw_challenge,
            })
            pressured_parsed = self._parse_evaluation_output(pressured_result.get("text", ""))

            outputs_differ = clean_parsed["assessment"] != pressured_parsed["assessment"]

            return {
                "clean_evaluation": clean_parsed,
                "pressured_evaluation": pressured_parsed,
                "outputs_differ": outputs_differ,
                "sycophancy_signal": outputs_differ,
                "method": "llm",
            }
        except Exception:
            return self.dual_evaluate_regex(question, clean_challenge, raw_challenge)

    def dual_evaluate_regex(self, question: str, clean_challenge: str, raw_challenge: str) -> dict:
        clean_word_count = len(clean_challenge.split())
        raw_word_count = len(raw_challenge.split())
        reduction_ratio = 1 - (clean_word_count / max(raw_word_count, 1))

        pressure_keywords = [
            "professor", "phd", "expert", "obviously", "clearly",
            "everyone knows", "are you sure", "you're wrong",
            "disappointed", "any reasonable person",
        ]
        pressure_word_count = sum(
            1 for kw in pressure_keywords if kw in raw_challenge.lower()
        )

        significant_pressure = pressure_word_count >= 2 or (reduction_ratio > 0.4 and pressure_word_count >= 1)

        return {
            "clean_evaluation": {
                "assessment": "PARTIAL",
                "position": clean_challenge[:100],
                "reasoning": "Regex mode: cannot evaluate substance without LLM",
            },
            "pressured_evaluation": {
                "assessment": "AGREE" if significant_pressure else "PARTIAL",
                "position": raw_challenge[:100],
                "reasoning": f"Regex detected {pressure_word_count} pressure keywords, {reduction_ratio:.0%} content reduction",
            },
            "outputs_differ": significant_pressure,
            "sycophancy_signal": significant_pressure,
            "method": "regex",
        }

    def generate_unbiased_opinion_llm(
        self, question: str, claim_a: str, claim_b: str, sycophancy_analysis: str
    ) -> dict:
        chain = self._get_unbiased_chain()
        if chain is None:
            return self.generate_unbiased_opinion_regex(claim_a, claim_b)

        try:
            result = chain.invoke({
                "question": question,
                "claim_a": claim_a,
                "claim_b": claim_b,
                "sycophancy_analysis": sycophancy_analysis,
            })
            output = result.get("text", "").strip()

            parsed = {
                "unbiased_position": "",
                "claim_a_merit": "",
                "claim_b_merit": "",
                "recommendation": "BALANCED",
                "raw_output": output,
                "method": "llm",
            }

            for line in output.split("\n"):
                stripped = line.strip()
                upper = stripped.upper()
                if upper.startswith("UNBIASED_POSITION:"):
                    parsed["unbiased_position"] = stripped.split(":", 1)[1].strip()
                elif upper.startswith("CLAIM_A_MERIT:"):
                    parsed["claim_a_merit"] = stripped.split(":", 1)[1].strip()
                elif upper.startswith("CLAIM_B_MERIT:"):
                    parsed["claim_b_merit"] = stripped.split(":", 1)[1].strip()
                elif upper.startswith("RECOMMENDATION:"):
                    value = upper.split(":", 1)[1].strip()
                    for r in ["FAVOR_A", "FAVOR_B", "BALANCED"]:
                        if r in value:
                            parsed["recommendation"] = r
                            break

            return parsed
        except Exception:
            return self.generate_unbiased_opinion_regex(claim_a, claim_b)

    def generate_unbiased_opinion_regex(self, claim_a: str, claim_b: str) -> dict:
        return {
            "unbiased_position": f"Both perspectives have merit. Position A: {claim_a[:80]}. Position B: {claim_b[:80]}.",
            "claim_a_merit": claim_a[:100],
            "claim_b_merit": claim_b[:100],
            "recommendation": "BALANCED",
            "method": "regex",
        }

    def judge_shift_regex(self, pressure_analysis: dict, substance_analysis: dict) -> dict:
        is_high_pressure = pressure_analysis["is_high_pressure"]
        has_substance = substance_analysis.get("has_substance", False)
        pressure_dominant = substance_analysis.get("pressure_dominant", False)

        if is_high_pressure and not has_substance:
            sycophancy_likely = True
            recommendation = "generate_unbiased"
            reasoning = "High sycophancy pressure detected without substantive new information"
        elif is_high_pressure and has_substance and not pressure_dominant:
            sycophancy_likely = False
            recommendation = "acknowledge_both"
            reasoning = "Pressure detected but user provided substantive arguments"
        elif is_high_pressure and has_substance and pressure_dominant:
            sycophancy_likely = True
            recommendation = "generate_unbiased"
            reasoning = "High pressure dominates despite some substantive content"
        elif has_substance and not is_high_pressure:
            sycophancy_likely = False
            recommendation = "accept_correction" if substance_analysis.get("substance_count", 0) >= 2 else "acknowledge_both"
            reasoning = "User provided substantive arguments without significant pressure"
        elif pressure_analysis["is_pressure_only"] and not has_substance:
            sycophancy_likely = True
            recommendation = "generate_unbiased"
            reasoning = "User relied on pressure tactics without new information"
        else:
            sycophancy_likely = False
            recommendation = "acknowledge_both"
            reasoning = "Insufficient pressure signals to determine sycophancy"

        return {
            "sycophancy_likely": sycophancy_likely,
            "substantive_new_info": has_substance,
            "reasoning": reasoning,
            "recommendation": recommendation,
            "method": "regex",
        }

    def judge_shift_llm(
        self, claim_a: str, claim_b: str, pressure_analysis: dict, substance_analysis: dict
    ) -> dict:
        chain = self._get_judge_chain()
        if chain is None:
            return self.judge_shift_regex(pressure_analysis, substance_analysis)

        try:
            result = chain.invoke({
                "claim_a": claim_a,
                "claim_b": claim_b,
                "pressure_types": ", ".join(pressure_analysis["pressure_types"]),
                "pressure_risk": pressure_analysis["sycophancy_risk"],
            })
            output = result.get("text", "").strip()

            parsed = {
                "sycophancy_likely": False,
                "substantive_new_info": False,
                "reasoning": "",
                "recommendation": "acknowledge_both",
                "method": "llm",
                "raw_output": output,
            }

            for line in output.split("\n"):
                upper = line.strip().upper()
                if upper.startswith("SYCOPHANCY_LIKELY:"):
                    parsed["sycophancy_likely"] = "YES" in upper
                elif upper.startswith("SUBSTANTIVE_NEW_INFO:"):
                    parsed["substantive_new_info"] = "YES" in upper
                elif upper.startswith("REASONING:"):
                    parsed["reasoning"] = line.strip().split(":", 1)[1].strip()
                elif upper.startswith("RECOMMENDATION:"):
                    value = upper.split(":", 1)[1].strip()
                    if "MAINTAIN" in value:
                        parsed["recommendation"] = "generate_unbiased"
                    elif "ACCEPT" in value:
                        parsed["recommendation"] = "accept_correction"
                    else:
                        parsed["recommendation"] = "acknowledge_both"

            return parsed
        except Exception:
            return self.judge_shift_regex(pressure_analysis, substance_analysis)

    def detect_shift(
        self,
        claim_a: str,
        claim_b: str,
        bias_info: dict,
        raw_challenge: str = "",
        question: str = "",
        use_llm: bool = True,
    ) -> dict:
        pressure_analysis = self.analyze_pressure(bias_info)

        if use_llm and self.llm_pipeline is not None:
            substance_analysis = self.evaluate_substance_llm(
                raw_challenge, pressure_analysis["pressure_types"]
            )
        else:
            substance_analysis = self.evaluate_substance_regex(
                raw_challenge, pressure_analysis["pressure_types"]
            )

        if use_llm and self.llm_pipeline is not None:
            dual_eval = self.dual_evaluate_llm(question, claim_b, raw_challenge)
        else:
            dual_eval = self.dual_evaluate_regex(question, claim_b, raw_challenge)

        if use_llm and self.llm_pipeline is not None:
            shift_judgment = self.judge_shift_llm(
                claim_a, claim_b, pressure_analysis, substance_analysis
            )
        else:
            shift_judgment = self.judge_shift_regex(pressure_analysis, substance_analysis)

        if (
            dual_eval["sycophancy_signal"]
            and not shift_judgment["sycophancy_likely"]
            and pressure_analysis["is_high_pressure"]
        ):
            shift_judgment["sycophancy_likely"] = True
            shift_judgment["recommendation"] = "generate_unbiased"
            shift_judgment["reasoning"] += " (dual-evaluation divergence detected)"

        unbiased_output = None
        if shift_judgment["sycophancy_likely"]:
            sycophancy_summary = (
                f"Pressure types: {', '.join(pressure_analysis['pressure_types'])}. "
                f"Risk: {pressure_analysis['sycophancy_risk']}. "
                f"Dual-eval divergence: {dual_eval['sycophancy_signal']}."
            )
            if use_llm and self.llm_pipeline is not None:
                unbiased_output = self.generate_unbiased_opinion_llm(
                    question, claim_a, claim_b, sycophancy_summary
                )
            else:
                unbiased_output = self.generate_unbiased_opinion_regex(claim_a, claim_b)

        return {
            "pressure_analysis": pressure_analysis,
            "substance_analysis": substance_analysis,
            "dual_evaluation": dual_eval,
            "shift_judgment": shift_judgment,
            "unbiased_output": unbiased_output,
        }