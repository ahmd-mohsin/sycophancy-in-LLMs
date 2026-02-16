import re
from typing import Optional

from module2.lc_compat import LLMChain
from module2.prompts.subjective_reasoning import (
    OPINION_SHIFT_JUDGE_PROMPT,
    SUBSTANCE_EVALUATOR_PROMPT,
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

    def _get_judge_chain(self):
        if self._judge_chain is not None:
            return self._judge_chain
        if self.llm_pipeline is not None and OPINION_SHIFT_JUDGE_PROMPT is not None:
            self._judge_chain = LLMChain(
                llm=self.llm_pipeline,
                prompt=OPINION_SHIFT_JUDGE_PROMPT,
            )
        return self._judge_chain

    def _get_substance_chain(self):
        if self._substance_chain is not None:
            return self._substance_chain
        if self.llm_pipeline is not None and SUBSTANCE_EVALUATOR_PROMPT is not None:
            self._substance_chain = LLMChain(
                llm=self.llm_pipeline,
                prompt=SUBSTANCE_EVALUATOR_PROMPT,
            )
        return self._substance_chain

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
        if llm_judge.get("opinion_steering"):
            if "opinion_steering" not in pressure_types:
                pressure_types.append("opinion_steering")
        if llm_judge.get("authority_pressure"):
            if "authority_pressure" not in pressure_types:
                pressure_types.append("authority_pressure")
        if llm_judge.get("emotional_pressure"):
            if "emotional_pressure" not in pressure_types:
                pressure_types.append("emotional_pressure")
        if llm_judge.get("identity_framing"):
            if "identity_framing" not in pressure_types:
                pressure_types.append("identity_framing")

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

    def judge_shift_regex(self, pressure_analysis: dict, substance_analysis: dict) -> dict:
        is_high_pressure = pressure_analysis["is_high_pressure"]
        is_pressure_only = pressure_analysis["is_pressure_only"]
        has_substance = substance_analysis.get("has_substance", False)
        pressure_dominant = substance_analysis.get("pressure_dominant", False)

        if is_high_pressure and not has_substance:
            sycophancy_likely = True
            recommendation = "maintain_original"
            reasoning = "High sycophancy pressure detected without substantive new information"
        elif is_high_pressure and has_substance and not pressure_dominant:
            sycophancy_likely = False
            recommendation = "acknowledge_both"
            reasoning = "Pressure detected but user provided substantive arguments"
        elif is_high_pressure and has_substance and pressure_dominant:
            sycophancy_likely = True
            recommendation = "maintain_original"
            reasoning = "High pressure dominates despite some substantive content"
        elif has_substance and not is_high_pressure:
            sycophancy_likely = False
            recommendation = "accept_correction" if substance_analysis.get("substance_count", 0) >= 2 else "acknowledge_both"
            reasoning = "User provided substantive arguments without significant pressure"
        elif is_pressure_only and not has_substance:
            sycophancy_likely = True
            recommendation = "maintain_original"
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
        self,
        claim_a: str,
        claim_b: str,
        pressure_analysis: dict,
        substance_analysis: dict,
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
                        parsed["recommendation"] = "maintain_original"
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
            shift_judgment = self.judge_shift_llm(
                claim_a, claim_b, pressure_analysis, substance_analysis
            )
        else:
            shift_judgment = self.judge_shift_regex(pressure_analysis, substance_analysis)

        return {
            "pressure_analysis": pressure_analysis,
            "substance_analysis": substance_analysis,
            "shift_judgment": shift_judgment,
        }