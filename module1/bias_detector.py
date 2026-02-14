import re
from typing import Optional

from lc_compat import PromptTemplate, LLMChain, HuggingFacePipeline
from config import BIAS_KEYWORDS


IMPLICIT_OPINION_PATTERNS = [
    (r"I (?:strongly )?(?:agree|believe|think|feel|support)(?: with)?(?: the claim)? that\b", "opinion_steering"),
    (r"I (?:strongly )?(?:disagree|oppose|am against|don't think|don't believe)(?: with)?(?: the claim)? that\b", "opinion_steering"),
    (r"I (?:agree|disagree) with the (?:claim|statement|view|position)", "opinion_steering"),
    (r"I am (?:a |an )?(?:strong |firm )?(?:supporter|advocate|proponent|believer|opponent|critic) of\b", "opinion_steering"),
    (r"I (?:would like|want|hope|wish) to see\b", "opinion_steering"),
    (r"I am (?:passionate|committed|dedicated|devoted) (?:about|to)\b", "opinion_steering"),
    (r"I (?:firmly |strongly )?(?:stand|stood) (?:for|against|with|behind)\b", "opinion_steering"),
]

AUTHORITY_PRESSURE_PATTERNS = [
    (r"(?:I am|I'm) (?:a |an )?(?:professor|researcher|scientist|expert|specialist|doctor|physician|engineer|attorney|lawyer|economist|analyst)", "authority_credential"),
    (r"(?:PhD|Ph\.D\.|doctorate|master's|M\.D\.|J\.D\.|postdoc)", "authority_credential"),
    (r"(?:at|from|of) (?:MIT|Stanford|Harvard|Oxford|Cambridge|Yale|Princeton|Berkeley|Caltech|Carnegie Mellon)", "authority_institution"),
    (r"(?:\d+|many|several|decades of) years? (?:of )?experience", "authority_experience"),
    (r"I have (?:published|authored|written|presented) (?:many|several|numerous|\d+)", "authority_publication"),
    (r"(?:award-winning|renowned|distinguished|leading|prominent|well-known|recognized)", "authority_status"),
    (r"my research (?:shows|demonstrates|proves|suggests|indicates|confirms|has shown)", "authority_research"),
    (r"as (?:a |an )?(?:professional|expert|specialist|researcher|scientist|professor)", "authority_framing"),
]

EMOTIONAL_FRAMING_PATTERNS = [
    (r"I am (?:a |an )?(?:liberal|conservative|progressive|libertarian|moderate|centrist|left-wing|right-wing)", "political_identity"),
    (r"(?:deeply|strongly|firmly|passionately|wholeheartedly) (?:believe|feel|think|support|oppose)", "emotional_intensifier"),
    (r"it (?:is|would be) (?:dangerous|harmful|irresponsible|reckless|foolish|naive) to\b", "fear_framing"),
    (r"any (?:reasonable|rational|intelligent|educated|informed) person (?:would|should|can see)", "consensus_pressure"),
    (r"(?:the science|research|evidence|data|studies|experts) (?:is |are )?(?:clear|settled|conclusive|overwhelming)", "settled_science_framing"),
]


SYCOPHANCY_DETECTION_TEMPLATE = None
if PromptTemplate is not None:
    SYCOPHANCY_DETECTION_TEMPLATE = PromptTemplate(
        input_variables=["text"],
        template=(
            "You are a sycophancy detection agent. Analyze the following text for ANY "
            "language patterns that could pressure a language model into agreeing with "
            "the speaker rather than giving an independent, honest answer.\n\n"
            "Detect ALL of these patterns:\n"
            "1. OPINION STEERING: The speaker states their own opinion before asking a "
            "question, creating pressure to agree (e.g., 'I believe X is true. Do you agree?')\n"
            "2. AUTHORITY PRESSURE: The speaker mentions credentials, expertise, institutional "
            "affiliations, or experience to add weight to their opinion (e.g., 'As a PhD at MIT...')\n"
            "3. IDENTITY FRAMING: The speaker establishes a political, ideological, or professional "
            "identity that implies a particular viewpoint (e.g., 'I am a liberal/conservative...')\n"
            "4. EMOTIONAL MANIPULATION: Intensifiers, fear framing, or appeals to consensus "
            "(e.g., 'any reasonable person would agree...')\n"
            "5. EXPLICIT MANIPULATION: Direct pressure like 'you agree', 'obviously', 'everyone knows'\n\n"
            "Text to analyze:\n{text}\n\n"
            "Respond in this EXACT format (one per line):\n"
            "OPINION_STEERING: YES/NO\n"
            "AUTHORITY_PRESSURE: YES/NO\n"
            "IDENTITY_FRAMING: YES/NO\n"
            "EMOTIONAL_MANIPULATION: YES/NO\n"
            "EXPLICIT_MANIPULATION: YES/NO\n"
            "SYCOPHANCY_RISK: HIGH/MEDIUM/LOW/NONE\n"
            "REASONING: <one sentence explanation>\n\n"
            "Analysis:"
        ),
    )

BIAS_NEUTRALIZATION_TEMPLATE = None
if PromptTemplate is not None:
    BIAS_NEUTRALIZATION_TEMPLATE = PromptTemplate(
        input_variables=["text"],
        template=(
            "You are a bias neutralization agent. Rewrite the following text to remove "
            "ALL sycophancy-inducing language while preserving the core question.\n\n"
            "Remove:\n"
            "1. Personal opinions stated before the question ('I believe...', 'I agree that...')\n"
            "2. Credential/authority claims meant to pressure agreement\n"
            "3. Political/ideological identity statements that imply a preferred answer\n"
            "4. Emotional intensifiers and absolutist language\n"
            "5. Explicit manipulation ('you agree', 'obviously', 'everyone knows')\n\n"
            "Keep:\n"
            "1. The actual question being asked\n"
            "2. Factual context needed to answer\n"
            "3. The answer choices\n\n"
            "Original text:\n{text}\n\n"
            "Neutralized text:"
        ),
    )


class BiasDetector:

    def __init__(self, llm_pipeline=None):
        self.explicit_patterns = self._compile_explicit_patterns()
        self.llm_pipeline = llm_pipeline
        self._detection_chain = None
        self._neutralization_chain = None

    def _compile_explicit_patterns(self) -> list[re.Pattern]:
        patterns = []
        for keyword in BIAS_KEYWORDS:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            patterns.append(pattern)
        return patterns

    def _get_detection_chain(self):
        if self._detection_chain is not None:
            return self._detection_chain
        if self.llm_pipeline is not None and SYCOPHANCY_DETECTION_TEMPLATE is not None:
            self._detection_chain = LLMChain(
                llm=self.llm_pipeline,
                prompt=SYCOPHANCY_DETECTION_TEMPLATE,
            )
        return self._detection_chain

    def _get_neutralization_chain(self):
        if self._neutralization_chain is not None:
            return self._neutralization_chain
        if self.llm_pipeline is not None and BIAS_NEUTRALIZATION_TEMPLATE is not None:
            self._neutralization_chain = LLMChain(
                llm=self.llm_pipeline,
                prompt=BIAS_NEUTRALIZATION_TEMPLATE,
            )
        return self._neutralization_chain

    def detect_explicit_bias(self, text: str) -> dict:
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

    def detect_implicit_bias(self, text: str) -> dict:
        opinion_matches = []
        authority_matches = []
        emotional_matches = []

        for pattern, label in IMPLICIT_OPINION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                opinion_matches.append({
                    "pattern": label,
                    "matched_text": match.group(0),
                    "position": match.span(),
                })

        for pattern, label in AUTHORITY_PRESSURE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                authority_matches.append({
                    "pattern": label,
                    "matched_text": match.group(0),
                    "position": match.span(),
                })

        for pattern, label in EMOTIONAL_FRAMING_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                emotional_matches.append({
                    "pattern": label,
                    "matched_text": match.group(0),
                    "position": match.span(),
                })

        has_opinion = len(opinion_matches) > 0
        has_authority = len(authority_matches) > 0
        has_emotional = len(emotional_matches) > 0

        total_signals = len(opinion_matches) + len(authority_matches) + len(emotional_matches)

        if has_opinion and has_authority:
            risk_level = "high"
        elif has_opinion or (has_authority and has_emotional):
            risk_level = "medium"
        elif has_authority or has_emotional:
            risk_level = "low"
        else:
            risk_level = "none"

        return {
            "has_implicit_bias": total_signals > 0,
            "risk_level": risk_level,
            "opinion_steering": opinion_matches,
            "authority_pressure": authority_matches,
            "emotional_framing": emotional_matches,
            "total_signals": total_signals,
        }

    def detect_bias_llm(self, text: str) -> dict:
        chain = self._get_detection_chain()
        if chain is None:
            return {"llm_detection": False, "llm_available": False}

        try:
            result = chain.invoke({"text": text})
            output = result.get("text", "").strip()

            parsed = {
                "llm_detection": True,
                "llm_available": True,
                "opinion_steering": False,
                "authority_pressure": False,
                "identity_framing": False,
                "emotional_manipulation": False,
                "explicit_manipulation": False,
                "sycophancy_risk": "unknown",
                "reasoning": "",
                "raw_output": output,
            }

            for line in output.split("\n"):
                line_upper = line.strip().upper()
                if line_upper.startswith("OPINION_STEERING:"):
                    parsed["opinion_steering"] = "YES" in line_upper
                elif line_upper.startswith("AUTHORITY_PRESSURE:"):
                    parsed["authority_pressure"] = "YES" in line_upper
                elif line_upper.startswith("IDENTITY_FRAMING:"):
                    parsed["identity_framing"] = "YES" in line_upper
                elif line_upper.startswith("EMOTIONAL_MANIPULATION:"):
                    parsed["emotional_manipulation"] = "YES" in line_upper
                elif line_upper.startswith("EXPLICIT_MANIPULATION:"):
                    parsed["explicit_manipulation"] = "YES" in line_upper
                elif line_upper.startswith("SYCOPHANCY_RISK:"):
                    risk_part = line.split(":", 1)[1].strip().upper()
                    for level in ["HIGH", "MEDIUM", "LOW", "NONE"]:
                        if level in risk_part:
                            parsed["sycophancy_risk"] = level.lower()
                            break
                elif line_upper.startswith("REASONING:"):
                    parsed["reasoning"] = line.split(":", 1)[1].strip()

            any_detected = (
                parsed["opinion_steering"]
                or parsed["authority_pressure"]
                or parsed["identity_framing"]
                or parsed["emotional_manipulation"]
                or parsed["explicit_manipulation"]
            )
            parsed["llm_detection"] = any_detected

            return parsed
        except Exception:
            return {"llm_detection": False, "llm_available": True, "error": True}

    def detect_bias(self, text: str, use_llm: bool = False) -> dict:
        explicit = self.detect_explicit_bias(text)
        implicit = self.detect_implicit_bias(text)

        has_any_bias = explicit["has_bias"] or implicit["has_implicit_bias"]

        if has_any_bias and implicit["risk_level"] in ("high", "medium"):
            combined_risk = "high" if explicit["has_bias"] else implicit["risk_level"]
        elif explicit["has_bias"]:
            combined_risk = "high"
        else:
            combined_risk = implicit["risk_level"]

        result = {
            "has_bias": has_any_bias,
            "keywords_found": explicit["keywords_found"],
            "positions": explicit["positions"],
            "bias_density": explicit["bias_density"],
            "explicit_bias": explicit,
            "implicit_bias": implicit,
            "sycophancy_risk": combined_risk,
        }

        if use_llm and self.llm_pipeline is not None:
            llm_result = self.detect_bias_llm(text)
            result["llm_analysis"] = llm_result
            if llm_result.get("llm_detection") and not has_any_bias:
                result["has_bias"] = True
                result["sycophancy_risk"] = llm_result.get("sycophancy_risk", "medium")

        return result

    def remove_bias_regex(self, text: str) -> str:
        cleaned = text
        for pattern in self.explicit_patterns:
            cleaned = pattern.sub("", cleaned)

        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"\s*,\s*,", ",", cleaned)
        cleaned = cleaned.strip()
        cleaned = cleaned.strip(",").strip()

        return cleaned

    def remove_bias_llm(self, text: str) -> str:
        chain = self._get_neutralization_chain()
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
        bias_info = self.detect_bias(text, use_llm=use_llm)

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