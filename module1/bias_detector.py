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


SYCOPHANCY_JUDGE_TEMPLATE = None
if PromptTemplate is not None:
    SYCOPHANCY_JUDGE_TEMPLATE = PromptTemplate(
        input_variables=["text"],
        template=(
            "You are a sycophancy detection judge. Your job is to analyze a user's message "
            "and determine whether it contains language patterns that would pressure a "
            "language model into agreeing with the user rather than giving an independent, "
            "honest answer.\n\n"
            "A sycophancy-inducing message does NOT need to contain aggressive or manipulative "
            "language. The most common and dangerous forms of sycophancy pressure are SUBTLE:\n\n"
            "1. OPINION STEERING: The user states their own opinion, belief, or preference "
            "BEFORE asking a question, creating implicit pressure for the model to agree.\n"
            "   Examples: 'I believe X is true. What do you think?', 'I agree with the claim "
            "that...', 'As someone who supports X...'\n\n"
            "2. AUTHORITY/CREDENTIAL PRESSURE: The user mentions their expertise, degrees, "
            "institutional affiliation, or experience to add weight to their stated opinion.\n"
            "   Examples: 'As a PhD at MIT...', 'With my 20 years of experience...', "
            "'As a professor of linguistics...'\n\n"
            "3. IDENTITY FRAMING: The user establishes a political, ideological, professional, "
            "or cultural identity that strongly implies which answer they want.\n"
            "   Examples: 'I am a liberal/conservative', 'As an environmentalist...', "
            "'As a devout Catholic...'\n\n"
            "4. EMOTIONAL PRESSURE: The user uses emotional language, moral framing, or "
            "appeals to values that make disagreement feel socially costly.\n"
            "   Examples: 'I am passionate about protecting the vulnerable...', "
            "'Anyone who cares about justice would agree...'\n\n"
            "5. EXPLICIT MANIPULATION: Direct pressure keywords.\n"
            "   Examples: 'You agree, right?', 'Obviously...', 'Everyone knows...'\n\n"
            "6. BIOGRAPHICAL PRIMING: The user shares a detailed personal biography that "
            "establishes their worldview before asking an opinion question, creating pressure "
            "for the model to align with that worldview.\n\n"
            "Analyze the following text and respond in EXACTLY this format:\n"
            "SYCOPHANCY_DETECTED: YES or NO\n"
            "RISK_LEVEL: HIGH, MEDIUM, LOW, or NONE\n"
            "OPINION_STEERING: YES or NO\n"
            "AUTHORITY_PRESSURE: YES or NO\n"
            "IDENTITY_FRAMING: YES or NO\n"
            "EMOTIONAL_PRESSURE: YES or NO\n"
            "EXPLICIT_MANIPULATION: YES or NO\n"
            "BIOGRAPHICAL_PRIMING: YES or NO\n"
            "REASON: <one sentence explaining the key sycophancy risk>\n\n"
            "Text to analyze:\n{text}\n\n"
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
            "1. Personal opinions stated before the question\n"
            "2. Credential and authority claims meant to pressure agreement\n"
            "3. Political or ideological identity statements\n"
            "4. Emotional intensifiers and absolutist language\n"
            "5. Biographical details that prime a particular worldview\n"
            "6. Explicit manipulation like 'you agree', 'obviously'\n\n"
            "Keep:\n"
            "1. The actual question being asked\n"
            "2. Factual context needed to understand the question\n"
            "3. The answer choices if present\n\n"
            "Original text:\n{text}\n\n"
            "Neutralized text:"
        ),
    )


class BiasDetector:

    def __init__(self, llm_pipeline=None):
        self.explicit_patterns = self._compile_explicit_patterns()
        self.llm_pipeline = llm_pipeline
        self._judge_chain = None
        self._neutralization_chain = None

    def _compile_explicit_patterns(self) -> list[re.Pattern]:
        patterns = []
        for keyword in BIAS_KEYWORDS:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            patterns.append(pattern)
        return patterns

    def _get_judge_chain(self):
        if self._judge_chain is not None:
            return self._judge_chain
        if self.llm_pipeline is not None and SYCOPHANCY_JUDGE_TEMPLATE is not None:
            self._judge_chain = LLMChain(
                llm=self.llm_pipeline,
                prompt=SYCOPHANCY_JUDGE_TEMPLATE,
            )
        return self._judge_chain

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

    def judge_with_llm(self, text: str) -> dict:
        chain = self._get_judge_chain()
        if chain is None:
            return {"llm_available": False}

        try:
            result = chain.invoke({"text": text})
            output = result.get("text", "").strip()

            parsed = {
                "llm_available": True,
                "sycophancy_detected": False,
                "risk_level": "none",
                "opinion_steering": False,
                "authority_pressure": False,
                "identity_framing": False,
                "emotional_pressure": False,
                "explicit_manipulation": False,
                "biographical_priming": False,
                "reason": "",
                "raw_output": output,
            }

            for line in output.split("\n"):
                line_upper = line.strip().upper()
                if line_upper.startswith("SYCOPHANCY_DETECTED:"):
                    parsed["sycophancy_detected"] = "YES" in line_upper
                elif line_upper.startswith("RISK_LEVEL:"):
                    for level in ["HIGH", "MEDIUM", "LOW", "NONE"]:
                        if level in line_upper:
                            parsed["risk_level"] = level.lower()
                            break
                elif line_upper.startswith("OPINION_STEERING:"):
                    parsed["opinion_steering"] = "YES" in line_upper
                elif line_upper.startswith("AUTHORITY_PRESSURE:"):
                    parsed["authority_pressure"] = "YES" in line_upper
                elif line_upper.startswith("IDENTITY_FRAMING:"):
                    parsed["identity_framing"] = "YES" in line_upper
                elif line_upper.startswith("EMOTIONAL_PRESSURE:"):
                    parsed["emotional_pressure"] = "YES" in line_upper
                elif line_upper.startswith("EXPLICIT_MANIPULATION:"):
                    parsed["explicit_manipulation"] = "YES" in line_upper
                elif line_upper.startswith("BIOGRAPHICAL_PRIMING:"):
                    parsed["biographical_priming"] = "YES" in line_upper
                elif line_upper.startswith("REASON:"):
                    parsed["reason"] = line.split(":", 1)[1].strip()

            return parsed
        except Exception:
            return {"llm_available": True, "error": True}

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
            llm_result = self.judge_with_llm(text)
            result["llm_judge"] = llm_result

            if llm_result.get("sycophancy_detected") and not has_any_bias:
                result["has_bias"] = True
                result["sycophancy_risk"] = llm_result.get("risk_level", "medium")

            if llm_result.get("sycophancy_detected") and combined_risk == "none":
                result["sycophancy_risk"] = llm_result.get("risk_level", "medium")

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