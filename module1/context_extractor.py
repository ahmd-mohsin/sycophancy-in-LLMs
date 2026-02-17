import re
from typing import Optional

from .lc_compat import PromptTemplate, LLMChain, HuggingFacePipeline
from .conversation_history import ConversationHistoryManager
from .bias_detector import BiasDetector


CONTEXT_SUMMARIZATION_TEMPLATE = None
if PromptTemplate is not None:
    CONTEXT_SUMMARIZATION_TEMPLATE = PromptTemplate(
        input_variables=["conversation_history"],
        template=(
            "You are a context summarization agent. Analyze the conversation history "
            "below and extract ONLY the following categories of information.\n\n"
            "Extract:\n"
            "1. USER_PROFILE: Any self-descriptions by the user (experience level, "
            "domain expertise, preferences, background)\n"
            "2. TOPICS: The main discussion topics covered\n"
            "3. FACTS: Factual statements made during the conversation that are "
            "relevant context\n\n"
            "DO NOT extract:\n"
            "- Agreements or disagreements between user and assistant\n"
            "- Evaluative judgments (e.g., 'X is better than Y')\n"
            "- Social dynamics or pleasantries\n"
            "- Any biased or leading statements\n\n"
            "Conversation:\n{conversation_history}\n\n"
            "Output in this exact format:\n"
            "USER_PROFILE: [comma-separated items]\n"
            "TOPICS: [comma-separated items]\n"
            "FACTS: [comma-separated items]\n\n"
            "Extraction:"
        ),
    )

USER_DESCRIPTION_PATTERNS = [
    r"i(?:'m| am) (?:a |an )?([\w\s]+?)(?:\.|,|$)",
    r"i have (\d+\s*(?:years?|months?)\s*(?:of\s*)?experience[\w\s]*?)(?:\.|,|$)",
    r"i(?:'ve| have) (?:been |worked )(?:as |in )?([\w\s]+?)(?:\.|,|$)",
    r"my background is in ([\w\s]+?)(?:\.|,|$)",
    r"i(?:'m| am) new to ([\w\s]+?)(?:\.|,|$)",
    r"i specialize in ([\w\s]+?)(?:\.|,|$)",
    r"i prefer ([\w\s]+?)(?:\.|,|$)",
]


class ContextExtractor:

    def __init__(
        self,
        llm_pipeline=None,
        history_manager: Optional[ConversationHistoryManager] = None,
        bias_detector: Optional[BiasDetector] = None,
    ):
        self.llm_pipeline = llm_pipeline
        self.history_manager = history_manager or ConversationHistoryManager()
        self.bias_detector = bias_detector or BiasDetector(llm_pipeline)
        self._chain = None

    def _get_chain(self):
        if self._chain is not None:
            return self._chain
        if self.llm_pipeline is not None and LLMChain is not None and CONTEXT_SUMMARIZATION_TEMPLATE is not None:
            self._chain = LLMChain(
                llm=self.llm_pipeline,
                prompt=CONTEXT_SUMMARIZATION_TEMPLATE,
            )
        return self._chain

    def _extract_user_profile_regex(self, turns: list[tuple[str, str]]) -> list[str]:
        profiles = []
        for user_msg, _ in turns:
            user_lower = user_msg.lower()
            for pattern in USER_DESCRIPTION_PATTERNS:
                matches = re.findall(pattern, user_lower)
                for match in matches:
                    cleaned = match.strip()
                    if cleaned and len(cleaned) > 2:
                        profiles.append(cleaned)
        return list(set(profiles))

    def _extract_topics_regex(self, turns: list[tuple[str, str]]) -> list[str]:
        all_text = " ".join(
            f"{user_msg} {asst_msg}" for user_msg, asst_msg in turns
        )
        words = re.findall(r"\b[a-zA-Z]{4,}\b", all_text.lower())
        stopwords = {
            "that", "this", "with", "from", "have", "been", "will",
            "would", "could", "should", "about", "which", "their",
            "there", "these", "those", "then", "than", "when", "what",
            "just", "also", "some", "very", "more", "much", "your",
            "like", "think", "know", "want", "need", "make", "good",
            "well", "here", "does", "into", "over", "only",
        }
        filtered = [w for w in words if w not in stopwords]

        freq = {}
        for word in filtered:
            freq[word] = freq.get(word, 0) + 1

        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        topics = [word for word, count in sorted_words[:10] if count >= 2]

        return topics if topics else [w for w, _ in sorted_words[:5]]

    def _extract_facts_regex(self, turns: list[tuple[str, str]]) -> list[str]:
        facts = []
        evaluative_patterns = [
            r"\bbest\b", r"\bworst\b", r"\bbetter\b", r"\bworse\b",
            r"\bshould\b", r"\bopinion\b", r"\bi think\b", r"\bi feel\b",
            r"\bi believe\b", r"\bprobably\b", r"\bmaybe\b",
        ]

        for _, asst_msg in turns:
            sentences = re.split(r"(?<=[.!?])\s+", asst_msg)
            for sentence in sentences:
                sentence_lower = sentence.lower()
                is_evaluative = any(
                    re.search(p, sentence_lower) for p in evaluative_patterns
                )
                if is_evaluative:
                    continue

                has_factual_verb = bool(
                    re.search(r"\b(is|are|was|were|has|have|contains?|consists?)\b", sentence_lower)
                )
                if has_factual_verb and len(sentence.split()) >= 4:
                    bias_check = self.bias_detector.detect_bias(sentence)
                    if not bias_check["has_bias"]:
                        facts.append(sentence.strip())

        return facts[:10]

    def _parse_llm_output(self, text: str) -> dict:
        result = {"user_profile": [], "topics": [], "facts": []}

        for line in text.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("USER_PROFILE:"):
                items = line.split(":", 1)[1].strip()
                result["user_profile"] = [
                    i.strip() for i in items.split(",") if i.strip()
                ]
            elif line.upper().startswith("TOPICS:"):
                items = line.split(":", 1)[1].strip()
                result["topics"] = [
                    i.strip() for i in items.split(",") if i.strip()
                ]
            elif line.upper().startswith("FACTS:"):
                items = line.split(":", 1)[1].strip()
                result["facts"] = [
                    i.strip() for i in items.split(",") if i.strip()
                ]

        return result

    def extract_context_llm(self, turns: list[tuple[str, str]]) -> dict:
        chain = self._get_chain()
        if chain is None:
            return self.extract_context_regex(turns)

        history_text = ""
        for i, (user_msg, asst_msg) in enumerate(turns, 1):
            history_text += f"Turn {i}:\n  User: {user_msg}\n  Assistant: {asst_msg}\n"

        try:
            result = chain.invoke({"conversation_history": history_text})
            parsed = self._parse_llm_output(result.get("text", ""))

            if not any(parsed.values()):
                return self.extract_context_regex(turns)

            filtered_facts = []
            for fact in parsed["facts"]:
                bias_check = self.bias_detector.detect_bias(fact)
                if not bias_check["has_bias"]:
                    filtered_facts.append(fact)
            parsed["facts"] = filtered_facts

            return parsed
        except Exception:
            return self.extract_context_regex(turns)

    def extract_context_regex(self, turns: list[tuple[str, str]]) -> dict:
        return {
            "user_profile": self._extract_user_profile_regex(turns),
            "topics": self._extract_topics_regex(turns),
            "facts": self._extract_facts_regex(turns),
        }

    def extract_context(
        self,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        session_id: Optional[str] = None,
        use_llm: bool = True,
    ) -> dict:
        if conversation_history is None and session_id is not None:
            conversation_history = self.history_manager.get_turns(session_id)

        if not conversation_history:
            return {"user_profile": [], "topics": [], "facts": []}

        if use_llm and self.llm_pipeline is not None:
            return self.extract_context_llm(conversation_history)
        return self.extract_context_regex(conversation_history)