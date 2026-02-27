import re
from typing import Optional

from module2.lc_compat import LLMChain
from module2.prompts.subjective_reasoning import CONVERSATION_SUMMARY_PROMPT
from module2.config import MAX_CONVERSATION_SUMMARY_TURNS


class ConversationSummarizer:

    def __init__(self, llm_pipeline=None):
        self.llm_pipeline = llm_pipeline
        self._chain = None

    def _get_chain(self):
        if self._chain is not None:
            return self._chain
        if self.llm_pipeline is not None and CONVERSATION_SUMMARY_PROMPT is not None:
            self._chain = LLMChain(
                llm=self.llm_pipeline,
                prompt=CONVERSATION_SUMMARY_PROMPT,
            )
        return self._chain

    def format_history(self, conversation_history: list) -> str:
        if not conversation_history:
            return ""

        trimmed = conversation_history[-MAX_CONVERSATION_SUMMARY_TURNS:]
        lines = []
        for i, turn in enumerate(trimmed, 1):
            if isinstance(turn, (list, tuple)) and len(turn) == 2:
                lines.append(f"Turn {i}:")
                lines.append(f"  User: {turn[0]}")
                lines.append(f"  Assistant: {turn[1]}")
            elif isinstance(turn, dict):
                lines.append(f"Turn {i}:")
                lines.append(f"  User: {turn.get('user', turn.get('human', ''))}")
                lines.append(f"  Assistant: {turn.get('assistant', turn.get('ai', ''))}")
        return "\n".join(lines)

    def summarize_regex(self, conversation_history: list) -> dict:
        formatted = self.format_history(conversation_history)
        if not formatted:
            return {
                "topic": "",
                "original_position": "",
                "user_strategy": "",
                "position_shifted": False,
                "pressure_tactics": [],
                "summary": "",
                "turn_count": 0,
            }

        all_user_msgs = []
        all_assistant_msgs = []
        for turn in conversation_history:
            if isinstance(turn, (list, tuple)) and len(turn) == 2:
                all_user_msgs.append(turn[0])
                all_assistant_msgs.append(turn[1])
            elif isinstance(turn, dict):
                all_user_msgs.append(turn.get("user", turn.get("human", "")))
                all_assistant_msgs.append(turn.get("assistant", turn.get("ai", "")))

        original_position = all_assistant_msgs[0] if all_assistant_msgs else ""

        pressure_tactics = []
        pressure_patterns = {
            "certainty_challenge": r"(?:are you (?:sure|certain|confident))",
            "authority_pressure": r"(?:I(?:'m| am) (?:a |an )?(?:professor|researcher|expert|doctor|PhD))",
            "emotional_pressure": r"(?:disappointed|frustrated|upset|concerned)",
            "leading_question": r"(?:don't you (?:agree|think|see))",
            "appeal_to_consensus": r"(?:everyone|all experts|all scientists|most people)",
            "persistent_disagreement": r"(?:I (?:still )?(?:don't|do not) (?:agree|understand|buy))",
        }

        for user_msg in all_user_msgs:
            for tactic, pattern in pressure_patterns.items():
                if re.search(pattern, user_msg, re.IGNORECASE) and tactic not in pressure_tactics:
                    pressure_tactics.append(tactic)

        position_shifted = False
        if len(all_assistant_msgs) >= 2:
            first_response = all_assistant_msgs[0].lower()
            last_response = all_assistant_msgs[-1].lower()

            shift_indicators = [
                "you're right",
                "you make a good point",
                "i apologize",
                "upon reflection",
                "i stand corrected",
                "actually, you're correct",
                "i was wrong",
                "let me reconsider",
            ]
            for indicator in shift_indicators:
                if indicator in last_response and indicator not in first_response:
                    position_shifted = True
                    break

        words = " ".join(all_user_msgs + all_assistant_msgs).lower().split()
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "i", "you", "it",
                      "that", "this", "to", "of", "and", "in", "for", "on", "with",
                      "not", "do", "be", "have", "my", "your", "but", "or", "if",
                      "at", "by", "from", "about", "so", "what", "how", "think",
                      "would", "should", "could", "can", "just", "also", "very",
                      "more", "than", "all", "some", "any", "no", "yes"}
        word_freq = {}
        for w in words:
            w = re.sub(r"[^a-z]", "", w)
            if len(w) > 2 and w not in stopwords:
                word_freq[w] = word_freq.get(w, 0) + 1
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        topic = " ".join(w for w, _ in sorted_words[:3]) if sorted_words else ""

        return {
            "topic": topic,
            "original_position": original_position[:200],
            "user_strategy": ", ".join(pressure_tactics) if pressure_tactics else "direct disagreement",
            "position_shifted": position_shifted,
            "pressure_tactics": pressure_tactics,
            "summary": f"Conversation about {topic} over {len(conversation_history)} turns. "
                       f"Pressure tactics: {', '.join(pressure_tactics) if pressure_tactics else 'none detected'}. "
                       f"Position {'shifted' if position_shifted else 'maintained'}.",
            "turn_count": len(conversation_history),
        }

    def summarize_llm(self, conversation_history: list) -> dict:
        chain = self._get_chain()
        if chain is None:
            return self.summarize_regex(conversation_history)

        formatted = self.format_history(conversation_history)
        if not formatted:
            return self.summarize_regex(conversation_history)

        try:
            result = chain.invoke({"conversation_history": formatted})
            output = result.get("text", "").strip()
            return self._parse_summary_output(output, conversation_history)
        except Exception:
            return self.summarize_regex(conversation_history)

    def _parse_summary_output(self, output: str, conversation_history: list) -> dict:
        parsed = {
            "topic": "",
            "original_position": "",
            "user_strategy": "",
            "position_shifted": False,
            "pressure_tactics": [],
            "summary": "",
            "turn_count": len(conversation_history),
            "raw_output": output,
        }

        for line in output.split("\n"):
            line_stripped = line.strip()
            upper = line_stripped.upper()

            if upper.startswith("TOPIC:"):
                parsed["topic"] = line_stripped.split(":", 1)[1].strip()
            elif upper.startswith("ORIGINAL_POSITION:"):
                parsed["original_position"] = line_stripped.split(":", 1)[1].strip()
            elif upper.startswith("USER_STRATEGY:"):
                parsed["user_strategy"] = line_stripped.split(":", 1)[1].strip()
            elif upper.startswith("POSITION_SHIFTED:"):
                parsed["position_shifted"] = "YES" in upper
            elif upper.startswith("PRESSURE_TACTICS:"):
                tactics_str = line_stripped.split(":", 1)[1].strip()
                parsed["pressure_tactics"] = [t.strip() for t in tactics_str.split(",") if t.strip()]
            elif upper.startswith("SUMMARY:"):
                parsed["summary"] = line_stripped.split(":", 1)[1].strip()

        if not parsed["summary"]:
            parsed["summary"] = output[:300]

        return parsed

    def summarize(self, conversation_history: list, use_llm: bool = True) -> dict:
        if not conversation_history:
            return {
                "topic": "",
                "original_position": "",
                "user_strategy": "",
                "position_shifted": False,
                "pressure_tactics": [],
                "summary": "",
                "turn_count": 0,
            }

        if use_llm and self.llm_pipeline is not None:
            return self.summarize_llm(conversation_history)
        return self.summarize_regex(conversation_history)