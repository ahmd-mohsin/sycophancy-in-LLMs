from typing import Optional

from .lc_compat import HuggingFacePipeline
from .config import HF_MODEL_NAME, HF_MAX_NEW_TOKENS, HF_TEMPERATURE
from .conversation_history import ConversationHistoryManager
from .bias_detector import BiasDetector
from .claim_extractor import ClaimExtractor
from .question_classifier import QuestionClassifier
from .context_extractor import ContextExtractor


class Module1:

    def __init__(
        self,
        llm_pipeline=None,
        history_manager: Optional[ConversationHistoryManager] = None,
        use_llm: bool = True,
    ):
        self.use_llm = use_llm
        self.history_manager = history_manager or ConversationHistoryManager()

        self.llm_pipeline = llm_pipeline
        if self.use_llm and self.llm_pipeline is None:
            self.llm_pipeline = self._create_pipeline()

        self.bias_detector = BiasDetector(self.llm_pipeline)
        self.claim_extractor = ClaimExtractor(self.llm_pipeline, self.bias_detector)
        self.question_classifier = QuestionClassifier(self.llm_pipeline)
        self.context_extractor = ContextExtractor(
            self.llm_pipeline, self.history_manager, self.bias_detector
        )

    def _create_pipeline(self):
        if HuggingFacePipeline is None:
            return None
        try:
            from transformers import pipeline as hf_pipeline

            pipe = hf_pipeline(
                "text-generation",
                model=HF_MODEL_NAME,
                max_new_tokens=HF_MAX_NEW_TOKENS,
                temperature=HF_TEMPERATURE,
                do_sample=True,
                return_full_text=False,
            )
            return HuggingFacePipeline(pipeline=pipe)
        except Exception:
            return None

    def process(
        self,
        question: str,
        assistant_answer: str,
        user_challenge: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Run the full Module 1 pipeline on a single turn.

        Context extraction priority (for subjective questions):
          1. conversation_history list of tuples — explicit multi-turn history
          2. session_id — looks up history from ConversationHistoryManager
          3. user_challenge string — single-question fallback (new path)
             used when calling from eval pipeline with no history available

        Args:
            question:             The core question being asked (bias-stripped).
            assistant_answer:     The assistant's current answer (claim_A source).
            user_challenge:       The user's raw challenge/pushback (claim_B source).
                                  Also used as fallback context source when no
                                  conversation history is available.
            conversation_history: Optional explicit multi-turn history as
                                  list of (user_msg, assistant_msg) tuples.
            session_id:           Optional session ID for history lookup.

        Returns:
            dict with keys: question_type, claim_A, claim_B,
                            context_summary, classification_details,
                            claim_details
        """
        claims = self.claim_extractor.extract_claims(
            assistant_answer=assistant_answer,
            user_challenge=user_challenge,
            question=question,
            use_llm=self.use_llm,
        )

        classification = self.question_classifier.classify(
            question=question,
            user_challenge=user_challenge,
            use_llm=self.use_llm,
        )

        context_summary = {}
        if classification["question_type"] == "subjective":
            context_summary = self._extract_context(
                question=question,
                user_challenge=user_challenge,
                conversation_history=conversation_history,
                session_id=session_id,
            )

        if session_id:
            self.history_manager.add_turn(session_id, question, assistant_answer)

        return {
            "question_type": classification["question_type"],
            "claim_A": claims["claim_A"],
            "claim_B": claims["claim_B"],
            "context_summary": context_summary,
            "classification_details": classification,
            "claim_details": claims,
        }

    def _extract_context(
        self,
        question: str,
        user_challenge: str,
        conversation_history: Optional[list[tuple[str, str]]],
        session_id: Optional[str],
    ) -> dict:
        """
        Internal routing for context extraction. Tries each source in
        priority order, returning the first non-empty result.

        Priority:
          1. Explicit conversation_history (multi-turn list of tuples)
          2. session_id  → history lookup via ConversationHistoryManager
          3. user_challenge string (single-question fallback)
          4. question string (last resort if user_challenge is empty)
        """
        # --- Priority 1: explicit multi-turn history ---
        if conversation_history:
            return self.context_extractor.extract_context(
                conversation_history=conversation_history,
                use_llm=self.use_llm,
            )

        # --- Priority 2: session-based history lookup ---
        if session_id:
            history = self.history_manager.get_turns(session_id)
            if history:
                return self.context_extractor.extract_context(
                    conversation_history=history,
                    use_llm=self.use_llm,
                )

        # --- Priority 3: single question string (eval pipeline path) ---
        # Prefer user_challenge because it contains the full biased prompt
        # (with authority claims, biographical priming, etc.) which is
        # richer context than the stripped clean question.
        context_source = user_challenge if user_challenge and user_challenge.strip() \
            else question

        return self.context_extractor.extract_context_from_question(
            question=context_source,
            use_llm=self.use_llm,
        )


def create_module1(use_llm: bool = True, history_path: Optional[str] = None) -> Module1:
    history_manager = ConversationHistoryManager(history_path) if history_path else None
    return Module1(use_llm=use_llm, history_manager=history_manager)