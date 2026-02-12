from typing import Optional

from langchain_huggingface import HuggingFacePipeline

from module1.config import HF_MODEL_NAME, HF_MAX_NEW_TOKENS, HF_TEMPERATURE
from module1.conversation_history import ConversationHistoryManager
from module1.bias_detector import BiasDetector
from module1.claim_extractor import ClaimExtractor
from module1.question_classifier import QuestionClassifier
from module1.context_extractor import ContextExtractor


class Module1:

    def __init__(
        self,
        llm_pipeline: Optional[HuggingFacePipeline] = None,
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

    def _create_pipeline(self) -> Optional[HuggingFacePipeline]:
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
            context_summary = self.context_extractor.extract_context(
                conversation_history=conversation_history,
                session_id=session_id,
                use_llm=self.use_llm,
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


def create_module1(use_llm: bool = True, history_path: Optional[str] = None) -> Module1:
    history_manager = ConversationHistoryManager(history_path) if history_path else None
    return Module1(use_llm=use_llm, history_manager=history_manager)