from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import Optional

from module1.config import (
    HF_MODEL_NAME,
    HF_API_TOKEN,
    HF_DEVICE,
    HF_MAX_NEW_TOKENS,
    HF_TEMPERATURE,
)
from module1.conversation_history import ConversationHistoryManager


class HFModelWrapper:

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        history_manager: Optional[ConversationHistoryManager] = None,
    ):
        self.model_name = model_name or HF_MODEL_NAME
        self.device = device or HF_DEVICE
        self.history_manager = history_manager or ConversationHistoryManager()
        self._tokenizer = None
        self._model = None
        self._pipeline = None

    def _load_model(self):
        if self._pipeline is not None:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=HF_API_TOKEN if HF_API_TOKEN else None,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=HF_API_TOKEN if HF_API_TOKEN else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device,
        )

        self._pipeline = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            max_new_tokens=HF_MAX_NEW_TOKENS,
            temperature=HF_TEMPERATURE,
            do_sample=True,
            return_full_text=False,
        )

    def generate(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        include_history: bool = False,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        self._load_model()

        full_prompt = prompt
        if include_history and session_id:
            history_text = self.history_manager.format_history_for_prompt(session_id)
            if history_text:
                full_prompt = (
                    f"Conversation history:\n{history_text}\n\n"
                    f"Current request:\n{prompt}"
                )

        generation_kwargs = {}
        if max_new_tokens is not None:
            generation_kwargs["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            generation_kwargs["temperature"] = temperature

        result = self._pipeline(full_prompt, **generation_kwargs)
        generated_text = result[0]["generated_text"].strip()

        if session_id:
            self.history_manager.add_turn(session_id, prompt, generated_text)

        return generated_text

    def generate_with_messages(
        self,
        messages: list[dict],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        self._load_model()

        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            lines = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    lines.append(f"System: {content}")
                elif role == "user":
                    lines.append(f"User: {content}")
                elif role == "assistant":
                    lines.append(f"Assistant: {content}")
            lines.append("Assistant:")
            prompt = "\n".join(lines)

        generation_kwargs = {}
        if max_new_tokens is not None:
            generation_kwargs["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            generation_kwargs["temperature"] = temperature

        result = self._pipeline(prompt, **generation_kwargs)
        return result[0]["generated_text"].strip()

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None