"""
utils/model_client.py

Unified client that abstracts over Ollama and HuggingFace backends.
Swap the backend in config.py — the rest of the code doesn't change.
"""

import json
import time
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class ModelClient:
    """
    Wraps Ollama and HuggingFace inference into a single `.generate()` call.

    Usage:
        client = ModelClient(backend="ollama", model_id="llama3.2:3b")
        response = client.generate("What is 2+2?")
    """

    def __init__(
        self,
        backend: str,
        model_id: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        ollama_base_url: str = "http://localhost:11434",
    ):
        assert backend in ("ollama", "huggingface"), \
            f"Unknown backend '{backend}'. Choose 'ollama' or 'huggingface'."

        self.backend = backend
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ollama_base_url = ollama_base_url

        if backend == "huggingface":
            self._load_hf_pipeline()

    # ─── HuggingFace ──────────────────────────────────────────────────────────

    def _load_hf_pipeline(self):
        """Lazy-load the HF pipeline so Ollama-only runs don't import torch."""
        try:
            from transformers import pipeline, AutoTokenizer
            import torch
            logger.info(f"Loading HuggingFace model: {self.model_id}")
            self._hf_pipe = pipeline(
                "text-generation",
                model=self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            logger.info("HuggingFace model loaded.")
        except ImportError:
            raise ImportError(
                "transformers and torch are required for HuggingFace backend. "
                "Run: pip install transformers torch accelerate"
            )

    def _generate_hf(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        outputs = self._hf_pipe(
            full_prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self._hf_pipe.tokenizer.eos_token_id,
        )
        generated = outputs[0]["generated_text"]
        # Strip the input prompt from output
        return generated[len(full_prompt):].strip()

    # ─── Ollama ───────────────────────────────────────────────────────────────

    # Models that crash with separate system prompts (thinking models, etc.)
    MERGE_SYSTEM_PROMPT_MODELS = ("deepseek-r1",)

    def _should_merge_system_prompt(self) -> bool:
        return any(name in self.model_id for name in self.MERGE_SYSTEM_PROMPT_MODELS)

    def _generate_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.ollama_base_url}/api/chat"

        # deepseek-r1 crashes when system prompt is passed separately —
        # merge into the user message instead
        if system_prompt and self._should_merge_system_prompt():
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{prompt}"}]
        else:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": 4096,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=300)  # Increased to 5 minutes
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.ollama_base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 404:
                raise ValueError(
                    f"Model '{self.model_id}' not found in Ollama. "
                    f"Pull it with: `ollama pull {self.model_id}`"
                )
            raise e

    # ─── Public API ───────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        """
        Generate a response. Retries on failure.

        Args:
            prompt:        User-facing prompt text.
            system_prompt: Optional system instruction.
            retries:       Number of retry attempts on error.
            retry_delay:   Seconds to wait between retries.

        Returns:
            Generated string response.
        """
        for attempt in range(1, retries + 1):
            try:
                if self.backend == "ollama":
                    return self._generate_ollama(prompt, system_prompt)
                else:
                    return self._generate_hf(prompt, system_prompt)
            except (ConnectionError, ValueError):
                raise  # Don't retry on config errors
            except Exception as e:
                logger.warning(f"Attempt {attempt}/{retries} failed: {e}")
                if attempt < retries:
                    time.sleep(retry_delay)
                else:
                    logger.error(f"All {retries} attempts failed.")
                    raise

    def ping(self) -> bool:
        """Check if the backend is reachable."""
        if self.backend == "ollama":
            try:
                resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
                return resp.status_code == 200
            except Exception:
                return False
        return True  # HF is always local