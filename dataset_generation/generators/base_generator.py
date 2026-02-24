"""
generators/base_generator.py

Abstract base class for all category generators.
Handles the full pipeline:
  1. Seed (C, Q, opinion) generation via orchestrator model
  2. Pressure prompt construction for all levels
  3. Opposite context generation
  4. Response collection via category-specific model
  5. Checkpointing and saving
"""

import abc
import json
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple

from config import MODEL_CONFIG, OUTPUT_DIR, PRESSURE_LEVELS, OLLAMA_BASE_URL
from pressure.templates import get_pressure_prompt, PRESSURE_TEMPLATES
from utils.model_client import ModelClient
from utils.io_utils import (
    build_sample,
    save_dataset,
    save_checkpoint,
    load_checkpoint,
    ensure_dir,
)

logger = logging.getLogger(__name__)


class BaseGenerator(abc.ABC):
    """
    Subclass this for each category. Override:
      - `category`          (str)
      - `seed_system_prompt` (str)  — instructions for generating (C, Q, opinion) seeds
      - `response_system_prompt` (str) — instructions for the answering model
      - `parse_seed`        (method) — parse orchestrator JSON into (topic, context, opposite_context, question, opinion)
    """

    category: str = "base"

    def __init__(self, resume: bool = True):
        cfg = MODEL_CONFIG[self.category]
        orch_cfg = MODEL_CONFIG["orchestrator"]

        self.model = ModelClient(
            backend=cfg["backend"],
            model_id=cfg["model_id"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            ollama_base_url=OLLAMA_BASE_URL,
        )
        self.orchestrator = ModelClient(
            backend=orch_cfg["backend"],
            model_id=orch_cfg["model_id"],
            temperature=orch_cfg["temperature"],
            max_tokens=orch_cfg["max_tokens"],
            ollama_base_url=OLLAMA_BASE_URL,
        )

        self.output_dir = OUTPUT_DIR
        ensure_dir(self.output_dir)

        # Checkpoint support
        self.checkpoint_path = f"{self.output_dir}/{self.category}_checkpoint.json"
        self.existing_samples: List[Dict] = []
        if resume:
            self.existing_samples = load_checkpoint(self.checkpoint_path)
            logger.info(
                f"[{self.category}] Resuming from {len(self.existing_samples)} existing samples."
            )

    # ─── Override these ───────────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def seed_system_prompt(self) -> str:
        """System prompt for orchestrator model to generate seed (C, Q, opinion) JSON."""
        ...

    @property
    @abc.abstractmethod
    def response_system_prompt(self) -> str:
        """System prompt for the category model when answering questions."""
        ...

    @abc.abstractmethod
    def parse_seed(self, raw: str) -> Optional[Dict[str, str]]:
        """
        Parse orchestrator output into:
        {
          "topic": str,
          "context": str,          # original context C
          "opposite_context": str, # flipped context C'
          "question": str,         # Q
          "opinion": str,          # the opinion being pressured toward
        }
        Return None on parse failure.
        """
        ...

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _assemble_prompt(
        self,
        pressure_prompt: str,
        context: str,
        question: str,
    ) -> str:
        """Combine P + C + Q into the final prompt sent to the model."""
        return (
            f"{pressure_prompt}\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}"
        )

    def _generate_seed(self) -> Optional[Dict[str, str]]:
        """Ask the orchestrator to generate one (topic, C, C', Q, opinion) tuple."""
        raw = self.orchestrator.generate(
            prompt="Generate one example now. Return only valid JSON, nothing else.",
            system_prompt=self.seed_system_prompt,
        )
        return self.parse_seed(raw)

    def _get_response(self, full_prompt: str) -> str:
        """Get the category model's response to the assembled prompt."""
        return self.model.generate(
            prompt=full_prompt,
            system_prompt=self.response_system_prompt,
        )

    def _build_samples_for_seed(
        self, seed: Dict[str, str], seed_index: int
    ) -> List[Dict[str, Any]]:
        """
        For one seed, produce 6 samples:
          3 pressure levels × 2 context types (original, opposite)
        """
        samples = []
        topic = seed["topic"]
        original_ctx = seed["context"]
        opposite_ctx = seed["opposite_context"]
        question = seed["question"]
        opinion = seed["opinion"]

        # Pick one template per pressure level (rotate by seed_index)
        for level in PRESSURE_LEVELS:
            templates = PRESSURE_TEMPLATES[level]
            pressure_prompt = templates[seed_index % len(templates)].format(opinion=opinion)

            for ctx_type, context in [("original", original_ctx), ("opposite", opposite_ctx)]:
                full_prompt = self._assemble_prompt(pressure_prompt, context, question)

                try:
                    response = self._get_response(full_prompt)
                except Exception as e:
                    logger.warning(f"Response generation failed for seed {seed_index}: {e}")
                    response = "ERROR: generation failed"

                sample = build_sample(
                    category=self.category,
                    topic=topic,
                    pressure_level=level,
                    pressure_prompt=pressure_prompt,
                    context=context,
                    context_type=ctx_type,
                    question=question,
                    opinion=opinion,
                    response=response,
                    full_prompt=full_prompt,
                    sample_id=f"{self.category}_{seed_index:04d}_{level}_{ctx_type}_{uuid.uuid4().hex[:6]}",
                )
                samples.append(sample)
                logger.debug(f"  [{self.category}] seed={seed_index} level={level} ctx={ctx_type} ✓")

        return samples

    # ─── Public API ───────────────────────────────────────────────────────────

    def generate(self, n_topics: int = 100) -> List[Dict[str, Any]]:
        """
        Generate n_topics seeds and produce 6 samples each (= 6×n_topics total).
        Checkpoints after every seed. Resumes from checkpoint if resume=True.

        Args:
            n_topics: Number of unique (C, Q) seeds to generate.

        Returns:
            List of all samples including any from a prior checkpoint run.
        """
        all_samples = list(self.existing_samples)
        # How many seeds have already been fully processed?
        completed_seeds = len(all_samples) // 6
        remaining = n_topics - completed_seeds

        logger.info(
            f"[{self.category}] Need {n_topics} seeds. "
            f"{completed_seeds} already done. Generating {remaining} more."
        )

        for i in range(completed_seeds, n_topics):
            logger.info(f"[{self.category}] Seed {i+1}/{n_topics} ...")

            # Retry seed generation up to 5 times
            seed = None
            for attempt in range(5):
                seed = self._generate_seed()
                if seed:
                    break
                logger.warning(f"  Seed parse failed (attempt {attempt+1}/5), retrying...")

            if not seed:
                logger.error(f"  Skipping seed {i+1} after 5 failed attempts.")
                continue

            new_samples = self._build_samples_for_seed(seed, seed_index=i)
            all_samples.extend(new_samples)

            # Checkpoint after every seed
            save_checkpoint(all_samples, self.category, self.output_dir)
            logger.info(f"  [{self.category}] Checkpoint saved ({len(all_samples)} samples total).")

        # Final save
        filepath = save_dataset(
            samples=all_samples,
            category=self.category,
            output_dir=self.output_dir,
            metadata={"n_topics": n_topics, "samples_per_topic": 6},
        )
        logger.info(f"[{self.category}] Dataset complete → {filepath}")
        return all_samples


def _parse_json_safe(raw: str) -> Optional[Dict]:
    """Strip markdown fences and parse JSON. Returns None on failure."""
    raw = raw.strip()
    # Strip ```json ... ``` fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting the first {...} block
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None