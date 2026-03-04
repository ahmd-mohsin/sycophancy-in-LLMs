"""
generators/base_generator.py  — v2

Key changes from v1:
  1. Baseline generation uses category-specific baseline_system_prompt (committed stance).
  2. Baseline is generated from context (orig) and opposite_context separately,
     ensuring baseline_orig and baseline_opp take genuinely opposing positions.
  3. NLI validation: groups where contra(baseline_orig, baseline_opp) < NLI_THRESHOLD
     are REJECTED and regenerated (up to MAX_SEED_RETRIES). This catches the
     "both baselines converge to same centrist answer" failure mode at generation time.
  4. opposite_opinion now stored in group metadata for proper Rpos reward computation.
  5. Scoring is printed per group so you can monitor NLI quality live.
"""

import abc
import json
import logging
import uuid
from typing import List, Dict, Any, Optional

from config import MODEL_CONFIG, OUTPUT_DIR, PRESSURE_LEVELS, OLLAMA_BASE_URL
from pressure.templates import PRESSURE_TEMPLATES
from utils.model_client import ModelClient
from utils.io_utils import (
    build_sample,
    save_dataset,
    save_checkpoint,
    load_checkpoint,
    ensure_dir,
)

logger = logging.getLogger(__name__)

# NLI validation threshold: baseline_orig vs baseline_opp must contradict
# each other by at least this probability. Groups below are regenerated.
NLI_THRESHOLD = 0.40
MAX_SEED_RETRIES = 5


class BaseGenerator(abc.ABC):
    category: str = "base"

    def __init__(self, resume: bool = True, validate_nli: bool = True):
        cfg      = MODEL_CONFIG[self.category]
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

        self.validate_nli = validate_nli
        self._nli_fn = None  # lazy-loaded

        self.output_dir = OUTPUT_DIR
        ensure_dir(self.output_dir)

        self.checkpoint_path = f"{self.output_dir}/{self.category}_checkpoint.json"
        self.existing_samples: List[Dict] = []
        if resume:
            self.existing_samples = load_checkpoint(self.checkpoint_path)
            logger.info(
                f"[{self.category}] Resuming from {len(self.existing_samples)} existing samples."
            )

    # ─── Abstract interface ────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def seed_system_prompt(self) -> str:
        """System prompt for orchestrator to generate seed JSON."""
        ...

    @property
    @abc.abstractmethod
    def response_system_prompt(self) -> str:
        """System prompt for category model when answering pressured questions."""
        ...

    @property
    @abc.abstractmethod
    def baseline_system_prompt(self) -> str:
        """
        System prompt for generating baselines.
        Must force a committed, position-first response (no hedging).
        This is separate from response_system_prompt because baselines
        have no pressure prompt — they just see context + question.
        """
        ...

    @abc.abstractmethod
    def parse_seed(self, raw: str) -> Optional[Dict[str, str]]:
        """
        Parse orchestrator output. Must return a dict with keys:
          topic, question, context, opposite_context, opinion, opposite_opinion
        Return None on parse failure.
        """
        ...

    # ─── NLI validation ────────────────────────────────────────────────────

    def _get_nli(self):
        if self._nli_fn is None:
            from metrics.nli import contradiction_score
            self._nli_fn = contradiction_score
        return self._nli_fn

    def _nli_valid(self, baseline_orig: str, baseline_opp: str) -> tuple[bool, float]:
        """
        Returns (is_valid, contra_score).
        A group is valid if baseline_orig and baseline_opp clearly contradict each other.
        """
        if not self.validate_nli:
            return True, 1.0
        contra = self._get_nli()(baseline_orig, baseline_opp)
        return contra >= NLI_THRESHOLD, contra

    # ─── Prompt assembly ──────────────────────────────────────────────────

    def _assemble_prompt(self, pressure_prompt: str, context: str, question: str) -> str:
        return (
            f"{pressure_prompt}\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}"
        )

    def _baseline_prompt(self, context: str, question: str) -> str:
        """Baseline has no pressure prompt — just context and question."""
        return f"Context: {context}\n\nQuestion: {question}"

    # ─── Generation helpers ───────────────────────────────────────────────

    def _generate_seed(self) -> Optional[Dict[str, str]]:
        raw = self.orchestrator.generate(
            prompt="Generate one example now. Return only valid JSON, nothing else.",
            system_prompt=self.seed_system_prompt,
        )
        return self.parse_seed(raw)

    def _get_response(self, full_prompt: str) -> str:
        return self.model.generate(
            prompt=full_prompt,
            system_prompt=self.response_system_prompt,
        )

    def _get_baseline(self, context: str, question: str) -> str:
        prompt = self._baseline_prompt(context, question)
        return self.model.generate(
            prompt=prompt,
            system_prompt=self.baseline_system_prompt,
        )

    # ─── Core: build one group ────────────────────────────────────────────

    def _build_group_for_seed(
        self, seed: Dict[str, str], seed_index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Build baselines and 6 pressured samples for one seed.
        Returns None if NLI validation fails (caller will retry with new seed).
        """
        topic         = seed["topic"]
        original_ctx  = seed["context"]
        opposite_ctx  = seed["opposite_context"]
        question      = seed["question"]
        opinion       = seed["opinion"]
        opp_opinion   = seed.get("opposite_opinion", "")

        # Generate baselines using committed baseline_system_prompt
        try:
            baseline_orig = self._get_baseline(original_ctx, question)
            baseline_opp  = self._get_baseline(opposite_ctx, question)
        except Exception as e:
            logger.warning(f"  Baseline generation failed: {e}")
            return None

        # NLI gate: reject groups where baselines don't clearly contradict
        valid, contra_score = self._nli_valid(baseline_orig, baseline_opp)
        logger.info(
            f"  [{self.category}] seed={seed_index} NLI contra(orig,opp)={contra_score:.3f} "
            f"{'✓ PASS' if valid else '✗ FAIL — will retry'}"
        )
        if not valid:
            return None

        # Build pressured samples
        samples = []
        for level in PRESSURE_LEVELS:
            templates     = PRESSURE_TEMPLATES[level]
            pressure_prompt = templates[seed_index % len(templates)].format(opinion=opinion)

            for ctx_type, context in [("original", original_ctx), ("opposite", opposite_ctx)]:
                full_prompt = self._assemble_prompt(pressure_prompt, context, question)

                try:
                    response = self._get_response(full_prompt)
                except Exception as e:
                    logger.warning(f"  Response failed for seed={seed_index} level={level} ctx={ctx_type}: {e}")
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
                logger.debug(f"    [{self.category}] seed={seed_index} level={level} ctx={ctx_type} ✓")

        return {
            "topic":           topic,
            "question":        question,
            "opinion":         opinion,
            "opposite_opinion": opp_opinion,
            "baseline_orig":   baseline_orig,
            "baseline_opp":    baseline_opp,
            "nli_contra":      round(contra_score, 4),
            "samples":         samples,
        }

    # ─── Public API ───────────────────────────────────────────────────────

    def generate(self, n_topics: int = 100) -> List[Dict[str, Any]]:
        """
        Generate n_topics valid groups (each with 6 pressured samples + 2 baselines).
        Groups that fail NLI validation are regenerated.
        Checkpoints after every valid group.
        """
        all_groups = list(self.existing_samples)
        completed  = len(all_groups)
        logger.info(
            f"[{self.category}] Need {n_topics} groups. "
            f"{completed} already done. Generating {n_topics - completed} more."
        )

        generated = completed
        attempt_global = 0

        while generated < n_topics:
            attempt_global += 1
            if attempt_global > n_topics * MAX_SEED_RETRIES * 2:
                logger.error(
                    f"[{self.category}] Exceeded maximum generation attempts. "
                    f"Generated {generated}/{n_topics} groups."
                )
                break

            logger.info(f"[{self.category}] Group {generated+1}/{n_topics} — attempt {attempt_global}")

            # Retry seed parse
            seed = None
            for parse_attempt in range(5):
                seed = self._generate_seed()
                if seed:
                    break
                logger.warning(f"  Seed parse failed (attempt {parse_attempt+1}/5)")

            if not seed:
                logger.error(f"  Skipping — 5 consecutive parse failures.")
                continue

            group = self._build_group_for_seed(seed, seed_index=generated)
            if group is None:
                # NLI validation failed — try a fresh seed
                continue

            all_groups.append(group)
            generated += 1

            save_checkpoint(all_groups, self.category, self.output_dir)
            logger.info(f"  [{self.category}] Checkpoint saved ({generated} groups).")

        filepath = save_dataset(
            samples=all_groups,
            category=self.category,
            output_dir=self.output_dir,
            metadata={"n_topics": n_topics, "samples_per_topic": 6, "version": "v2"},
        )
        logger.info(f"[{self.category}] Dataset complete → {filepath}")
        return all_groups


def _parse_json_safe(raw: str) -> Optional[Dict]:
    """Strip markdown fences and parse JSON. Returns None on failure."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None