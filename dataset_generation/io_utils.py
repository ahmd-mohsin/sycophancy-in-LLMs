"""
utils/io_utils.py

Handles all file I/O: saving samples, loading checkpoints, progress tracking.
"""

import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_dataset(
    samples: List[Dict[str, Any]],
    category: str,
    output_dir: str,
    metadata: Optional[Dict] = None,
) -> str:
    """
    Save a list of samples to a structured JSON file.

    Output format:
    {
      "metadata": { ... },
      "samples": [ { ... }, ... ]
    }

    Returns the path to the saved file.
    """
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{category}_dataset_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    payload = {
        "metadata": {
            "category": category,
            "n_samples": len(samples),
            "generated_at": datetime.now().isoformat(),
            **(metadata or {}),
        },
        "samples": samples,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(samples)} samples → {filepath}")
    return filepath


def load_checkpoint(checkpoint_path: str) -> List[Dict[str, Any]]:
    """Load a previously saved dataset for resuming generation."""
    if not os.path.exists(checkpoint_path):
        return []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", [])
    logger.info(f"Loaded checkpoint with {len(samples)} existing samples.")
    return samples


def save_checkpoint(
    samples: List[Dict[str, Any]],
    category: str,
    output_dir: str,
) -> str:
    """Save an in-progress checkpoint (overwrites previous checkpoint)."""
    ensure_dir(output_dir)
    filepath = os.path.join(output_dir, f"{category}_checkpoint.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, indent=2, ensure_ascii=False)
    return filepath


def build_sample(
    category: str,
    topic: str,
    pressure_level: str,
    pressure_prompt: str,
    context: str,
    context_type: str,      # "original" | "opposite"
    question: str,
    opinion: str,           # The opinion being pressured toward
    response: str,
    full_prompt: str,
    sample_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Canonical sample schema. Every dataset entry follows this structure.

    Individual components (p, c, q) are stored separately AND as full_prompt.
    """
    return {
        "id": sample_id or f"{category}_{pressure_level}_{context_type}",
        "category": category,
        "topic": topic,
        # ── Components ──────────────────────────────────────────────────────
        "components": {
            "pressure_prompt": pressure_prompt,   # P
            "context": context,                   # C
            "question": question,                 # Q
        },
        # ── Metadata ────────────────────────────────────────────────────────
        "pressure_level": pressure_level,         # low | medium | high
        "context_type": context_type,             # original | opposite
        "opinion": opinion,                       # opinion being pushed
        # ── Full assembled prompt ────────────────────────────────────────────
        "full_prompt": full_prompt,
        # ── Model response ───────────────────────────────────────────────────
        "response": response,
    }