"""
Configuration for sycophancy dataset generation.
Each category uses a different model.
All models are 8B or above.
"""

from dataclasses import dataclass, field
from typing import Optional

# ─── Model Assignments ───────────────────────────────────────────────────────
# Swap any of these to a HuggingFace model ID by setting backend="huggingface"

MODEL_CONFIG = {
    "math": {
        "backend": "ollama",
        "model_id": "llama3.1:8b",       # strong reasoning, stable with system prompts
        "temperature": 0.3,
        "max_tokens": 512,
    },
    "physics": {
        "backend": "ollama",
        "model_id": "llama3.1:8b",   # Fast general model
        "temperature": 0.3,
        "max_tokens": 512,
    },
    "political": {
        "backend": "ollama",
        "model_id": "llama3.1:8b",        # good at nuanced opinion
        "temperature": 0.7,
        "max_tokens": 512,
    },
    "opinion": {
        "backend": "ollama",
        "model_id": "gemma2:9b",          # natural conversational tone
        "temperature": 0.8,
        "max_tokens": 512,
    },
    # Orchestrator model: used to generate (C, Q) seed pairs
    "orchestrator": {
        "backend": "ollama",
        "model_id": "llama3.1:8b",        # reliable JSON generation
        "temperature": 0.9,
        "max_tokens": 1024,
    },
}

# ─── Dataset Sizes ───────────────────────────────────────────────────────────
DATASET_CONFIG = {
    "math":      {"n_topics": 100},
    "physics":   {"n_topics": 100},
    "political": {"n_topics": 100},
    "opinion":   {"n_topics": 100},
}

# ─── Pressure Levels ─────────────────────────────────────────────────────────
PRESSURE_LEVELS = ["low", "medium", "high"]

# ─── Output ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
OLLAMA_BASE_URL = "http://localhost:11434"