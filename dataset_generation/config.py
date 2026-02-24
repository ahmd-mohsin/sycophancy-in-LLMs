"""
Configuration for sycophancy dataset generation.
Each category uses a different model.
"""

from dataclasses import dataclass, field
from typing import Optional

# ─── Model Assignments ───────────────────────────────────────────────────────
# Swap any of these to a HuggingFace model ID by setting backend="huggingface"

MODEL_CONFIG = {
    "math": {
        "backend": "ollama",          # "ollama" | "huggingface"
        "model_id": "deepseek-r1:7b", # ollama model name OR HF repo id
        "temperature": 0.3,
        "max_tokens": 512,
    },
    "physics": {
        "backend": "ollama",
        "model_id": "mistral:7b",
        "temperature": 0.3,
        "max_tokens": 512,
    },
    "political": {
        "backend": "ollama",
        "model_id": "llama3.2:3b",
        "temperature": 0.7,
        "max_tokens": 512,
    },
    "opinion": {
        "backend": "ollama",
        "model_id": "gemma2:2b",
        "temperature": 0.8,
        "max_tokens": 512,
    },
    # Orchestrator model: used to generate (C, Q) seed pairs
    "orchestrator": {
        "backend": "ollama",
        "model_id": "llama3.2:3b",
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