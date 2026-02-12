import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_DEVICE = os.getenv("HF_DEVICE", "auto")
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.3"))

CONVERSATION_HISTORY_PATH = os.getenv(
    "CONVERSATION_HISTORY_PATH",
    str(BASE_DIR / "conversation_history.json"),
)

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "50"))

SYCOPHANCY_REMOVAL_PATTERNS = [
    "no,",
    "no ",
    "wrong",
    "are you sure",
    "i think",
    "actually",
    "that's not right",
    "you're wrong",
    "incorrect",
    "that's incorrect",
    "i disagree",
    "i believe",
    "you're mistaken",
    "not true",
    "that's false",
]

BIAS_KEYWORDS = [
    "you agree",
    "you said so",
    "you confirmed",
    "you admitted",
    "even you think",
    "as you said",
    "you already agreed",
    "you just said",
    "see, you think so too",
    "obviously",
    "clearly",
    "everyone knows",
    "it's common knowledge",
    "undeniably",
    "without question",
    "no one would disagree",
]

TEMPORAL_MARKERS = [
    "current",
    "now",
    "today",
    "latest",
    "who is",
    "2024",
    "2025",
    "2026",
    "recently",
    "at the moment",
    "presently",
    "right now",
    "as of",
    "this year",
    "this month",
]

FACTUAL_INDICATOR_KEYWORDS = {
    "single_answer": [
        "what is",
        "how many",
        "how much",
        "when did",
        "where is",
        "who was",
        "which",
        "name the",
        "define",
    ],
    "calculation": [
        "calculate",
        "compute",
        "solve",
        "what is the sum",
        "what is the product",
        "evaluate",
        "+",
        "-",
        "*",
        "/",
        "=",
    ],
    "historical": [
        "in what year",
        "when was",
        "who founded",
        "who invented",
        "who discovered",
        "history of",
        "origin of",
    ],
    "scientific": [
        "what causes",
        "what is the formula",
        "what element",
        "what compound",
        "scientific name",
        "what is the speed of",
        "law of",
        "theory of",
    ],
}

LANGCHAIN_CHUNK_SIZE = int(os.getenv("LANGCHAIN_CHUNK_SIZE", "1000"))
LANGCHAIN_CHUNK_OVERLAP = int(os.getenv("LANGCHAIN_CHUNK_OVERLAP", "200"))