import os

HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_DEVICE = os.getenv("HF_DEVICE", "auto")
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.3"))

NUM_REASONING_CHAINS = 3
NUM_SOURCE_CHAINS = 3
CHAIN_TEMPERATURE = 0.3
ADVERSARIAL_CHAIN_TEMPERATURE = 0.5

FACTUAL_HIGH_CONFIDENCE = 1.0
FACTUAL_MEDIUM_CONFIDENCE = 0.67
FACTUAL_LOW_CONFIDENCE = 0.33
TIME_SENSITIVE_MAX_CONFIDENCE = 0.9
SUBJECTIVE_MAX_CONFIDENCE = 0.8

MINIMUM_AGREEMENT_RATIO = 0.67
UNCERTAIN_THRESHOLD = 0.5

HIGH_PRESSURE_RISK_LEVELS = ["high", "medium"]

PRESSURE_TYPES_REQUIRING_SUBSTANCE = [
    "authority_pressure",
    "authority_credential",
    "authority_institution",
    "authority_experience",
    "authority_publication",
    "authority_status",
    "authority_research",
    "authority_framing",
    "emotional_framing",
    "certainty_challenge",
    "appeal_to_consensus",
    "guilt_tripping",
    "pressure_to_change",
    "consensus_pressure",
    "leading_question",
    "direct_challenge",
    "emotional_intensifier",
    "fear_framing",
    "settled_science_framing",
    "political_identity",
    "values_signaling",
    "opinion_steering",
    "credibility_challenge",
    "skepticism_expression",
    "disagreement_signal",
    "persistent_disagreement",
]

SUBSTANTIVE_SIGNAL_TYPES = [
    "appeal_to_external_sources",
    "appeal_to_common_knowledge",
    "personal_experience_appeal",
]

MAX_CONVERSATION_SUMMARY_TURNS = 10