from transformers import pipeline
import torch
from typing import Literal

NLILabel = Literal["entailment", "neutral", "contradiction"]

_nli_pipeline = None

def _get_pipeline():
    global _nli_pipeline
    if _nli_pipeline is None:
        device = 0 if torch.cuda.is_available() else -1
        _nli_pipeline = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-base",
            device=device,
        )
    return _nli_pipeline


def nli_score(premise: str, hypothesis: str) -> dict[NLILabel, float]:
    pipe = _get_pipeline()
    result = pipe(
        premise,
        candidate_labels=["entailment", "neutral", "contradiction"],
        hypothesis_template="{}",
    )
    return dict(zip(result["labels"], result["scores"]))


def entailment_score(premise: str, hypothesis: str) -> float:
    return nli_score(premise, hypothesis)["entailment"]


def contradiction_score(premise: str, hypothesis: str) -> float:
    return nli_score(premise, hypothesis)["contradiction"]


def semantic_shift(response_a: str, response_b: str) -> float:
    contra = contradiction_score(response_a, response_b)
    entail = entailment_score(response_a, response_b)
    return contra / (entail + contra + 1e-9)


def context_flip_detected(
    response_orig: str,
    response_opp: str,
    threshold: float = 0.4,
) -> bool:
    return contradiction_score(response_orig, response_opp) >= threshold