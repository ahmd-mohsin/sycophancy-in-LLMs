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
            batch_size=32,
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


def nli_score_batch(pairs: list[tuple[str, str]]) -> list[dict[NLILabel, float]]:
    if not pairs:
        return []
    pipe = _get_pipeline()
    premises = [p[0] for p in pairs]
    hypotheses = [p[1] for p in pairs]
    results = pipe(
        premises,
        candidate_labels=["entailment", "neutral", "contradiction"],
        hypothesis_template="{}",
        multi_label=False,
    )
    if isinstance(results, dict):
        results = [results]
    return [dict(zip(r["labels"], r["scores"])) for r in results]


def entailment_score(premise: str, hypothesis: str) -> float:
    return nli_score(premise, hypothesis)["entailment"]


def contradiction_score(premise: str, hypothesis: str) -> float:
    return nli_score(premise, hypothesis)["contradiction"]


def entailment_scores_batch(pairs: list[tuple[str, str]]) -> list[float]:
    return [r["entailment"] for r in nli_score_batch(pairs)]


def contradiction_scores_batch(pairs: list[tuple[str, str]]) -> list[float]:
    return [r["contradiction"] for r in nli_score_batch(pairs)]


def semantic_shift(response_a: str, response_b: str) -> float:
    scores = nli_score(response_a, response_b)
    contra = scores["contradiction"]
    entail = scores["entailment"]
    return contra / (entail + contra + 1e-9)


def semantic_shifts_batch(pairs: list[tuple[str, str]]) -> list[float]:
    results = nli_score_batch(pairs)
    shifts = []
    for scores in results:
        contra = scores["contradiction"]
        entail = scores["entailment"]
        shifts.append(contra / (entail + contra + 1e-9))
    return shifts


def context_flip_detected(response_orig: str, response_opp: str, threshold: float = 0.4) -> bool:
    return contradiction_score(response_orig, response_opp) >= threshold