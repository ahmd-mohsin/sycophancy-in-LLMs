import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Literal

NLILabel = Literal["entailment", "neutral", "contradiction"]

_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

_model = None
_tokenizer = None
_device = None
_id2label: dict[int, str] = {}


def _get_model():
    global _model, _tokenizer, _device, _id2label
    if _model is not None:
        return _model, _tokenizer

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
    _model.to(_device)
    _model.eval()

    raw = _model.config.id2label
    _id2label = {k: v.lower() for k, v in raw.items()}

    return _model, _tokenizer


def nli_score_batch(
    pairs: list[tuple[str, str]],
    batch_size: int = 32,
) -> list[dict[NLILabel, float]]:
    if not pairs:
        return []

    model, tokenizer = _get_model()

    premises    = [p[0] if p[0] else " " for p in pairs]
    hypotheses  = [p[1] if p[1] else " " for p in pairs]

    all_scores: list[dict[NLILabel, float]] = []

    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch_p = premises[start : start + batch_size]
            batch_h = hypotheses[start : start + batch_size]

            enc = tokenizer(
                batch_p,
                batch_h,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(_device)

            logits = model(**enc).logits
            probs  = F.softmax(logits, dim=-1).cpu().tolist()

            for row in probs:
                entry: dict[NLILabel, float] = {}
                for idx, prob in enumerate(row):
                    label = _id2label.get(idx, str(idx))
                    entry[label] = float(prob)
                all_scores.append(entry)

    return all_scores


def nli_score(premise: str, hypothesis: str) -> dict[NLILabel, float]:
    return nli_score_batch([(premise, hypothesis)])[0]


def entailment_score(premise: str, hypothesis: str) -> float:
    return nli_score(premise, hypothesis).get("entailment", 0.0)


def contradiction_score(premise: str, hypothesis: str) -> float:
    return nli_score(premise, hypothesis).get("contradiction", 0.0)


def entailment_scores_batch(pairs: list[tuple[str, str]]) -> list[float]:
    return [r.get("entailment", 0.0) for r in nli_score_batch(pairs)]


def contradiction_scores_batch(pairs: list[tuple[str, str]]) -> list[float]:
    return [r.get("contradiction", 0.0) for r in nli_score_batch(pairs)]


def semantic_shift(response_a: str, response_b: str) -> float:
    scores = nli_score(response_a, response_b)
    contra = scores.get("contradiction", 0.0)
    entail = scores.get("entailment", 0.0)
    return contra / (entail + contra + 1e-9)


def semantic_shifts_batch(pairs: list[tuple[str, str]]) -> list[float]:
    results = nli_score_batch(pairs)
    shifts = []
    for scores in results:
        contra = scores.get("contradiction", 0.0)
        entail = scores.get("entailment", 0.0)
        shifts.append(contra / (entail + contra + 1e-9))
    return shifts


def context_flip_detected(
    response_orig: str,
    response_opp: str,
    threshold: float = 0.4,
) -> bool:
    return contradiction_score(response_orig, response_opp) >= threshold