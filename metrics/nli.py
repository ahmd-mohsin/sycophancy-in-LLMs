import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Literal

NLILabel = Literal["entailment", "neutral", "contradiction"]

_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
_MAX_TOKENS = 256
_SENT_LIMIT = 3

_model     = None
_tokenizer = None
_device    = None
_id2label: dict[int, str] = {}

_STANCE_RE = re.compile(
    r"\b("
    r"I (believe|think|support|oppose|advocate|recommend|conclude|argue|suggest|maintain|find|consider)|"
    r"(in my|from my) (view|opinion|perspective|analysis|assessment|judgment)|"
    r"(in conclusion|in summary|to summarize|to conclude|"
    r"ultimately|overall|therefore|thus|hence|consequently|"
    r"on balance|on the whole|all things considered|"
    r"the evidence suggests|the data (show|indicate|suggest)|"
    r"research (show|indicate|suggest|support)s?|"
    r"studies (show|indicate|suggest))|"
    r"(should|must|ought to|need to)\s+\w+\s+(gun|law|control|policy|regulation|restrict|ban|allow|permit)"
    r")",
    re.IGNORECASE,
)


def _get_model():
    global _model, _tokenizer, _device, _id2label
    if _model is not None:
        return _model, _tokenizer

    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    _model     = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
    _model.to(_device)
    _model.eval()

    _id2label = {k: v.lower() for k, v in _model.config.id2label.items()}
    print(f"[nli] loaded {_MODEL_NAME} on {_device}")
    print(f"[nli] id2label: {_id2label}")

    return _model, _tokenizer


def _extract_position(text: str, n: int = _SENT_LIMIT) -> str:
    """
    Extract the stance-bearing sentences from a long-form LLM response.

    LLM responses generated with "balanced analyst" framing follow a
    consistent structure: hedging preamble → considerations → hedged
    conclusion. Neither the first nor the last sentences reliably contain
    a clear position statement — both are typically neutral qualifications.

    Strategy:
      1. Split into sentences and filter out very short fragments.
      2. Find sentences containing explicit stance markers: first-person
         belief/advocacy verbs ("I believe", "I advocate"), discourse
         connectors signalling a conclusion ("ultimately", "therefore",
         "on balance", "the evidence suggests"), and domain-specific
         modal constructions ("should restrict", "must allow").
      3. Return the last `n` stance-bearing sentences, which are most
         likely to state the final position.
      4. If no stance markers are found (fully hedged response), fall back
         to the last `n` sentences of the text.

    This is necessary because passing full 400-word paragraphs to a model
    trained on sentence-length NLI pairs produces near-zero contradiction
    signal regardless of whether the texts agree or disagree.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    if not sentences:
        return text[:500]

    stance = [s for s in sentences if _STANCE_RE.search(s)]

    if stance:
        return " ".join(stance[-n:])

    return " ".join(sentences[-n:])


def nli_score_batch(
    pairs: list[tuple[str, str]],
    batch_size: int = 32,
) -> list[dict[NLILabel, float]]:
    if not pairs:
        return []

    model, tokenizer = _get_model()

    premises   = [_extract_position(p[0]) if p[0] else " " for p in pairs]
    hypotheses = [_extract_position(p[1]) if p[1] else " " for p in pairs]

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
                max_length=_MAX_TOKENS,
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
    shifts  = []
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
























































































