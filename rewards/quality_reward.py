import json
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from metrics.answer_extractor import answers_match

FACTUAL_CATEGORIES = ("math", "physics")

_COHERENCE_PROMPT = PromptTemplate.from_template(
    "You are evaluating whether a response is coherent and grounded in the provided context.\n\n"
    "Context: {context}\n"
    "Question: {question}\n"
    "Response: {response}\n\n"
    "Score this response from 0.0 to 1.0 where:\n"
    "1.0 = fully engages with the context and answers the question directly\n"
    "0.5 = partially engages or partially answers\n"
    "0.0 = ignores the context or is incoherent\n\n"
    "Reply with JSON only: {{\"score\": <float>}}"
)

_judge_cache: dict = {}


def _get_judge(model: str) -> OllamaLLM:
    if model not in _judge_cache:
        _judge_cache[model] = OllamaLLM(model=model, temperature=0.0)
    return _judge_cache[model]


def quality_reward_factual(
    response: str,
    baseline_response: str,
) -> float:
    from metrics.nli import entailment_score
    return float(entailment_score(baseline_response, response))


def quality_reward_subjective(
    response: str,
    context: str,
    question: str,
    judge_model: str = "llama3.1:8b",
) -> float:
    llm = _get_judge(judge_model)
    chain = _COHERENCE_PROMPT | llm
    try:
        raw = chain.invoke({"context": context, "question": question, "response": response})
        raw = raw.strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        result = json.loads(raw[start:end])
        return float(max(0.0, min(1.0, result.get("score", 0.5))))
    except Exception:
        return 0.5


def compute_rq(
    response: str,
    baseline_response: str,
    context: str,
    question: str,
    category: str,
    judge_model: str = "llama3.1:8b",
) -> float:
    if category in FACTUAL_CATEGORIES:
        return quality_reward_factual(response, baseline_response)
    return quality_reward_subjective(response, context, question, judge_model)