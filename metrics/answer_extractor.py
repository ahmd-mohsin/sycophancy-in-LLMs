import re
from typing import Optional


_PATTERNS = [
    r"(?:the\s+)?answer\s+is\s*[:\=]?\s*([-+]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
    r"(?:=|equals)\s*([-+]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*(?:$|[^\d])",
    r"\\boxed\{([^}]+)\}",
    r"therefore[,\s]+(?:the\s+)?(?:answer|result|solution)\s+(?:is\s+)?([-+]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
    r"(?:x|y|z|n|k)\s*=\s*([-+]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
    r"([-+]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*(?:m/s|kg|J|N|W|Pa|K|mol|rad|s|Hz|V|A|Ω|C|T|m|km|cm|mm)",
    r"^\s*([-+]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$",
]

_FRACTION_PATTERN = re.compile(r"\\frac\{([^}]+)\}\{([^}]+)\}")


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_answer(response: str) -> Optional[str]:
    for pattern in _PATTERNS:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

    frac_match = _FRACTION_PATTERN.search(response)
    if frac_match:
        num, den = frac_match.group(1), frac_match.group(2)
        try:
            return str(float(num) / float(den))
        except (ValueError, ZeroDivisionError):
            return f"{num}/{den}"

    return None


def answers_match(response: str, ground_truth: str, tolerance: float = 1e-4) -> bool:
    extracted = extract_answer(response)
    if extracted is None:
        return False

    try:
        return abs(float(extracted) - float(ground_truth)) <= tolerance
    except ValueError:
        return _normalize(extracted) == _normalize(ground_truth)


def extract_latex_answer(response: str) -> Optional[str]:
    box_match = re.search(r"\\boxed\{([^}]+)\}", response)
    if box_match:
        return box_match.group(1).strip()
    return extract_answer(response)