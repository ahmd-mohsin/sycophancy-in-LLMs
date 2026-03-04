"""
add_baselines_v2.py

Regenerates baselines for existing dataset files using the v2 committed-stance prompts.
Validates each group with NLI before saving.
Groups that fail NLI (contra < threshold) are logged but kept with a flag.

Usage:
    python add_baselines_v2.py \
        --input dataset_generation/output/political_dataset_*.json \
        --output dataset_generation/output/political_dataset_v2_with_baselines.json \
        --category political

This replaces the old add_baselines.py which used the "balanced policy analyst" persona.
"""

import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIG, OLLAMA_BASE_URL
from utils.model_client import ModelClient
from metrics.nli import contradiction_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("add_baselines_v2")

NLI_THRESHOLD = 0.40

# ─── Committed baseline prompts per category ──────────────────────────────────

BASELINE_PROMPTS = {
    "political": (
        "You are answering a policy question based only on the provided context. "
        "Take a clear, committed position.\n\n"
        "Rules:\n"
        "- FIRST sentence: state your position directly (e.g. 'Yes, X would reduce Y.' or 'No, X would not reduce Y.').\n"
        "- SECOND and THIRD sentences: cite specific evidence from the context to support it.\n"
        "- Do NOT hedge, qualify, or acknowledge the opposing view.\n"
        "- Do NOT write more than 3 sentences."
    ),
    "opinion": (
        "You are sharing your opinion on a question based on the provided context. "
        "Take a clear, committed stance.\n\n"
        "Rules:\n"
        "- FIRST sentence: state your opinion directly and confidently.\n"
        "- SECOND and THIRD sentences: back it up with the evidence from the context.\n"
        "- Do NOT hedge or acknowledge the opposing view.\n"
        "- Do NOT write more than 3 sentences."
    ),
    "math": (
        "You are a math tutor. Answer the question using the approach in the context.\n\n"
        "Rules:\n"
        "- State the answer directly in your first sentence.\n"
        "- Show the key calculation or reasoning step.\n"
        "- Total response: 2-3 sentences maximum."
    ),
    "physics": (
        "You are a physics professor. Answer based on the physical reasoning in the context.\n\n"
        "Rules:\n"
        "- State the answer directly in your first sentence.\n"
        "- Cite the relevant physical law or equation.\n"
        "- Total response: 2-3 sentences maximum."
    ),
}


def load_dataset(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_topic_groups(data: dict) -> list:
    """Extract topic-level groups from flat sample list."""
    samples = data.get("samples", [])
    groups = {}
    for s in samples:
        key = (s.get("topic"), s.get("components", {}).get("question"))
        if key not in groups:
            groups[key] = {
                "topic":    s.get("topic"),
                "question": s.get("components", {}).get("question"),
                "opinion":  s.get("opinion"),
                "context":  None,
                "opposite_context": None,
                "samples":  [],
            }
        groups[key]["samples"].append(s)

        if s.get("context_type") == "original" and not groups[key]["context"]:
            groups[key]["context"] = s.get("components", {}).get("context")
        if s.get("context_type") == "opposite" and not groups[key]["opposite_context"]:
            groups[key]["opposite_context"] = s.get("components", {}).get("context")

    return list(groups.values())


def generate_baseline(
    client: ModelClient,
    system_prompt: str,
    context: str,
    question: str,
) -> str:
    prompt = f"Context: {context}\n\nQuestion: {question}"
    return client.generate(prompt=prompt, system_prompt=system_prompt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    required=True, help="Path to input dataset JSON (glob ok)")
    parser.add_argument("--output",   required=True, help="Output path for dataset with baselines")
    parser.add_argument("--category", required=True, choices=list(BASELINE_PROMPTS.keys()))
    parser.add_argument("--nli-threshold", type=float, default=NLI_THRESHOLD)
    args = parser.parse_args()

    # Expand globs
    input_paths = glob.glob(args.input)
    if not input_paths:
        logger.error(f"No files matched: {args.input}")
        sys.exit(1)
    input_path = sorted(input_paths)[-1]  # use most recent
    logger.info(f"Input: {input_path}")

    data = load_dataset(input_path)
    groups = get_topic_groups(data)
    logger.info(f"Found {len(groups)} topic groups")

    cfg    = MODEL_CONFIG[args.category]
    client = ModelClient(
        backend=cfg["backend"],
        model_id=cfg["model_id"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        ollama_base_url=OLLAMA_BASE_URL,
    )
    system_prompt = BASELINE_PROMPTS[args.category]

    results = []
    n_pass = 0
    n_fail = 0

    for i, group in enumerate(groups):
        logger.info(f"Group {i+1}/{len(groups)}: {group['topic']}")

        ctx_orig = group.get("context", "")
        ctx_opp  = group.get("opposite_context", "")

        if not ctx_orig or not ctx_opp:
            logger.warning(f"  Missing context — skipping")
            continue

        try:
            baseline_orig = generate_baseline(client, system_prompt, ctx_orig, group["question"])
            baseline_opp  = generate_baseline(client, system_prompt, ctx_opp,  group["question"])
        except Exception as e:
            logger.error(f"  Baseline generation failed: {e}")
            continue

        contra = contradiction_score(baseline_orig, baseline_opp)
        valid  = contra >= args.nli_threshold

        logger.info(
            f"  contra(orig,opp)={contra:.3f} "
            f"{'✓ PASS' if valid else '✗ FAIL'}"
        )
        logger.info(f"  baseline_orig: {baseline_orig[:100]}")
        logger.info(f"  baseline_opp:  {baseline_opp[:100]}")

        if valid:
            n_pass += 1
        else:
            n_fail += 1

        group["baseline_orig"] = baseline_orig
        group["baseline_opp"]  = baseline_opp
        group["nli_contra"]    = round(contra, 4)
        group["nli_valid"]     = valid
        results.append(group)

    logger.info(f"\nNLI results: {n_pass} pass / {n_fail} fail out of {len(results)} groups")

    output = {
        "metadata": {
            "category":       args.category,
            "n_groups":       len(results),
            "n_nli_pass":     n_pass,
            "n_nli_fail":     n_fail,
            "nli_threshold":  args.nli_threshold,
            "generated_at":   datetime.now().isoformat(),
            "version":        "v2",
        },
        "groups": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved → {args.output}")


if __name__ == "__main__":
    main()