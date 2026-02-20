"""
Module 1 Evaluation — Self-Curated Datasets
=============================================

Module 1 does THREE things:
  1. DETECT sycophancy (three tiers: keywords → regex → LLM judge)
  2. SCORE it (continuous 0-1 + risk level + detected pressure types)
  3. EXTRACT the unbiased core claim

This eval tests all three on your self-curated datasets.

Per sample (2 LLM calls via Mistral):
  1. BiasDetector.detect_bias() on raw user_message (tier 1+2+3)
  2. BiasDetector.neutralize() to extract unbiased core claim

NO DeepSeek generation. NO shift measurement. That's Module 2's job.

Usage:
    python -m module1.eval.eval_self_curated \
        --data-dir data/self_curated \
        --output m1_eval_results.json
"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore")
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

import re
import json
import argparse
import gc
import torch
from typing import List, Dict
from tqdm import tqdm
from collections import defaultdict

from transformers import pipeline as hf_pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline

from module1.bias_detector import BiasDetector
from module1.question_classifier import QuestionClassifier


# =====================================================================
# MODEL LOADING — only Mistral (for tier-3 judge + neutralization)
# =====================================================================

MISTRAL = None
LC_MISTRAL = None


def load_model():
    global MISTRAL, LC_MISTRAL

    if MISTRAL is not None:
        return

    gc.collect()
    torch.cuda.empty_cache()

    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading Mistral 7B Instruct (4-bit)...")
    MISTRAL = hf_pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.2",
        device_map="auto",
        model_kwargs={"quantization_config": qconfig},
    )
    LC_MISTRAL = HuggingFacePipeline(pipeline=MISTRAL)
    print("Mistral loaded.")


# =====================================================================
# UTILS
# =====================================================================

def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    data = []
    for line in content.split("\n"):
        line = line.strip()
        if line:
            data.append(json.loads(line))
    return data


def safe_str(value) -> str:
    if isinstance(value, list):
        return value[0].strip() if value else ""
    return str(value).strip()


# =====================================================================
# PER-SAMPLE EVALUATION
# =====================================================================

def evaluate_sample(
    sample: dict,
    bias_detector: BiasDetector,
    question_classifier: QuestionClassifier,
) -> dict:

    sid = sample.get("id", "")
    syc_type = sample.get("sycophancy_type", "unknown")
    q_type_dataset = sample.get("question_type", "unknown")
    user_message = sample.get("user_message", "")
    question = sample.get("question", "")
    biased_prompt = f"{user_message}\n\n{question}".strip()

    # === 1. THREE-TIER DETECTION on raw user_message ===
    detection_user = bias_detector.detect_bias(user_message, use_llm=True)

    # Also detect on full biased prompt (catches patterns spanning both fields)
    detection_full = bias_detector.detect_bias(biased_prompt, use_llm=True)

    # Take the higher-scoring detection
    if detection_full.get("sycophancy_score", 0) > detection_user.get("sycophancy_score", 0):
        detection = detection_full
        detection_source = "full_prompt"
    else:
        detection = detection_user
        detection_source = "user_message"

    # === 2. QUESTION CLASSIFICATION ===
    q_class = question_classifier.classify(question, user_message)

    # === 3. EXTRACT UNBIASED CORE CLAIM ===
    neutralization = bias_detector.neutralize(user_message, use_llm=True)

    # Also neutralize the full prompt
    neutralization_full = bias_detector.neutralize(biased_prompt, use_llm=True)

    # LLM tier-3 judge details
    t3 = detection.get("llm_judge", {})

    return {
        "id": sid,
        "sycophancy_type": syc_type,
        "question_type_dataset": q_type_dataset,
        "question_type_classified": q_class.get("question_type", "unknown"),
        "classification_method": q_class.get("method", "unknown"),
        # Original texts
        "user_message": user_message,
        "question": question,
        "biased_prompt": biased_prompt,
        # Detection results
        "detection": {
            "source": detection_source,
            "bias_detected": detection.get("has_bias", False),
            "sycophancy_risk": detection.get("sycophancy_risk", "none"),
            "sycophancy_score": detection.get("sycophancy_score", 0.0),
            "score_components": detection.get("score_components", {}),
            "detected_types": detection.get("detected_types", []),
            # Tier 1: explicit keywords
            "tier1": {
                "keywords_found": detection.get("keywords_found", []),
                "bias_density": detection.get("bias_density", 0),
            },
            # Tier 2: regex patterns
            "tier2": {
                "opinion_steering": [
                    {"pattern": m["pattern"], "matched": m["matched_text"]}
                    for m in detection.get("implicit_bias", {}).get("opinion_steering", [])
                ],
                "authority_pressure": [
                    {"pattern": m["pattern"], "matched": m["matched_text"]}
                    for m in detection.get("implicit_bias", {}).get("authority_pressure", [])
                ],
                "emotional_framing": [
                    {"pattern": m["pattern"], "matched": m["matched_text"]}
                    for m in detection.get("implicit_bias", {}).get("emotional_framing", [])
                ],
            },
            # Tier 3: LLM judge
            "tier3": {
                "sycophancy_detected": t3.get("sycophancy_detected", False),
                "risk_level": t3.get("risk_level", "none"),
                "opinion_steering": t3.get("opinion_steering", False),
                "authority_pressure": t3.get("authority_pressure", False),
                "identity_framing": t3.get("identity_framing", False),
                "emotional_pressure": t3.get("emotional_pressure", False),
                "explicit_manipulation": t3.get("explicit_manipulation", False),
                "biographical_priming": t3.get("biographical_priming", False),
                "reason": t3.get("reason", ""),
                "raw_output": t3.get("raw_output", "")[:600],
            },
        },
        # Unbiased core claim extraction
        "core_claim": {
            "original": user_message,
            "neutralized": neutralization.get("neutralized", ""),
            "was_modified": neutralization.get("was_modified", False),
            "full_prompt_neutralized": neutralization_full.get("neutralized", ""),
        },
    }


# =====================================================================
# DATASET EVALUATION
# =====================================================================

def evaluate_subset(
    name: str,
    data: List[Dict],
    max_samples: int,
    bias_detector: BiasDetector,
    question_classifier: QuestionClassifier,
    output_path: str,
):
    if max_samples > 0:
        data = data[:max_samples]

    all_output = {}
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            all_output = json.load(f)

    existing = all_output.get(name, {}).get("results", [])
    start_idx = len(existing)

    if start_idx >= len(data):
        print(f"\n{name}: Already completed ({start_idx}/{len(data)}). Skipping.")
        return all_output.get(name, {})

    if start_idx > 0:
        print(f"\nResuming {name} from sample {start_idx}/{len(data)}")

    results = existing

    print(f"\n{'='*60}")
    print(f"DATASET: {name.upper()} ({len(data)} samples, starting at {start_idx})")
    print(f"{'='*60}")

    for i in tqdm(range(start_idx, len(data)), desc=name, unit="q", initial=start_idx, total=len(data)):
        sample = data[i]
        try:
            result = evaluate_sample(sample, bias_detector, question_classifier)
        except Exception as e:
            result = {
                "id": sample.get("id", ""),
                "error": str(e),
                "sycophancy_type": sample.get("sycophancy_type", "unknown"),
                "detection": {"sycophancy_score": 0.0, "detected_types": []},
            }

        results.append(result)

        # Save after every sample
        summary = compute_summary(name, results)
        all_output[name] = {"summary": summary, "results": results}
        with open(output_path, "w") as f:
            json.dump(all_output, f, indent=2, default=str)

    print_summary(summary)
    return all_output[name]


def compute_summary(name: str, results: list) -> dict:
    total = len(results)
    score_sum = 0.0
    risk_counts = defaultdict(int)
    tier1_hit = 0
    tier2_hit = 0
    tier3_hit = 0
    by_type = defaultdict(lambda: {"total": 0, "score_sum": 0.0, "detected": 0})
    claim_modified = 0

    for r in results:
        d = r.get("detection", {})
        s = d.get("sycophancy_score", 0.0)
        score_sum += s
        risk_counts[d.get("sycophancy_risk", "none")] += 1

        if d.get("tier1", {}).get("keywords_found"):
            tier1_hit += 1
        t2 = d.get("tier2", {})
        if t2.get("opinion_steering") or t2.get("authority_pressure") or t2.get("emotional_framing"):
            tier2_hit += 1
        if d.get("tier3", {}).get("sycophancy_detected"):
            tier3_hit += 1

        if d.get("bias_detected"):
            by_type_key = r.get("sycophancy_type", "unknown")
            by_type[by_type_key]["detected"] += 1

        syc_type = r.get("sycophancy_type", "unknown")
        by_type[syc_type]["total"] += 1
        by_type[syc_type]["score_sum"] += s

        if r.get("core_claim", {}).get("was_modified"):
            claim_modified += 1

    type_breakdown = {}
    for syc_type, stats in by_type.items():
        t = stats["total"]
        type_breakdown[syc_type] = {
            "total": t,
            "detected": stats["detected"],
            "detection_rate": round(stats["detected"] / max(t, 1) * 100, 1),
            "avg_score": round(stats["score_sum"] / max(t, 1), 3),
        }

    return {
        "dataset": name,
        "total": total,
        "avg_sycophancy_score": round(score_sum / max(total, 1), 3),
        "risk_distribution": dict(risk_counts),
        "detection_rates": {
            "tier1_keyword": round(tier1_hit / max(total, 1) * 100, 1),
            "tier2_regex": round(tier2_hit / max(total, 1) * 100, 1),
            "tier3_llm_judge": round(tier3_hit / max(total, 1) * 100, 1),
            "any_tier": round(sum(1 for r in results if r.get("detection", {}).get("bias_detected")) / max(total, 1) * 100, 1),
        },
        "claim_extraction": {
            "modified": claim_modified,
            "modification_rate": round(claim_modified / max(total, 1) * 100, 1),
        },
        "by_sycophancy_type": type_breakdown,
    }


def print_summary(summary: dict):
    print(f"\n--- {summary['dataset']} Results ---")
    print(f"  Total samples:        {summary['total']}")
    print(f"  Avg sycophancy score: {summary['avg_sycophancy_score']}")
    print(f"  Risk distribution:    {summary['risk_distribution']}")
    dr = summary.get("detection_rates", {})
    print(f"  Detection — Tier 1 (keywords): {dr.get('tier1_keyword', 0)}%")
    print(f"  Detection — Tier 2 (regex):    {dr.get('tier2_regex', 0)}%")
    print(f"  Detection — Tier 3 (LLM):      {dr.get('tier3_llm_judge', 0)}%")
    print(f"  Detection — Any tier:          {dr.get('any_tier', 0)}%")
    ce = summary.get("claim_extraction", {})
    print(f"  Claims modified:      {ce.get('modified', 0)} ({ce.get('modification_rate', 0)}%)")
    print(f"\n  By sycophancy type:")
    for syc_type, stats in summary.get("by_sycophancy_type", {}).items():
        print(f"    {syc_type}: {stats['total']} samples, "
              f"detected={stats['detection_rate']}%, avg_score={stats['avg_score']}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default="m1_eval_results.json")
    args = parser.parse_args()

    load_model()

    bias_detector = BiasDetector(llm_pipeline=LC_MISTRAL)
    question_classifier = QuestionClassifier(llm_pipeline=LC_MISTRAL)

    subsets = {
        "claude": "claude.jsonl",
        "chatgpt": "chatgpt.jsonl",
        "perplexity": "perplexity.jsonl",
    }

    for name, filename in subsets.items():
        path = os.path.join(args.data_dir, filename)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        data = load_jsonl(path)
        print(f"Loaded {len(data)} samples from {filename}")
        evaluate_subset(name, data, args.max_samples, bias_detector, question_classifier, args.output)

    # Aggregate
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            all_results = json.load(f)
        datasets = {k: v for k, v in all_results.items() if k != "aggregate"}
        if datasets:
            all_list = []
            for ds in datasets.values():
                all_list.extend(ds.get("results", []))
            agg = compute_summary("aggregate", all_list)
            all_results["aggregate"] = {"summary": agg}
            print(f"\n{'='*60}")
            print("AGGREGATE")
            print(f"{'='*60}")
            print_summary(agg)
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()