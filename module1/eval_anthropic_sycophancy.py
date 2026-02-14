import json
import os
import sys
import time
import re
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from bias_detector import BiasDetector
from module1_main import Module1


def load_dataset(filepath: str, max_samples: int = None) -> list[dict]:
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
            if max_samples and len(data) >= max_samples:
                break
    return data


def extract_biography(question: str) -> str:
    match = re.search(
        r"(Hello, my name is .+?)(?:Do you agree|If you had to choose|Which)",
        question,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return question[:500]


def extract_question_only(full_question: str) -> str:
    markers = [
        "Do you agree or disagree",
        "If you had to choose",
        "Which of the following",
        "What is your view",
    ]
    for marker in markers:
        idx = full_question.find(marker)
        if idx != -1:
            return full_question[idx:].strip()
    return full_question[-500:]


def evaluate_detection(data: list[dict], dataset_name: str, use_llm: bool = False, llm_pipeline=None) -> dict:

    detector = BiasDetector(llm_pipeline=llm_pipeline)
    module = Module1(use_llm=False)

    results = {
        "dataset": dataset_name,
        "total_samples": len(data),
        "detection": {
            "explicit_bias_caught": 0,
            "implicit_bias_caught": 0,
            "any_bias_caught": 0,
            "llm_judge_caught": 0,
            "total_sycophancy_detected": 0,
            "missed": 0,
        },
        "risk_levels": defaultdict(int),
        "implicit_breakdown": {
            "opinion_steering": 0,
            "authority_pressure": 0,
            "emotional_framing": 0,
        },
        "classification": {
            "subjective": 0,
            "factual": 0,
            "time_sensitive": 0,
        },
        "ground_truth_comparison": {
            "true_positives": 0,
            "false_negatives": 0,
        },
        "sample_analyses": [],
    }

    for i, sample in enumerate(data):
        question = sample["question"]
        sycophantic_answer = sample["answer_matching_behavior"].strip()
        independent_answer = sample["answer_not_matching_behavior"]
        if isinstance(independent_answer, list):
            independent_answer = independent_answer[0].strip()
        else:
            independent_answer = independent_answer.strip()

        biography = extract_biography(question)
        actual_question = extract_question_only(question)

        bias_result = detector.detect_bias(biography, use_llm=use_llm)

        has_explicit = bias_result["explicit_bias"]["has_bias"]
        has_implicit = bias_result["implicit_bias"]["has_implicit_bias"]
        has_any = bias_result["has_bias"]
        risk = bias_result["sycophancy_risk"]

        if has_explicit:
            results["detection"]["explicit_bias_caught"] += 1
        if has_implicit:
            results["detection"]["implicit_bias_caught"] += 1
        if has_any:
            results["detection"]["any_bias_caught"] += 1

        llm_caught = False
        if use_llm and "llm_judge" in bias_result:
            llm_result = bias_result["llm_judge"]
            if llm_result.get("sycophancy_detected"):
                results["detection"]["llm_judge_caught"] += 1
                llm_caught = True

        detected = has_any or llm_caught
        if detected:
            results["detection"]["total_sycophancy_detected"] += 1
            results["ground_truth_comparison"]["true_positives"] += 1
        else:
            results["detection"]["missed"] += 1
            results["ground_truth_comparison"]["false_negatives"] += 1

        results["risk_levels"][risk] += 1

        for m in bias_result["implicit_bias"]["opinion_steering"]:
            results["implicit_breakdown"]["opinion_steering"] += 1
            break
        for m in bias_result["implicit_bias"]["authority_pressure"]:
            results["implicit_breakdown"]["authority_pressure"] += 1
            break
        for m in bias_result["implicit_bias"]["emotional_framing"]:
            results["implicit_breakdown"]["emotional_framing"] += 1
            break

        m1_result = module.process(
            question=actual_question,
            assistant_answer=f"I believe the best answer is {sycophantic_answer}",
            user_challenge=f"I think the answer should be {independent_answer} instead.",
        )
        qtype = m1_result["question_type"]
        results["classification"][qtype] += 1

        if i < 10:
            sample_info = {
                "index": i,
                "biography_snippet": biography[:200] + "..." if len(biography) > 200 else biography,
                "sycophantic_answer": sycophantic_answer,
                "independent_answer": independent_answer,
                "detected": detected,
                "risk_level": risk,
                "explicit_keywords": bias_result["explicit_bias"]["keywords_found"],
                "opinion_steering": [m["matched_text"] for m in bias_result["implicit_bias"]["opinion_steering"]],
                "authority_pressure": [m["matched_text"] for m in bias_result["implicit_bias"]["authority_pressure"]],
                "emotional_framing": [m["matched_text"] for m in bias_result["implicit_bias"]["emotional_framing"]],
                "question_type": qtype,
            }
            if use_llm and "llm_judge" in bias_result:
                lj = bias_result["llm_judge"]
                sample_info["llm_judge"] = {
                    "sycophancy_detected": lj.get("sycophancy_detected"),
                    "risk_level": lj.get("risk_level"),
                    "reason": lj.get("reason", ""),
                }
            results["sample_analyses"].append(sample_info)

        if (i + 1) % 500 == 0:
            det_rate = results["detection"]["total_sycophancy_detected"] / (i + 1) * 100
            print(f"    [{i+1}/{len(data)}] Detection rate: {det_rate:.1f}%")

    total = results["total_samples"]
    tp = results["ground_truth_comparison"]["true_positives"]
    fn = results["ground_truth_comparison"]["false_negatives"]

    results["metrics"] = {
        "detection_rate": tp / total * 100 if total > 0 else 0,
        "miss_rate": fn / total * 100 if total > 0 else 0,
        "explicit_only_rate": results["detection"]["explicit_bias_caught"] / total * 100 if total > 0 else 0,
        "implicit_detection_rate": results["detection"]["implicit_bias_caught"] / total * 100 if total > 0 else 0,
    }

    return results


def print_report(results: dict):
    r = results
    total = r["total_samples"]

    print()
    print("=" * 70)
    print(f"  SYCOPHANCY DETECTION REPORT: {r['dataset']}")
    print("=" * 70)
    print(f"\n  Total samples: {total}")
    print(f"  (Every sample in this dataset IS sycophancy-inducing by design)")

    print(f"\n  --- Detection Results (vs Ground Truth) ---")
    tp = r["ground_truth_comparison"]["true_positives"]
    fn = r["ground_truth_comparison"]["false_negatives"]
    print(f"  Correctly detected (true positives):  {tp}/{total} ({r['metrics']['detection_rate']:.1f}%)")
    print(f"  Missed (false negatives):             {fn}/{total} ({r['metrics']['miss_rate']:.1f}%)")

    print(f"\n  --- Detection Breakdown ---")
    d = r["detection"]
    print(f"  Explicit bias keywords caught:  {d['explicit_bias_caught']}/{total} ({d['explicit_bias_caught']/total*100:.1f}%)")
    print(f"  Implicit patterns caught:       {d['implicit_bias_caught']}/{total} ({d['implicit_bias_caught']/total*100:.1f}%)")
    print(f"  Combined (any method):          {d['any_bias_caught']}/{total} ({d['any_bias_caught']/total*100:.1f}%)")
    if d["llm_judge_caught"] > 0:
        print(f"  LLM judge caught:               {d['llm_judge_caught']}/{total} ({d['llm_judge_caught']/total*100:.1f}%)")
    print(f"  Total detected:                 {d['total_sycophancy_detected']}/{total} ({d['total_sycophancy_detected']/total*100:.1f}%)")

    print(f"\n  --- Risk Level Distribution ---")
    for level in ["high", "medium", "low", "none"]:
        count = r["risk_levels"].get(level, 0)
        print(f"  {level.upper():8s}: {count}/{total} ({count/total*100:.1f}%)")

    print(f"\n  --- Implicit Pattern Breakdown ---")
    ib = r["implicit_breakdown"]
    print(f"  Opinion steering:    {ib['opinion_steering']}/{total} ({ib['opinion_steering']/total*100:.1f}%)")
    print(f"  Authority pressure:  {ib['authority_pressure']}/{total} ({ib['authority_pressure']/total*100:.1f}%)")
    print(f"  Emotional framing:   {ib['emotional_framing']}/{total} ({ib['emotional_framing']/total*100:.1f}%)")

    print(f"\n  --- Question Classification ---")
    c = r["classification"]
    for qtype in ["subjective", "factual", "time_sensitive"]:
        count = c.get(qtype, 0)
        print(f"  {qtype:15s}: {count}/{total} ({count/total*100:.1f}%)")

    if r["sample_analyses"]:
        print(f"\n  --- Sample Analyses (first {len(r['sample_analyses'])}) ---")
        for sa in r["sample_analyses"]:
            detected_str = "DETECTED" if sa["detected"] else "MISSED"
            print(f"\n  Sample {sa['index']} [{detected_str}] (risk: {sa['risk_level']}, type: {sa['question_type']}):")
            print(f"    Bio: {sa['biography_snippet']}")
            if sa["explicit_keywords"]:
                print(f"    Explicit keywords: {sa['explicit_keywords']}")
            if sa["opinion_steering"]:
                print(f"    Opinion steering: {sa['opinion_steering']}")
            if sa["authority_pressure"]:
                print(f"    Authority pressure: {sa['authority_pressure']}")
            if sa["emotional_framing"]:
                print(f"    Emotional framing: {sa['emotional_framing']}")
            if "llm_judge" in sa:
                lj = sa["llm_judge"]
                print(f"    LLM judge: detected={lj['sycophancy_detected']}, risk={lj['risk_level']}")
                if lj.get("reason"):
                    print(f"    LLM reason: {lj['reason']}")

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate sycophancy detection on Anthropic dataset")
    parser.add_argument("--nlp", type=str,
                        default="/home/ahmed/sycophancy-in-LLMs/module1/sycophancy_on_nlp_survey.jsonl")
    parser.add_argument("--political", type=str,
                        default="/home/ahmed/sycophancy-in-LLMs/module1/sycophancy_on_political_typology_quiz.jsonl")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per dataset (default: all)")
    parser.add_argument("--use-llm-judge", action="store_true",
                        help="Use LLM as judge (requires model, slow)")
    parser.add_argument("--output", type=str, default="sycophancy_detection_results.json")
    args = parser.parse_args()

    llm_pipeline = None
    if args.use_llm_judge:
        print("Loading LLM for judge mode...")
        try:
            from transformers import pipeline as hf_pipeline
            from lc_compat import HuggingFacePipeline

            model_name = os.getenv("HF_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
            pipe = hf_pipeline(
                "text-generation",
                model=model_name,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                return_full_text=False,
            )
            llm_pipeline = HuggingFacePipeline(pipeline=pipe)
            print(f"LLM judge loaded: {model_name}")
        except Exception as e:
            print(f"Failed to load LLM judge: {e}")
            print("Continuing with regex-only detection.")

    all_results = {}

    datasets = []
    if os.path.exists(args.nlp):
        datasets.append(("NLP Survey", args.nlp))
    else:
        print(f"NLP dataset not found at {args.nlp}")

    if os.path.exists(args.political):
        datasets.append(("Political Typology", args.political))
    else:
        print(f"Political dataset not found at {args.political}")

    if not datasets:
        print("No datasets found.")
        sys.exit(1)

    for name, path in datasets:
        print(f"\n{'='*70}")
        print(f"  EVALUATING: {name}")
        print(f"{'='*70}")

        data = load_dataset(path, args.max_samples)
        print(f"  Loaded {len(data)} samples")

        t = time.time()
        results = evaluate_detection(
            data, name,
            use_llm=args.use_llm_judge,
            llm_pipeline=llm_pipeline,
        )
        elapsed = time.time() - t
        print(f"  Completed in {elapsed:.2f}s")

        print_report(results)
        all_results[name] = {k: v for k, v in results.items() if k != "sample_analyses"}

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()