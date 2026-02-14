import json
import os
import sys
import time
import re
import warnings
import logging
import argparse
from collections import defaultdict
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(__file__))

from bias_detector import BiasDetector, SYCOPHANCY_JUDGE_TEMPLATE
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


def build_judge_prompt(biography: str) -> str:
    if SYCOPHANCY_JUDGE_TEMPLATE is not None:
        return SYCOPHANCY_JUDGE_TEMPLATE.format(text=biography)
    return f"[Judge template unavailable] Text: {biography}"


def evaluate_detection(
    data: list[dict],
    dataset_name: str,
    use_llm: bool = False,
    llm_pipeline=None,
) -> dict:
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
        "per_sample_results": [],
    }

    pbar = tqdm(enumerate(data), total=len(data), desc=f"  {dataset_name}", unit="sample")
    for i, sample in pbar:
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
        llm_judge_output = None
        if use_llm and "llm_judge" in bias_result:
            llm_judge_output = bias_result["llm_judge"]
            if llm_judge_output.get("sycophancy_detected"):
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

        if bias_result["implicit_bias"]["opinion_steering"]:
            results["implicit_breakdown"]["opinion_steering"] += 1
        if bias_result["implicit_bias"]["authority_pressure"]:
            results["implicit_breakdown"]["authority_pressure"] += 1
        if bias_result["implicit_bias"]["emotional_framing"]:
            results["implicit_breakdown"]["emotional_framing"] += 1

        m1_result = module.process(
            question=actual_question,
            assistant_answer=f"I believe the best answer is {sycophantic_answer}",
            user_challenge=f"I think the answer should be {independent_answer} instead.",
        )
        qtype = m1_result["question_type"]
        results["classification"][qtype] += 1

        sample_record = {
            "sample_id": i,
            "biography": biography,
            "question": actual_question,
            "ground_truth": {
                "sycophantic_answer": sycophantic_answer,
                "independent_answer": independent_answer,
                "is_sycophancy_inducing": True,
            },
            "regex_detection": {
                "explicit_bias": {
                    "detected": has_explicit,
                    "keywords": bias_result["explicit_bias"]["keywords_found"],
                },
                "implicit_bias": {
                    "detected": has_implicit,
                    "opinion_steering": [m["matched_text"] for m in bias_result["implicit_bias"]["opinion_steering"]],
                    "authority_pressure": [m["matched_text"] for m in bias_result["implicit_bias"]["authority_pressure"]],
                    "emotional_framing": [m["matched_text"] for m in bias_result["implicit_bias"]["emotional_framing"]],
                },
            },
            "llm_judge": None,
            "final_verdict": {
                "sycophancy_detected": detected,
                "risk_level": risk,
                "question_type": qtype,
                "correct": detected,
            },
        }

        if use_llm and llm_judge_output is not None:
            sample_record["llm_judge"] = {
                "prompt_sent_to_judge": build_judge_prompt(biography),
                "judge_raw_output": llm_judge_output.get("raw_output", ""),
                "sycophancy_detected": llm_judge_output.get("sycophancy_detected", False),
                "risk_level": llm_judge_output.get("risk_level", "unknown"),
                "opinion_steering": llm_judge_output.get("opinion_steering", False),
                "authority_pressure": llm_judge_output.get("authority_pressure", False),
                "identity_framing": llm_judge_output.get("identity_framing", False),
                "emotional_pressure": llm_judge_output.get("emotional_pressure", False),
                "explicit_manipulation": llm_judge_output.get("explicit_manipulation", False),
                "biographical_priming": llm_judge_output.get("biographical_priming", False),
                "reason": llm_judge_output.get("reason", ""),
            }

        results["per_sample_results"].append(sample_record)

        det_rate = results["detection"]["total_sycophancy_detected"] / (i + 1) * 100
        pbar.set_postfix({"det%": f"{det_rate:.1f}", "TP": results["detection"]["total_sycophancy_detected"]})

    total = results["total_samples"]
    tp = results["ground_truth_comparison"]["true_positives"]
    fn = results["ground_truth_comparison"]["false_negatives"]

    results["metrics"] = {
        "detection_rate": tp / total * 100 if total > 0 else 0,
        "miss_rate": fn / total * 100 if total > 0 else 0,
        "explicit_only_rate": results["detection"]["explicit_bias_caught"] / total * 100 if total > 0 else 0,
        "implicit_detection_rate": results["detection"]["implicit_bias_caught"] / total * 100 if total > 0 else 0,
    }
    if use_llm:
        results["metrics"]["llm_judge_rate"] = results["detection"]["llm_judge_caught"] / total * 100

    return results


def print_report(results: dict):
    r = results
    total = r["total_samples"]

    print()
    print("=" * 70)
    print(f"  SYCOPHANCY DETECTION REPORT: {r['dataset']}")
    print("=" * 70)
    print(f"\n  Total samples: {total}")
    print(f"  (Every sample IS sycophancy-inducing by design)")

    print(f"\n  --- Detection Results (vs Ground Truth) ---")
    tp = r["ground_truth_comparison"]["true_positives"]
    fn = r["ground_truth_comparison"]["false_negatives"]
    print(f"  Correctly detected (TP):  {tp}/{total} ({r['metrics']['detection_rate']:.1f}%)")
    print(f"  Missed (FN):              {fn}/{total} ({r['metrics']['miss_rate']:.1f}%)")

    print(f"\n  --- Detection Breakdown ---")
    d = r["detection"]
    print(f"  Explicit keywords:   {d['explicit_bias_caught']}/{total} ({d['explicit_bias_caught']/total*100:.1f}%)")
    print(f"  Implicit patterns:   {d['implicit_bias_caught']}/{total} ({d['implicit_bias_caught']/total*100:.1f}%)")
    print(f"  Regex combined:      {d['any_bias_caught']}/{total} ({d['any_bias_caught']/total*100:.1f}%)")
    if d["llm_judge_caught"] > 0:
        print(f"  LLM judge:           {d['llm_judge_caught']}/{total} ({d['llm_judge_caught']/total*100:.1f}%)")
    print(f"  TOTAL detected:      {d['total_sycophancy_detected']}/{total} ({d['total_sycophancy_detected']/total*100:.1f}%)")

    print(f"\n  --- Risk Levels ---")
    for level in ["high", "medium", "low", "none"]:
        count = r["risk_levels"].get(level, 0)
        bar = "█" * int(count / total * 40)
        print(f"  {level.upper():8s} {count:>5}/{total} ({count/total*100:5.1f}%) {bar}")

    print(f"\n  --- Implicit Patterns ---")
    ib = r["implicit_breakdown"]
    print(f"  Opinion steering:   {ib['opinion_steering']}/{total} ({ib['opinion_steering']/total*100:.1f}%)")
    print(f"  Authority pressure: {ib['authority_pressure']}/{total} ({ib['authority_pressure']/total*100:.1f}%)")
    print(f"  Emotional framing:  {ib['emotional_framing']}/{total} ({ib['emotional_framing']/total*100:.1f}%)")

    print(f"\n  --- Classification ---")
    c = r["classification"]
    for qtype in ["subjective", "factual", "time_sensitive"]:
        count = c.get(qtype, 0)
        print(f"  {qtype:15s}: {count}/{total} ({count/total*100:.1f}%)")

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate sycophancy detection on Anthropic dataset")
    parser.add_argument("--nlp", type=str,
                        default="/home/ahmed/sycophancy-in-LLMs/module1/sycophancy_on_nlp_survey.jsonl")
    parser.add_argument("--political", type=str,
                        default="/home/ahmed/sycophancy-in-LLMs/module1/sycophancy_on_political_typology_quiz.jsonl")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Max samples per dataset (default: 100)")
    parser.add_argument("--use-llm-judge", action="store_true",
                        help="Use LLM as judge alongside regex")
    parser.add_argument("--output", type=str, default="sycophancy_detection_results.json")
    args = parser.parse_args()

    llm_pipeline = None
    if args.use_llm_judge:
        print("Loading LLM judge...")
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from transformers import pipeline as hf_pipeline
            from lc_compat import HuggingFacePipeline

            model_name = os.getenv("HF_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if torch.cuda.is_available():
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                    )
                except RuntimeError:
                    print("  float16 OOM, trying 4-bit...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                        ),
                        device_map="auto",
                        low_cpu_mem_usage=True,
                    )
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name)

            pipe = hf_pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=200,
                do_sample=False,
                return_full_text=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            llm_pipeline = HuggingFacePipeline(pipeline=pipe)

            device = next(model.parameters()).device
            print(f"  Loaded {model_name} on {device}")
        except Exception as e:
            print(f"  Failed: {e}")
            print("  Continuing with regex only.")

    all_results = {}

    datasets = []
    if os.path.exists(args.nlp):
        datasets.append(("NLP_Survey", args.nlp))
    else:
        print(f"  NLP dataset not found: {args.nlp}")
    if os.path.exists(args.political):
        datasets.append(("Political_Typology", args.political))
    else:
        print(f"  Political dataset not found: {args.political}")

    if not datasets:
        print("No datasets found.")
        sys.exit(1)

    for name, path in datasets:
        print(f"\n  Loading {name}...")
        data = load_dataset(path, args.max_samples)
        print(f"  {len(data)} samples loaded\n")

        t = time.time()
        results = evaluate_detection(
            data, name,
            use_llm=args.use_llm_judge,
            llm_pipeline=llm_pipeline,
        )
        elapsed = time.time() - t
        print(f"\n  Completed in {elapsed:.1f}s")

        print_report(results)
        all_results[name] = results

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    total_samples = sum(r["total_samples"] for r in all_results.values())
    print(f"\nSaved {total_samples} sample results to {args.output}")
    print(f"Each sample contains: biography, question, regex detection, LLM judge prompt+verdict, final verdict")


if __name__ == "__main__":
    main()