import json
import os
import sys
import csv
import time
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


def load_lines(filepath: str) -> list[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_pushback_csv(filepath: str) -> list[list[str]]:
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            cleaned = [cell.strip() for cell in row if cell.strip()]
            if cleaned:
                rows.append(cleaned)
    return rows


def build_judge_prompt(text: str) -> str:
    if SYCOPHANCY_JUDGE_TEMPLATE is not None:
        return SYCOPHANCY_JUDGE_TEMPLATE.format(text=text)
    return f"[Judge template unavailable] Text: {text}"


def evaluate_false_presuppositions(
    presuppositions: list[str],
    corrections: list[str],
    pushback_rows: list[list[str]],
    use_llm: bool = False,
    llm_pipeline=None,
    max_samples: int = None,
) -> dict:
    detector = BiasDetector(llm_pipeline=llm_pipeline)
    module = Module1(use_llm=False)

    n = min(len(presuppositions), len(corrections), len(pushback_rows))
    if max_samples:
        n = min(n, max_samples)

    results = {
        "benchmark": "SYCON-Bench False Presuppositions",
        "total_questions": n,
        "total_pushback_turns": 0,
        "detection_by_turn": defaultdict(lambda: {"correct": 0, "total": 0}),
        "overall": {
            "total_turns": 0,
            "correctly_detected": 0,
            "missed": 0,
            "explicit_caught": 0,
            "implicit_caught": 0,
            "llm_judge_caught": 0,
        },
        "risk_levels": defaultdict(int),
        "classification": defaultdict(int),
        "per_question_results": [],
    }

    pbar = tqdm(range(n), desc="  FalsePresup", unit="q")
    for i in pbar:
        presupposition = presuppositions[i]
        correction = corrections[i]
        pushback_turns = pushback_rows[i] if i < len(pushback_rows) else []

        question_record = {
            "question_id": i,
            "false_presupposition": presupposition,
            "correction": correction,
            "turns": [],
        }

        for turn_idx, user_challenge in enumerate(pushback_turns):
            if not user_challenge.strip():
                continue

            bias_result = detector.detect_bias(user_challenge, use_llm=use_llm)

            has_explicit = bias_result["explicit_bias"]["has_bias"]
            has_implicit = bias_result["implicit_bias"]["has_implicit_bias"]
            has_any = bias_result["has_bias"]
            risk = bias_result["sycophancy_risk"]

            llm_caught = False
            llm_judge_output = None
            if use_llm and "llm_judge" in bias_result:
                llm_judge_output = bias_result["llm_judge"]
                if llm_judge_output.get("sycophancy_detected"):
                    llm_caught = True

            detected = has_any or llm_caught

            m1_result = module.process(
                question=presupposition,
                assistant_answer=correction,
                user_challenge=user_challenge,
            )
            qtype = m1_result["question_type"]

            turn_record = {
                "turn": turn_idx + 1,
                "user_challenge": user_challenge,
                "expected_bias": True,
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
                turn_record["llm_judge"] = {
                    "prompt_sent_to_judge": build_judge_prompt(user_challenge),
                    "judge_raw_output": llm_judge_output.get("raw_output", ""),
                    "sycophancy_detected": llm_judge_output.get("sycophancy_detected", False),
                    "risk_level": llm_judge_output.get("risk_level", "unknown"),
                    "reason": llm_judge_output.get("reason", ""),
                }

            question_record["turns"].append(turn_record)

            results["overall"]["total_turns"] += 1
            if detected:
                results["overall"]["correctly_detected"] += 1
            else:
                results["overall"]["missed"] += 1

            if has_explicit:
                results["overall"]["explicit_caught"] += 1
            if has_implicit:
                results["overall"]["implicit_caught"] += 1
            if llm_caught:
                results["overall"]["llm_judge_caught"] += 1

            results["detection_by_turn"][turn_idx + 1]["total"] += 1
            if detected:
                results["detection_by_turn"][turn_idx + 1]["correct"] += 1

            results["risk_levels"][risk] += 1
            results["classification"][qtype] += 1

        results["per_question_results"].append(question_record)
        results["total_pushback_turns"] = results["overall"]["total_turns"]

        total_so_far = results["overall"]["total_turns"]
        correct_so_far = results["overall"]["correctly_detected"]
        acc = correct_so_far / total_so_far * 100 if total_so_far > 0 else 0
        pbar.set_postfix({"det%": f"{acc:.1f}", "turns": total_so_far})

    total = results["overall"]["total_turns"]
    if total > 0:
        results["metrics"] = {
            "overall_detection_rate": results["overall"]["correctly_detected"] / total * 100,
            "miss_rate": results["overall"]["missed"] / total * 100,
            "per_turn_detection_rate": {},
        }
        for turn_num, data in sorted(results["detection_by_turn"].items()):
            if data["total"] > 0:
                results["metrics"]["per_turn_detection_rate"][f"turn_{turn_num}"] = data["correct"] / data["total"] * 100
    else:
        results["metrics"] = {"overall_detection_rate": 0, "miss_rate": 100}

    return results


def print_report(results: dict):
    total = results["overall"]["total_turns"]
    if total == 0:
        print("  No turns evaluated.")
        return

    correct = results["overall"]["correctly_detected"]

    print()
    print("=" * 70)
    print(f"  SYCON-BENCH FALSE PRESUPPOSITIONS: DETECTION REPORT")
    print("=" * 70)
    print(f"\n  Questions: {results['total_questions']}")
    print(f"  Total pushback turns: {total}")
    print(f"  (Every pushback turn IS sycophancy-inducing by design)")

    print(f"\n  --- Overall Detection ---")
    print(f"  Detected (TP): {correct}/{total} ({results['metrics']['overall_detection_rate']:.1f}%)")
    print(f"  Missed (FN):   {results['overall']['missed']}/{total} ({results['metrics']['miss_rate']:.1f}%)")

    print(f"\n  --- Detection Methods ---")
    o = results["overall"]
    print(f"  Explicit keywords:  {o['explicit_caught']}/{total} ({o['explicit_caught']/total*100:.1f}%)")
    print(f"  Implicit patterns:  {o['implicit_caught']}/{total} ({o['implicit_caught']/total*100:.1f}%)")
    if o["llm_judge_caught"] > 0:
        print(f"  LLM judge:          {o['llm_judge_caught']}/{total} ({o['llm_judge_caught']/total*100:.1f}%)")

    print(f"\n  --- Detection by Pushback Turn ---")
    for turn_num in sorted(results["detection_by_turn"].keys()):
        data = results["detection_by_turn"][turn_num]
        acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        bar = "█" * int(acc / 2.5)
        print(f"  Turn {turn_num}: {data['correct']:>3}/{data['total']} ({acc:5.1f}%) {bar}")

    print(f"\n  --- Risk Levels ---")
    for level in ["high", "medium", "low", "none"]:
        count = results["risk_levels"].get(level, 0)
        bar = "█" * int(count / total * 40)
        print(f"  {level.upper():8s} {count:>5}/{total} ({count/total*100:5.1f}%) {bar}")

    print(f"\n  --- Classification ---")
    for qtype in ["subjective", "factual", "time_sensitive"]:
        count = results["classification"].get(qtype, 0)
        print(f"  {qtype:15s}: {count}/{total} ({count/total*100:.1f}%)")

    missed_examples = []
    for q in results["per_question_results"]:
        for t in q["turns"]:
            if not t["final_verdict"]["correct"]:
                missed_examples.append({
                    "question_id": q["question_id"],
                    "turn": t["turn"],
                    "challenge": t["user_challenge"][:150],
                    "presupposition": q["false_presupposition"][:100],
                })
                if len(missed_examples) >= 5:
                    break
        if len(missed_examples) >= 5:
            break

    if missed_examples:
        print(f"\n  --- Missed Examples (first {len(missed_examples)}) ---")
        for ex in missed_examples:
            print(f"\n  Q{ex['question_id']} Turn {ex['turn']}:")
            print(f"    Presupposition: {ex['presupposition']}...")
            print(f"    Challenge: {ex['challenge']}...")

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Module 1 on SYCON-Bench False Presuppositions")
    parser.add_argument("--presuppositions", type=str,
                        default="/home/ahmed/sycophancy-in-LLMs/module1/sycon_bench/false_presuppositions/presuppositions.txt")
    parser.add_argument("--corrections", type=str,
                        default="/home/ahmed/sycophancy-in-LLMs/module1/sycon_bench/false_presuppositions/corrections.txt")
    parser.add_argument("--pushback", type=str,
                        default="/home/ahmed/sycophancy-in-LLMs/module1/sycon_bench/false_presuppositions/pushback.csv")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max questions to evaluate (default: all)")
    parser.add_argument("--use-llm-judge", action="store_true",
                        help="Use LLM as judge alongside regex")
    parser.add_argument("--output", type=str, default="sycon_false_presup_results.json")
    args = parser.parse_args()

    for fpath, fname in [(args.presuppositions, "presuppositions"), (args.corrections, "corrections"), (args.pushback, "pushback")]:
        if not os.path.exists(fpath):
            print(f"  ERROR: {fname} not found at {fpath}")
            print(f"  Update --{fname} path or place file at expected location.")
            sys.exit(1)

    presuppositions = load_lines(args.presuppositions)
    corrections = load_lines(args.corrections)
    pushback_rows = load_pushback_csv(args.pushback)

    print(f"  Loaded: {len(presuppositions)} presuppositions, {len(corrections)} corrections, {len(pushback_rows)} pushback rows")

    llm_pipeline = None
    if args.use_llm_judge:
        print("  Loading LLM judge...")
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
                        model_name, torch_dtype=torch.float16,
                        device_map="auto", low_cpu_mem_usage=True,
                    )
                except RuntimeError:
                    print("  float16 OOM, trying 4-bit...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
                        device_map="auto", low_cpu_mem_usage=True,
                    )
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name)

            pipe = hf_pipeline(
                "text-generation", model=model, tokenizer=tokenizer,
                max_new_tokens=200, do_sample=False, return_full_text=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            llm_pipeline = HuggingFacePipeline(pipeline=pipe)
            print(f"  Loaded {model_name} on {next(model.parameters()).device}")
        except Exception as e:
            print(f"  Failed: {e}")
            print("  Continuing with regex only.")

    total_turns = sum(len(row) for row in pushback_rows[:args.max_samples or len(pushback_rows)])
    print(f"\n  Evaluating {min(len(presuppositions), args.max_samples or 999)} questions, ~{total_turns} pushback turns\n")

    t = time.time()
    results = evaluate_false_presuppositions(
        presuppositions, corrections, pushback_rows,
        use_llm=args.use_llm_judge,
        llm_pipeline=llm_pipeline,
        max_samples=args.max_samples,
    )
    elapsed = time.time() - t
    print(f"\n  Completed in {elapsed:.1f}s")

    print_report(results)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()