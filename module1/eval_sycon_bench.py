import json
import os
import sys
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
from sycon_bench_debate_data import DEBATE_TOPICS, PRESSURE_TEMPLATES, generate_counter_stance


def build_judge_prompt(text: str) -> str:
    if SYCOPHANCY_JUDGE_TEMPLATE is not None:
        return SYCOPHANCY_JUDGE_TEMPLATE.format(text=text)
    return f"[Judge template unavailable] Text: {text}"


def evaluate_sycon_bench(
    topics: list[dict],
    use_llm: bool = False,
    llm_pipeline=None,
) -> dict:
    detector = BiasDetector(llm_pipeline=llm_pipeline)
    module = Module1(use_llm=False)

    results = {
        "benchmark": "SYCON-Bench Debate",
        "total_topics": len(topics),
        "turns_per_topic": len(PRESSURE_TEMPLATES),
        "total_turns": len(topics) * len(PRESSURE_TEMPLATES),
        "detection_by_turn": {t["turn"]: {"correct": 0, "total": 0} for t in PRESSURE_TEMPLATES},
        "detection_by_level": defaultdict(lambda: {"correct": 0, "total": 0}),
        "detection_by_category": defaultdict(lambda: {"correct": 0, "total": 0}),
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
        "per_topic_results": [],
    }

    pbar = tqdm(topics, desc="  SYCON-Bench", unit="topic")
    for topic in pbar:
        counter_stance = generate_counter_stance(topic["argument"])

        topic_record = {
            "topic_id": topic["id"],
            "category": topic["category"],
            "question": topic["question"],
            "model_stance": topic["argument"],
            "counter_stance": counter_stance,
            "turns": [],
        }

        conversation_history = []

        conversation_history.append((
            topic["question"],
            f"I believe that {topic['argument']}",
        ))

        for pressure in PRESSURE_TEMPLATES:
            user_challenge = pressure["template"].format(counter_stance=counter_stance)
            expected_bias = pressure["expected_bias"]

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
                question=topic["question"],
                assistant_answer=f"I believe that {topic['argument']}",
                user_challenge=user_challenge,
                conversation_history=conversation_history,
            )
            qtype = m1_result["question_type"]

            if expected_bias:
                correct = detected
            else:
                correct = True

            turn_record = {
                "turn": pressure["turn"],
                "level": pressure["level"],
                "user_challenge": user_challenge,
                "expected_bias": expected_bias,
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
                    "correct": correct,
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

            topic_record["turns"].append(turn_record)

            results["overall"]["total_turns"] += 1
            if correct:
                results["overall"]["correctly_detected"] += 1
            else:
                results["overall"]["missed"] += 1

            if has_explicit:
                results["overall"]["explicit_caught"] += 1
            if has_implicit:
                results["overall"]["implicit_caught"] += 1
            if llm_caught:
                results["overall"]["llm_judge_caught"] += 1

            results["detection_by_turn"][pressure["turn"]]["total"] += 1
            if correct:
                results["detection_by_turn"][pressure["turn"]]["correct"] += 1

            results["detection_by_level"][pressure["level"]]["total"] += 1
            if correct:
                results["detection_by_level"][pressure["level"]]["correct"] += 1

            results["detection_by_category"][topic["category"]]["total"] += 1
            if correct:
                results["detection_by_category"][topic["category"]]["correct"] += 1

            results["risk_levels"][risk] += 1
            results["classification"][qtype] += 1

            conversation_history.append((user_challenge, f"I maintain that {topic['argument']}"))

        results["per_topic_results"].append(topic_record)

        total_so_far = results["overall"]["total_turns"]
        correct_so_far = results["overall"]["correctly_detected"]
        acc = correct_so_far / total_so_far * 100 if total_so_far > 0 else 0
        pbar.set_postfix({"acc%": f"{acc:.1f}", "turns": total_so_far})

    total = results["overall"]["total_turns"]
    results["metrics"] = {
        "overall_accuracy": results["overall"]["correctly_detected"] / total * 100 if total > 0 else 0,
        "miss_rate": results["overall"]["missed"] / total * 100 if total > 0 else 0,
        "per_turn_accuracy": {},
        "per_level_accuracy": {},
        "per_category_accuracy": {},
    }

    for turn_num, data in results["detection_by_turn"].items():
        if data["total"] > 0:
            results["metrics"]["per_turn_accuracy"][f"turn_{turn_num}"] = data["correct"] / data["total"] * 100

    for level, data in results["detection_by_level"].items():
        if data["total"] > 0:
            results["metrics"]["per_level_accuracy"][level] = data["correct"] / data["total"] * 100

    for cat, data in results["detection_by_category"].items():
        if data["total"] > 0:
            results["metrics"]["per_category_accuracy"][cat] = data["correct"] / data["total"] * 100

    return results


def print_report(results: dict):
    total = results["overall"]["total_turns"]
    correct = results["overall"]["correctly_detected"]

    print()
    print("=" * 70)
    print(f"  SYCON-BENCH DEBATE: SYCOPHANCY DETECTION REPORT")
    print("=" * 70)
    print(f"\n  Topics: {results['total_topics']}")
    print(f"  Turns per topic: {results['turns_per_topic']}")
    print(f"  Total turns evaluated: {total}")

    print(f"\n  --- Overall Accuracy ---")
    print(f"  Correct: {correct}/{total} ({results['metrics']['overall_accuracy']:.1f}%)")
    print(f"  Missed:  {results['overall']['missed']}/{total} ({results['metrics']['miss_rate']:.1f}%)")

    print(f"\n  --- Detection Methods ---")
    o = results["overall"]
    print(f"  Explicit keywords:  {o['explicit_caught']}/{total} ({o['explicit_caught']/total*100:.1f}%)")
    print(f"  Implicit patterns:  {o['implicit_caught']}/{total} ({o['implicit_caught']/total*100:.1f}%)")
    if o["llm_judge_caught"] > 0:
        print(f"  LLM judge:          {o['llm_judge_caught']}/{total} ({o['llm_judge_caught']/total*100:.1f}%)")

    print(f"\n  --- Accuracy by Pressure Level ---")
    for turn_num in sorted(results["detection_by_turn"].keys()):
        data = results["detection_by_turn"][turn_num]
        level_name = PRESSURE_TEMPLATES[turn_num - 1]["level"]
        acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        bar = "█" * int(acc / 2.5)
        print(f"  Turn {turn_num} ({level_name:25s}): {data['correct']:>3}/{data['total']} ({acc:5.1f}%) {bar}")

    print(f"\n  --- Accuracy by Category ---")
    for cat, data in sorted(results["detection_by_category"].items()):
        acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        bar = "█" * int(acc / 2.5)
        print(f"  {cat:25s}: {data['correct']:>3}/{data['total']} ({acc:5.1f}%) {bar}")

    print(f"\n  --- Risk Level Distribution ---")
    for level in ["high", "medium", "low", "none"]:
        count = results["risk_levels"].get(level, 0)
        bar = "█" * int(count / total * 40)
        print(f"  {level.upper():8s} {count:>5}/{total} ({count/total*100:5.1f}%) {bar}")

    print(f"\n  --- Classification ---")
    for qtype in ["subjective", "factual", "time_sensitive"]:
        count = results["classification"].get(qtype, 0)
        print(f"  {qtype:15s}: {count}/{total} ({count/total*100:.1f}%)")

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Module 1 on SYCON-Bench Debate")
    parser.add_argument("--max-topics", type=int, default=100,
                        help="Max debate topics to evaluate (default: 100 = all)")
    parser.add_argument("--use-llm-judge", action="store_true",
                        help="Use LLM as judge alongside regex")
    parser.add_argument("--output", type=str, default="sycon_bench_results.json")
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

    topics = DEBATE_TOPICS[:args.max_topics]
    print(f"\n  Evaluating {len(topics)} debate topics x {len(PRESSURE_TEMPLATES)} turns = {len(topics) * len(PRESSURE_TEMPLATES)} total turns\n")

    t = time.time()
    results = evaluate_sycon_bench(topics, use_llm=args.use_llm_judge, llm_pipeline=llm_pipeline)
    elapsed = time.time() - t
    print(f"\n  Completed in {elapsed:.1f}s")

    print_report(results)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved results to {args.output}")
    print(f"Contains {len(results['per_topic_results'])} topics x {len(PRESSURE_TEMPLATES)} turns with full detection details")


if __name__ == "__main__":
    main()