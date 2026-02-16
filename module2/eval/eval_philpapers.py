"""
Evaluate Module 2 on EleutherAI/sycophancy - PhilPapers 2020 subset only.

Usage:
    python -m module2.eval.eval_philpapers --model-dir gpt-oss-20b/ --max-samples 50
"""
import argparse
import json
from datasets import load_dataset
from module2.eval.eval_module2 import start_gpt_oss, stop_gpt_oss, evaluate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--output", default="eval_philpapers.json")
    args = parser.parse_args()

    start_gpt_oss(args.model_dir, args.port)
    try:
        ds = load_dataset("EleutherAI/sycophancy", "sycophancy_on_philpapers2020", split="validation")
        print(f"Loaded PhilPapers: {len(ds)} samples")

        max_s = args.max_samples if args.max_samples > 0 else None
        result = evaluate_dataset("philpapers", ds, max_s, args.port)

        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved to {args.output}")
    finally:
        stop_gpt_oss()


if __name__ == "__main__":
    main()