"""
Evaluate Module 2 on EleutherAI/sycophancy - NLP Survey subset only.

Usage:
    python -m module2.eval.eval_nlp_survey --model-dir gpt-oss-20b/ --max-samples 50
"""
import argparse
import json
from datasets import load_dataset
from module2.eval.eval_module2 import load_model, evaluate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--output", default="eval_nlp_survey.json")
    args = parser.parse_args()

    load_model(args.model_dir)

    ds = load_dataset("EleutherAI/sycophancy", "sycophancy_on_nlp_survey", split="validation")
    print(f"Loaded NLP Survey: {len(ds)} samples")

    max_s = args.max_samples if args.max_samples > 0 else None
    result = evaluate_dataset("nlp_survey", ds, max_s)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()