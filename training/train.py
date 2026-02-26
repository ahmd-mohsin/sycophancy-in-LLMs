import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model_loader import load_model, get_peft_model, save_model
from training.sft_trainer import run_sft
from training.grpo_trainer import run_grpo
from training.evaluator import evaluate_on_dataset, compare_phases
from training.data_splitter import split_all_datasets
from rewards.reward_aggregator import RewardWeights
from rewards.sft_data_builder import build_sft_dataset
from rewards.reward_dataset import build_reward_dataset
from rewards.grpo_data_builder import build_grpo_training_data


def parse_args():
    parser = argparse.ArgumentParser(description="Sycophancy RL Training Pipeline")

    parser.add_argument("--model", default="unsloth/llama-3.1-8b-instruct")
    parser.add_argument("--dataset-dir", default="dataset_generation/output")
    parser.add_argument("--output-dir", default="training/output")
    parser.add_argument("--judge-model", default="llama3.1:8b")
    parser.add_argument("--category", default="all", choices=["math", "physics", "political", "opinion", "all"])
    parser.add_argument("--max-seq-length", type=int, default=2048)

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.3)

    parser.add_argument("--sft-batch-size", type=int, default=2)
    parser.add_argument("--grpo-batch-size", type=int, default=1)
    parser.add_argument("--sft-lr", type=float, default=2e-4)
    parser.add_argument("--grpo-lr", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4)

    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--skip-grpo", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-data-prep", action="store_true")

    parser.add_argument("--sft-data-path", default=None)
    parser.add_argument("--grpo-data-path", default=None)
    parser.add_argument("--eval-dataset", default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    rewards_output = os.path.join(args.output_dir, "rewards")
    sft_ckpt = os.path.join(args.output_dir, "checkpoints", "sft")
    grpo_ckpt = os.path.join(args.output_dir, "checkpoints", "grpo")
    results_dir = os.path.join(args.output_dir, "results")
    splits_dir = os.path.join(args.output_dir, "splits")

    for d in [rewards_output, sft_ckpt, grpo_ckpt, results_dir, splits_dir]:
        os.makedirs(d, exist_ok=True)

    weights = RewardWeights(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
    )

    print(f"\n{'='*60}")
    print(f"  SYCOPHANCY REDUCTION TRAINING")
    print(f"{'='*60}")
    print(f"  Model      : {args.model}")
    print(f"  Dataset    : {args.dataset_dir}")
    print(f"  Weights    : α={weights.alpha} β={weights.beta} γ={weights.gamma} δ={weights.delta}")
    print(f"{'='*60}\n")

    sft_data_path = args.sft_data_path
    grpo_data_path = args.grpo_data_path

    if not args.skip_data_prep:
        print("── Data Preparation ─────────────────────────────────")

        print("Splitting datasets...")
        split_all_datasets(args.dataset_dir, splits_dir)

        print("\nBuilding SFT dataset...")
        sft_data_path = build_sft_dataset(args.dataset_dir, rewards_output)

        print("\nScoring reward groups...")
        reward_path = build_reward_dataset(
            dataset_dir=args.dataset_dir,
            output_dir=rewards_output,
            weights=weights,
            judge_model=args.judge_model,
        )

        print("\nBuilding GRPO training data...")
        grpo_data_path = build_grpo_training_data(reward_path, rewards_output)

    print("\n── Model Loading ────────────────────────────────────")
    model, tokenizer = load_model(args.model, max_seq_length=args.max_seq_length)
    model = get_peft_model(model)

    eval_dataset = args.eval_dataset
    if eval_dataset is None:
        import glob
        test_files = glob.glob(os.path.join(splits_dir, "*_test.json"))
        eval_dataset = test_files[0] if test_files else None

    if not args.skip_eval and eval_dataset:
        print("\n── Pre-training Evaluation ──────────────────────────")
        evaluate_on_dataset(model, tokenizer, eval_dataset, results_dir, phase="pre_training")

    if not args.skip_sft and sft_data_path:
        print("\n── Phase 1: SFT Warmup ──────────────────────────────")
        model, tokenizer, sft_stats = run_sft(
            model=model,
            tokenizer=tokenizer,
            sft_dataset_path=sft_data_path,
            output_dir=sft_ckpt,
            max_seq_length=args.max_seq_length,
            per_device_batch_size=args.sft_batch_size,
            learning_rate=args.sft_lr,
        )

        if not args.skip_eval and eval_dataset:
            print("\n── Post-SFT Evaluation ──────────────────────────────")
            evaluate_on_dataset(model, tokenizer, eval_dataset, results_dir, phase="post_sft")

    if not args.skip_grpo and grpo_data_path:
        print("\n── Phase 2: GRPO Training ───────────────────────────")
        model, tokenizer, grpo_stats = run_grpo(
            model=model,
            tokenizer=tokenizer,
            grpo_dataset_path=grpo_data_path,
            output_dir=grpo_ckpt,
            weights=weights,
            category=args.category if args.category != "all" else "opinion",
            judge_model=args.judge_model,
            max_seq_length=args.max_seq_length,
            per_device_batch_size=args.grpo_batch_size,
            learning_rate=args.grpo_lr,
            num_generations=args.num_generations,
        )

        if not args.skip_eval and eval_dataset:
            print("\n── Post-GRPO Evaluation ─────────────────────────────")
            evaluate_on_dataset(model, tokenizer, eval_dataset, results_dir, phase="post_grpo")

    print("\n── Final Comparison ─────────────────────────────────")
    comparison = compare_phases(results_dir)
    comparison_path = os.path.join(results_dir, "phase_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    for phase, metrics in comparison.items():
        print(f"\n  {phase}:")
        for k, v in metrics.items():
            print(f"    {k:<25} {v:.4f}")

    save_model(model, tokenizer, os.path.join(args.output_dir, "final_model"))
    print(f"\nDone. Final model → {args.output_dir}/final_model")


if __name__ == "__main__":
    main()