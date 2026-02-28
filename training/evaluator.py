import json
import os
import torch
from datetime import datetime

from metrics.dataset_loader import load_sample_groups
from metrics.sycophancy_metrics import compute_all, aggregate_metrics, SampleGroup
from rewards.grpo_data_builder import build_grpo_prompt


def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    batch_size: int = 4,
) -> list[str]:
    model.eval()
    responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"].shape[1]
            decoded = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            responses.append(decoded.strip())
    model.train()
    return responses


def evaluate_on_dataset(
    model,
    tokenizer,
    dataset_path: str,
    output_dir: str = "training/results",
    phase: str = "post_sft",
    max_new_tokens: int = 256,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    groups = load_sample_groups(dataset_path)
    print(f"\nEvaluating {len(groups)} groups ({phase})...")

    metric_results = []
    for i, group in enumerate(groups):
        evaluated_group = _evaluate_group(model, tokenizer, group, max_new_tokens)
        result = compute_all(evaluated_group)
        metric_results.append(result)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(groups)} groups evaluated")

    aggregated = aggregate_metrics(metric_results)

    output = {
        "phase": phase,
        "dataset": dataset_path,
        "n_groups": len(groups),
        "evaluated_at": datetime.now().isoformat(),
        "metrics": aggregated,
    }

    out_path = os.path.join(
        output_dir,
        f"eval_{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    _print_metrics(aggregated, phase)
    return output


def _evaluate_group(
    model,
    tokenizer,
    group: SampleGroup,
    max_new_tokens: int,
) -> SampleGroup:
    updated_pressured = []
    for sample in group.pressured:
        messages = build_grpo_prompt(
            pressure_prompt=sample["components"]["pressure_prompt"],
            context=sample["components"]["context"],
            question=sample["components"]["question"],
        )
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        responses = generate_responses(
            model, tokenizer, [prompt], max_new_tokens=max_new_tokens
        )
        updated = dict(sample)
        updated["response"] = responses[0]
        updated_pressured.append(updated)

    return SampleGroup(
        topic=group.topic,
        question=group.question,
        opinion=group.opinion,
        baseline_orig=group.baseline_orig,
        baseline_opp=group.baseline_opp,
        pressured=updated_pressured,
    )


def compare_phases(results_dir: str) -> dict:
    import glob
    files = sorted(glob.glob(os.path.join(results_dir, "eval_*.json")))
    comparison = {}
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        comparison[data["phase"]] = data["metrics"]
    return comparison


def _print_metrics(metrics: dict, phase: str):
    print(f"\n{'='*50}")
    print(f"  EVALUATION — {phase.upper()}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<25} {v:.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    import argparse
    import glob
    from unsloth import FastLanguageModel

    parser = argparse.ArgumentParser(description="Evaluate a trained model on sycophancy metrics")
    parser.add_argument("--model", required=True, help="Path to model (local dir or HF id)")
    parser.add_argument("--dataset-dir", default="dataset_generation/output", help="Dataset directory")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--phase", default="post_grpo", help="Label for this eval phase")
    parser.add_argument("--results-dir", default="training/output/results")
    args = parser.parse_args()

    # Find test split
    splits_dir = "training/splits"
    test_files = sorted(glob.glob(os.path.join(splits_dir, "*_test.json")))
    if not test_files:
        # Fall back to first file in dataset dir
        test_files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.json")))
    if not test_files:
        print("ERROR: No test dataset found. Run data prep first.")
        exit(1)
    dataset_path = test_files[0]
    print(f"Evaluating on: {dataset_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        device_map={"": 0},
    )
    model.eval()

    result = evaluate_on_dataset(
        model, tokenizer,
        dataset_path=dataset_path,
        output_dir=args.results_dir,
        phase=args.phase,
        max_new_tokens=256,
    )

    print("\nFull comparison across all eval phases:")
    comparison = compare_phases(args.results_dir)
    for phase, metrics in comparison.items():
        print(f"\n  {phase}:")
        for k, v in metrics.items():
            print(f"    {k:<25} {v:.4f}")