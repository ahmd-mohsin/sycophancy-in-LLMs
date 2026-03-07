import json
import os
import glob
import torch
from datetime import datetime

from metrics.dataset_loader import load_all_groups
from metrics.sycophancy_metrics import compute_all, aggregate_metrics, SampleGroup
from rewards.grpo_data_builder import build_grpo_prompt


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------

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
        batch = prompts[i : i + batch_size]
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


# ---------------------------------------------------------------------------
# Group-level evaluation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dataset-level evaluation
# ---------------------------------------------------------------------------

def evaluate_on_dataset(
    model,
    tokenizer,
    dataset_path: str,
    output_dir: str = "training/results",
    phase: str = "post_grpo",
    max_new_tokens: int = 256,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    # FIX: use load_all_groups (handles checkpoint format) instead of
    # load_sample_groups (expects legacy *_with_baselines.json format)
    groups = load_all_groups(dataset_path)
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


# ---------------------------------------------------------------------------
# Cross-phase comparison
# ---------------------------------------------------------------------------

def compare_phases(results_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(results_dir, "eval_*.json")))
    comparison = {}
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        comparison[data["phase"]] = data["metrics"]
    return comparison


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _print_metrics(metrics: dict, phase: str):
    print(f"\n{'='*50}")
    print(f"  EVALUATION — {phase.upper()}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<25} {v:.4f}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from unsloth import FastLanguageModel
    from metrics.dataset_loader import load_sample_groups
    from metrics.sycophancy_metrics import compute_all, aggregate_metrics

    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on sycophancy metrics"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model (local dir or HF id). "
             "Use training/output/final_model for the latest v3 run.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="training/output/rewards",
        help="Directory containing *_checkpoint.json or split JSON files.",
    )
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument(
        "--phase",
        default="post_grpo_v3",
        help="Label for this eval phase (used in output filename).",
    )
    parser.add_argument(
        "--results-dir",
        default="training/output/results",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Which split to evaluate: test | val | all. "
             "Looks for files matching *<split>*.json first, "
             "then falls back to *_checkpoint.json.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # FIX 1: Correct glob — try split-specific files first, then
    # checkpoint files. Original code only looked for *_with_baselines.json
    # which never exists in this project.
    # ------------------------------------------------------------------
    def find_dataset_files(dataset_dir: str, split: str) -> list[str]:
        # Priority 1: explicit split files (e.g. grpo_training_test.json)
        split_files = sorted(glob.glob(
            os.path.join(dataset_dir, f"*{split}*.json")
        ))
        if split_files:
            return split_files

        # Priority 2: checkpoint files (raw generation output)
        checkpoint_files = sorted(glob.glob(
            os.path.join(dataset_dir, "*_checkpoint.json")
        ))
        if checkpoint_files:
            print(
                f"[WARN] No *{split}*.json found — "
                f"falling back to *_checkpoint.json files."
            )
            return checkpoint_files

        # Priority 3: any JSON in the directory
        all_json = sorted(glob.glob(os.path.join(dataset_dir, "*.json")))
        # Exclude results files to avoid loading previously saved eval output
        all_json = [f for f in all_json if "eval_" not in os.path.basename(f)]
        return all_json

    dataset_files = find_dataset_files(args.dataset_dir, args.split)

    if not dataset_files:
        print(
            f"ERROR: No dataset files found in {args.dataset_dir}.\n"
            f"Expected one of:\n"
            f"  *{args.split}*.json\n"
            f"  *_checkpoint.json\n"
            f"Run dataset generation or data splitting first."
        )
        exit(1)

    print(f"Found {len(dataset_files)} dataset file(s):")
    for f in dataset_files:
        print(f"  {f}")

    # ------------------------------------------------------------------
    # FIX 2: Load the trained final_model weights, not the base model.
    # The --model argument must point to training/output/final_model.
    # load_in_4bit keeps VRAM usage manageable for inference.
    # ------------------------------------------------------------------
    print(f"\nLoading model from: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        device_map={"": 0},
    )
    # Switch to inference mode (disables dropout, enables faster attn)
    FastLanguageModel.for_inference(model)
    model.eval()

    # ------------------------------------------------------------------
    # FIX 3: Use load_all_groups (handles *_checkpoint.json format).
    # Original code used load_sample_groups which expects legacy flat
    # {"samples": [...]} format and returns 0 groups on checkpoint files.
    # ------------------------------------------------------------------
    all_groups = []
    for f in dataset_files:
        loaded = load_sample_groups(f)
        all_groups.extend(loaded)
        print(f"  Loaded {len(loaded)} groups from {os.path.basename(f)}")

    if not all_groups:
        print(
            "ERROR: 0 groups loaded. Check that dataset files contain "
            "valid group data and that dataset_loader.py is the corrected version."
        )
        exit(1)

    print(f"\nTotal groups for evaluation: {len(all_groups)}")

    # ------------------------------------------------------------------
    # FIX 4: Removed [::5] subsampling — evaluate the full test split.
    # For the paper all 120 test groups must be evaluated.
    # If you want a quick smoke-test during debugging, pass --split val
    # (120 groups) rather than subsampling the test set.
    # ------------------------------------------------------------------
    eval_groups = all_groups
    print(f"Evaluating {len(eval_groups)} groups (full split, no subsampling)...")

    os.makedirs(args.results_dir, exist_ok=True)

    metric_results = []
    for i, group in enumerate(eval_groups):
        evaluated_group = _evaluate_group(
            model, tokenizer, group, max_new_tokens=256
        )
        result = compute_all(evaluated_group)
        metric_results.append(result)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(eval_groups)} groups done")

    aggregated = aggregate_metrics(metric_results)
    _print_metrics(aggregated, args.phase)

    out_path = os.path.join(
        args.results_dir,
        f"eval_{args.phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(out_path, "w") as f:
        json.dump(
            {
                "phase": args.phase,
                "model": args.model,
                "split": args.split,
                "n_groups": len(eval_groups),
                "dataset_files": dataset_files,
                "evaluated_at": datetime.now().isoformat(),
                "metrics": aggregated,
            },
            f,
            indent=2,
        )
    print(f"Results saved → {out_path}")

    # ------------------------------------------------------------------
    # Print full comparison across all eval phases saved in results dir
    # ------------------------------------------------------------------
    print("\nFull comparison across all eval phases:")
    comparison = compare_phases(args.results_dir)
    for phase_name, metrics in comparison.items():
        print(f"\n  {phase_name}:")
        for k, v in metrics.items():
            print(f"    {k:<25} {v:.4f}")