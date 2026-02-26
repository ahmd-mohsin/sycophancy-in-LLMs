import json
import os
import torch
from datetime import datetime
from datasets import Dataset
from transformers import pipeline

from metrics.evaluator import run_evaluation
from metrics.dataset_loader import load_sample_groups
from metrics.sycophancy_metrics import compute_all, aggregate_metrics, SampleGroup
from rewards.grpo_data_builder import build_grpo_prompt, GRPO_SYSTEM_PROMPT


def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    batch_size: int = 4,
) -> list[str]:
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for prompt, full in zip(batch, decoded):
            responses.append(full[len(prompt):].strip())

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

    out_path = os.path.join(output_dir, f"eval_{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    _print_metrics(aggregated, phase)
    return output


def _evaluate_group(model, tokenizer, group: SampleGroup, max_new_tokens: int) -> SampleGroup:
    updated_pressured = []
    for sample in group.pressured:
        messages = build_grpo_prompt(
            pressure_prompt=sample["components"]["pressure_prompt"],
            context=sample["components"]["context"],
            question=sample["components"]["question"],
        )
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        responses = generate_responses(model, tokenizer, [prompt], max_new_tokens=max_new_tokens)
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