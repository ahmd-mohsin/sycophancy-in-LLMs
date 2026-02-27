import glob
import json
import os
from datetime import datetime

from rewards.reward_dataset import load_reward_dataset
from metrics.dataset_loader import load_sample_groups

GRPO_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer based on the evidence in the context. "
    "Do not change your answer based on who is asking or their credentials."
)


def _build_context_map(original_dataset_dir: str) -> dict[str, str]:
    """Build sample_id -> actual context mapping from original dataset files."""
    context_map = {}
    files = glob.glob(os.path.join(original_dataset_dir, "*_with_baselines.json"))
    for filepath in files:
        groups = load_sample_groups(filepath)
        for group in groups:
            for sample in group.pressured:
                sid = sample.get("id") or sample.get("sample_id")
                ctx = sample.get("components", {}).get("context", "")
                if sid and ctx:
                    context_map[sid] = ctx
    return context_map


def build_grpo_training_data(
    reward_dataset_path: str,
    output_dir: str = "rewards/output",
    original_dataset_dir: str = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    groups = load_reward_dataset(reward_dataset_path)

    # Build context map from original dataset if provided
    context_map = {}
    if original_dataset_dir:
        context_map = _build_context_map(original_dataset_dir)

    records = []
    for group in groups:
        completions = []
        advantages = []

        for sample in group["samples"]:
            sample_id = sample["sample_id"]
            pressure_level = sample["pressure_level"]
            context_type = sample["context_type"]
            advantage = sample["advantage"]

            # Get actual context (much shorter than baseline response)
            context = context_map.get(sample_id, "")

            completions.append({
                "sample_id": sample_id,
                "pressure_level": pressure_level,
                "context_type": context_type,
                "context": context,
            })
            advantages.append(advantage)

        records.append({
            "topic": group["topic"],
            "question": group["question"],
            "opinion": group["opinion"],
            "baseline_orig": group["baseline_orig"],
            "baseline_opp": group["baseline_opp"],
            "system": GRPO_SYSTEM_PROMPT,
            "completions": completions,
            "advantages": advantages,
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"grpo_training_{timestamp}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "n_groups": len(records),
                "generated_at": datetime.now().isoformat(),
                "has_contexts": bool(context_map),
            },
            "groups": records,
        }, f, indent=2)

    print(f"GRPO training data saved → {out_path} ({len(records)} groups)")
    return out_path


def build_grpo_prompt(
    pressure_prompt: str,
    context: str,
    question: str,
) -> list[dict]:
    user_content = f"{pressure_prompt}\n\nContext: {context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": GRPO_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def load_grpo_training_data(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["groups"]