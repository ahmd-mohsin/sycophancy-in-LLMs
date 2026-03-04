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

FACTUAL_CATEGORIES = {"math", "physics"}


def _build_context_map(original_dataset_dir: str) -> dict[str, str]:
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


def _infer_category(group: dict) -> str:
    stored = group.get("category")
    if stored:
        return stored
    raise ValueError(
        f"Group for topic '{group.get('topic')}' has no 'category' field. "
        "Rebuild the reward dataset with the updated reward_aggregator.py that "
        "writes 'category' into each group."
    )


def build_grpo_training_data(
    reward_dataset_path: str,
    output_dir: str = "rewards/output",
    original_dataset_dir: str = None,
) -> str:
    if not original_dataset_dir:
        raise ValueError(
            "original_dataset_dir is required. "
            "Pass --dataset-dir so actual context strings are embedded, "
            "not truncated baseline responses."
        )

    os.makedirs(output_dir, exist_ok=True)
    groups = load_reward_dataset(reward_dataset_path)
    context_map = _build_context_map(original_dataset_dir)

    missing_ctx = 0
    records = []
    for group in groups:
        category = _infer_category(group)
        completions = []

        for sample in group["samples"]:
            sample_id    = sample["sample_id"]
            pressure_level = sample["pressure_level"]
            context_type   = sample["context_type"]

            context = context_map.get(sample_id, "")
            if not context:
                missing_ctx += 1

            completions.append({
                "sample_id":     sample_id,
                "pressure_level": pressure_level,
                "context_type":  context_type,
                "context":       context,
            })

        records.append({
            "topic":        group["topic"],
            "question":     group["question"],
            "opinion":      group["opinion"],
            "baseline_orig": group["baseline_orig"],
            "baseline_opp":  group["baseline_opp"],
            "category":     category,
            "system":       GRPO_SYSTEM_PROMPT,
            "completions":  completions,
        })

    if missing_ctx:
        print(
            f"WARNING: {missing_ctx} completions had no context string. "
            "Check that original_dataset_dir matches the dataset used to build "
            "the reward dataset."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"grpo_training_{timestamp}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "n_groups":    len(records),
                    "generated_at": datetime.now().isoformat(),
                    "missing_contexts": missing_ctx,
                },
                "groups": records,
            },
            f,
            indent=2,
        )

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