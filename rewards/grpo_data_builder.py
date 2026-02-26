import json
import os
from datetime import datetime

from rewards.reward_dataset import load_reward_dataset


GRPO_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer based on the evidence in the context. "
    "Do not change your answer based on who is asking or their credentials."
)


def build_grpo_training_data(
    reward_dataset_path: str,
    output_dir: str = "rewards/output",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    groups = load_reward_dataset(reward_dataset_path)

    records = []
    for group in groups:
        completions = []
        advantages = []

        for sample in group["samples"]:
            sample_id = sample["sample_id"]
            pressure_level = sample["pressure_level"]
            context_type = sample["context_type"]
            advantage = sample["advantage"]

            completions.append({
                "sample_id": sample_id,
                "pressure_level": pressure_level,
                "context_type": context_type,
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