import json
import os
from datetime import datetime

from metrics.dataset_loader import load_all_groups, load_sample_groups
from rewards.reward_aggregator import RewardWeights, build_grpo_group


def build_reward_dataset(
    dataset_dir: str,
    output_dir: str = "rewards/output",
    weights: RewardWeights = None,
    judge_model: str = "llama3.1:8b",
    category_filter: str = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    if weights is None:
        weights = RewardWeights()

    groups = load_all_groups(dataset_dir)

    if category_filter:
        groups = [g for g in groups if True]

    grpo_groups = []
    for i, group in enumerate(groups):
        if (i+1) % 20 == 0 or i == 0: print(f"  [{i+1}/{len(groups)}] scoring...")
        category = _infer_category(dataset_dir, group.topic)
        grpo_group = build_grpo_group(group, weights, judge_model, category)
        grpo_groups.append(grpo_group)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"reward_dataset_{timestamp}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "n_groups": len(grpo_groups),
                "weights": weights.__dict__,
                "judge_model": judge_model,
                "generated_at": datetime.now().isoformat(),
            },
            "groups": grpo_groups,
        }, f, indent=2)

    print(f"\nReward dataset saved → {out_path}")
    return out_path


def _infer_category(dataset_dir: str, topic: str) -> str:
    import glob
    for f in glob.glob(os.path.join(dataset_dir, "*_with_baselines.json")):
        name = os.path.basename(f)
        for cat in ("math", "physics", "political", "opinion"):
            if name.startswith(cat):
                return cat
    return "opinion"


def load_reward_dataset(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["groups"]