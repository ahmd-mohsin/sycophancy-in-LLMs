import glob
import json
import os
from datetime import datetime

from metrics.dataset_loader import load_all_groups
from rewards.reward_aggregator import RewardWeights, build_grpo_group


# ── Category inference ────────────────────────────────────────────────────────
# Build a topic→category map by scanning all checkpoint/dataset files once.
# This is the only reliable way — _infer_category previously returned 'opinion'
# for every group because the glob pattern was wrong AND it never checked
# whether the topic actually belonged to that file.

def _build_topic_category_map(dataset_dir: str) -> dict[str, str]:
    """
    Scan all dataset/checkpoint files and map each topic string to its category.
    Uses checkpoint files preferentially (most up-to-date).
    """
    topic_to_cat: dict[str, str] = {}

    patterns = [
        ("*_checkpoint.json",      lambda n: n.split("_")[0]),
        ("*_dataset_*.json",       lambda n: n.split("_")[0]),
        ("*_with_baselines.json",  lambda n: n.split("_")[0]),
    ]

    for pattern, cat_fn in patterns:
        for filepath in glob.glob(os.path.join(dataset_dir, pattern)):
            filename = os.path.basename(filepath)
            category = cat_fn(filename)
            if category not in ("math", "physics", "political", "opinion"):
                continue
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Handle all four on-disk formats (mirrors dataset_loader._detect_format)
                if isinstance(data, list):
                    groups = data
                elif "groups" in data:
                    groups = data["groups"]
                elif "samples" in data:
                    items = data["samples"]
                    # v2 nested format: "samples" holds group dicts with baseline_orig
                    if items and isinstance(items[0], dict) and "baseline_orig" in items[0]:
                        groups = items
                    else:
                        groups = []   # flat pressured-sample list — no topic-level groups
                else:
                    groups = []
                for g in groups:
                    topic = g.get("topic", "")
                    if topic and topic not in topic_to_cat:
                        topic_to_cat[topic] = category
            except Exception as e:
                print(f"  [warn] Could not parse {filename}: {e}")

    return topic_to_cat


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

    # Build topic→category map before loading groups
    topic_to_cat = _build_topic_category_map(dataset_dir)
    print(f"  Category map built: {len(topic_to_cat)} topics across "
          f"{len(set(topic_to_cat.values()))} categories")

    groups = load_all_groups(dataset_dir)
    print(f"  Loaded {len(groups)} groups from {dataset_dir}")

    if not groups:
        raise RuntimeError(
            f"load_all_groups returned 0 groups from {dataset_dir}. "
            f"Check that dataset_loader.py uses the correct glob pattern "
            f"(*_checkpoint.json or *_dataset_*.json)."
        )

    # Apply category filter correctly (was 'if True' — never filtered anything)
    if category_filter and category_filter != "all":
        before = len(groups)
        groups = [g for g in groups
                  if topic_to_cat.get(g.topic, "opinion") == category_filter]
        print(f"  Filtered to category='{category_filter}': {before} → {len(groups)} groups")

    grpo_groups = []
    category_counts = {}

    for i, group in enumerate(groups):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(groups)}] scoring...")

        # Use map; fall back to 'opinion' only if topic genuinely not found
        category = topic_to_cat.get(group.topic, "opinion")
        if group.topic not in topic_to_cat:
            print(f"  [warn] Topic '{group.topic}' not in category map — defaulting to 'opinion'")

        category_counts[category] = category_counts.get(category, 0) + 1
        grpo_group = build_grpo_group(group, weights, judge_model, category)
        grpo_group["category"] = category  # ensure category is stored in output
        grpo_groups.append(grpo_group)

    print(f"\n  Category breakdown: {category_counts}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"reward_dataset_{timestamp}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "n_groups": len(grpo_groups),
                "category_counts": category_counts,
                "weights": weights.__dict__,
                "judge_model": judge_model,
                "generated_at": timestamp,   # single consistent timestamp
            },
            "groups": grpo_groups,
        }, f, indent=2)

    print(f"\nReward dataset saved → {out_path}")
    return out_path


def load_reward_dataset(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["groups"]