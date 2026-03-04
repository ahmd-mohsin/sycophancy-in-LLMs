import json
import os
import random
from collections import defaultdict


def split_dataset(
    dataset_path: str,
    output_dir: str = "training/splits",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    with open(dataset_path, "r") as f:
        data = json.load(f)

    # Normalise across three on-disk formats:
    #   1. Raw list of groups  (checkpoint format)
    #   2. {"groups": [...]}   (reward-dataset format)
    #   3. {"samples": [...]}  (flat pressured-sample format, legacy / final dataset)
    if isinstance(data, list):
        groups = data
        samples = [s for g in groups for s in g.get("samples", [])]
        metadata = {}
    elif "groups" in data:
        groups = data["groups"]
        samples = [s for g in groups for s in g.get("samples", [])]
        metadata = data.get("metadata", {})
    else:
        samples = data["samples"]
        metadata = data.get("metadata", {})

    topic_map = defaultdict(list)
    for s in samples:
        key = (s["topic"], s["components"]["question"])
        topic_map[key].append(s)

    topics = list(topic_map.keys())
    random.shuffle(topics)

    n = len(topics)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_topics = set(topics[:n_train])
    val_topics = set(topics[n_train:n_train + n_val])
    test_topics = set(topics[n_train + n_val:])

    splits = {"train": [], "val": [], "test": []}
    for topic_key, topic_samples in topic_map.items():
        if topic_key in train_topics:
            splits["train"].extend(topic_samples)
        elif topic_key in val_topics:
            splits["val"].extend(topic_samples)
        else:
            splits["test"].extend(topic_samples)

    paths = {}
    for split_name, split_samples in splits.items():
        base = os.path.splitext(os.path.basename(dataset_path))[0]
        out_path = os.path.join(output_dir, f"{base}_{split_name}.json")
        with open(out_path, "w") as f:
            json.dump({
                "metadata": {
                    **metadata,
                    "split": split_name,
                    "n_samples": len(split_samples),
                    "n_topics": len(train_topics if split_name == "train" else val_topics if split_name == "val" else test_topics),
                },
                "samples": split_samples,
            }, f, indent=2)
        paths[split_name] = out_path
        print(f"  {split_name:5s} → {len(split_samples):4d} samples ({out_path})")

    return paths


def split_all_datasets(
    dataset_dir: str,
    output_dir: str = "training/splits",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, dict[str, str]]:
    import glob

    # Prefer final dataset files over checkpoints when both exist for a category.
    # Final files match *_dataset_*.json; checkpoints match *_checkpoint.json.
    seen: dict[str, str] = {}
    for pattern in ("*_dataset_*.json", "*_checkpoint.json"):
        for f in glob.glob(os.path.join(dataset_dir, pattern)):
            name = os.path.basename(f).split("_")[0]
            if name not in seen:
                seen[name] = f

    all_splits = {}
    for name, f in seen.items():
        print(f"\nSplitting {name} ({os.path.basename(f)})...")
        all_splits[name] = split_dataset(f, output_dir, train_frac, val_frac, seed)
    return all_splits