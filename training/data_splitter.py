"""
training/data_splitter.py

Splits dataset files into train/val/test by topic group.
Handles both file formats:
  - Checkpoint format: raw list of group dicts with nested "samples"
  - Dataset format: dict with {"samples": [flat list]}
"""

import glob
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

    # ── Normalise to flat sample list ─────────────────────────────────────────
    # Checkpoint format: raw list of group dicts, each with nested "samples"
    # Dataset format:    dict with top-level "samples" flat list
    if isinstance(data, list):
        # Checkpoint: list of groups → flatten samples, injecting group-level
        # baseline fields into each sample so the flat format is self-contained
        flat_samples = []
        for group in data:
            group_topic         = group.get("topic", "")
            group_question      = group.get("question", "")
            group_baseline_orig = group.get("baseline_orig", "")
            group_baseline_opp  = group.get("baseline_opp", "")
            group_nli           = group.get("nli_contra", None)

            for s in group.get("samples", []):
                # Inject group-level fields if missing from sample
                s.setdefault("baseline_orig", group_baseline_orig)
                s.setdefault("baseline_opp",  group_baseline_opp)
                s.setdefault("nli_contra",     group_nli)
                # Normalise question: samples store it in components dict
                if "components" not in s:
                    s["components"] = {"question": group_question}
                elif "question" not in s["components"]:
                    s["components"]["question"] = group_question
                flat_samples.append(s)

        metadata = {
            "source": os.path.basename(dataset_path),
            "format": "checkpoint",
            "n_groups": len(data),
        }

    elif isinstance(data, dict) and "samples" in data:
        flat_samples = data["samples"]
        metadata     = data.get("metadata", {})

    elif isinstance(data, dict) and "groups" in data:
        # with_baselines format
        flat_samples = []
        for group in data["groups"]:
            group_question      = group.get("question", "")
            group_baseline_orig = group.get("baseline_orig", "")
            group_baseline_opp  = group.get("baseline_opp", "")
            for s in group.get("samples", []):
                s.setdefault("baseline_orig", group_baseline_orig)
                s.setdefault("baseline_opp",  group_baseline_opp)
                if "components" not in s:
                    s["components"] = {"question": group_question}
                elif "question" not in s["components"]:
                    s["components"]["question"] = group_question
                flat_samples.append(s)
        metadata = data.get("metadata", {})

    else:
        raise ValueError(
            f"Unrecognised format in {dataset_path}. "
            f"Expected a list of groups or a dict with 'samples'/'groups'."
        )

    # ── Group by topic so we split at topic level, not sample level ───────────
    topic_map = defaultdict(list)
    for s in flat_samples:
        # Safely extract question from components or top-level
        question = (
            s.get("components", {}).get("question")
            or s.get("question", "")
        )
        key = (s.get("topic", ""), question)
        topic_map[key].append(s)

    topics = list(topic_map.keys())
    random.shuffle(topics)

    n       = len(topics)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_topics = set(topics[:n_train])
    val_topics   = set(topics[n_train : n_train + n_val])
    test_topics  = set(topics[n_train + n_val :])

    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    for topic_key, topic_samples in topic_map.items():
        if topic_key in train_topics:
            splits["train"].extend(topic_samples)
        elif topic_key in val_topics:
            splits["val"].extend(topic_samples)
        else:
            splits["test"].extend(topic_samples)

    paths = {}
    base  = os.path.splitext(os.path.basename(dataset_path))[0]

    for split_name, split_samples in splits.items():
        n_topics_in_split = len(
            train_topics if split_name == "train"
            else val_topics if split_name == "val"
            else test_topics
        )
        out_path = os.path.join(output_dir, f"{base}_{split_name}.json")
        with open(out_path, "w") as f:
            json.dump({
                "metadata": {
                    **metadata,
                    "split":    split_name,
                    "n_samples": len(split_samples),
                    "n_topics":  n_topics_in_split,
                },
                "samples": split_samples,
            }, f, indent=2)
        paths[split_name] = out_path
        print(f"    {split_name:5s} → {len(split_samples):5d} samples, "
              f"{n_topics_in_split} topics  ({out_path})")

    return paths


def split_all_datasets(
    dataset_dir: str,
    output_dir: str  = "training/splits",
    train_frac: float = 0.70,
    val_frac: float   = 0.15,
    seed: int         = 42,
) -> dict[str, dict[str, str]]:
    """
    Split all dataset files found in dataset_dir.
    Uses checkpoint files as the primary source (most up-to-date).
    Falls back to *_dataset_*.json if no checkpoints found.
    """
    # Prefer checkpoint files — always current
    files = sorted(glob.glob(os.path.join(dataset_dir, "*_checkpoint.json")))

    if not files:
        # Fallback: timestamped dataset files
        files = sorted(glob.glob(os.path.join(dataset_dir, "*_dataset_*.json")))

    if not files:
        raise RuntimeError(
            f"No dataset files found in '{dataset_dir}'.\n"
            f"Expected: *_checkpoint.json or *_dataset_*.json\n"
            f"Found: {glob.glob(os.path.join(dataset_dir, '*.json'))}"
        )

    all_splits = {}
    for f in files:
        # Extract category name from filename
        name = os.path.basename(f).split("_")[0]
        print(f"  Splitting {name} ({os.path.basename(f)})...")
        all_splits[name] = split_dataset(f, output_dir, train_frac, val_frac, seed)

    return all_splits