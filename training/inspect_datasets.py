import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.dataset_loader import load_all_groups, load_sample_groups
from rewards.sft_data_builder import load_sft_dataset
from rewards.grpo_data_builder import load_grpo_training_data


def find_latest(path_pattern: str) -> str | None:
    """Return most recently modified file matching the glob pattern."""
    matches = glob.glob(path_pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def describe_raw_datasets(dataset_dir: str) -> None:
    """Print characteristics of original *_with_baselines.json datasets."""
    print("\n=== RAW DATASETS (WITH BASELINES) ===")
    files = sorted(glob.glob(os.path.join(dataset_dir, "*_with_baselines.json")))
    if not files:
        print(f"No *_with_baselines.json files found under {dataset_dir}")
        return

    print(f"Found {len(files)} dataset files:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    all_groups = load_all_groups(dataset_dir)
    print(f"\nTotal SampleGroups across all files: {len(all_groups)}")

    per_file_groups: dict[str, int] = {}
    per_category_samples: Counter = Counter()
    pressure_counts: Counter = Counter()
    context_type_counts: Counter = Counter()

    for f in files:
        groups = load_sample_groups(f)
        per_file_groups[os.path.basename(f)] = len(groups)

        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for s in data.get("samples", []):
            cat = s.get("category", "unknown")
            per_category_samples[cat] += 1
            pressure = s.get("pressure_level", "none")
            pressure_counts[pressure] += 1
            ctx_type = s.get("context_type", "unknown")
            context_type_counts[ctx_type] += 1

    print("\nGroups per file:")
    for name, n in per_file_groups.items():
        print(f"  {name:<45} {n:4d} groups")

    print("\nSample counts by category:")
    for cat, n in sorted(per_category_samples.items()):
        print(f"  {cat:<10} {n:5d} samples")

    print("\nSample counts by pressure level:")
    for level, n in sorted(pressure_counts.items()):
        print(f"  {level:<10} {n:5d} samples")

    print("\nSample counts by context type:")
    for ctx, n in sorted(context_type_counts.items()):
        print(f"  {ctx:<10} {n:5d} samples")

    if all_groups:
        g = all_groups[0]
        print("\nExample SampleGroup structure (first group):")
        print(f"  topic          : {g.topic}")
        print(f"  question       : {g.question}")
        print(f"  opinion        : {g.opinion}")
        print(f"  baseline_orig  : {len(g.baseline_orig.split()):4d} words")
        print(f"  baseline_opp   : {len(g.baseline_opp.split()):4d} words")
        print(f"  pressured count: {len(g.pressured)}")
        pressure_ctx = Counter((s.get('pressure_level'), s.get('context_type')) for s in g.pressured)
        print("  pressured by (pressure_level, context_type):")
        for key, n in sorted(pressure_ctx.items()):
            print(f"    {key}: {n}")


def describe_sft_pairs(rewards_dir: str) -> None:
    """Print how SFT pairs were formed and their statistics."""
    print("\n=== SFT DATASET (PHASE 1) ===")
    latest_sft = find_latest(os.path.join(rewards_dir, "sft_dataset_*.json"))
    if not latest_sft:
        print(f"No SFT dataset found under {rewards_dir}")
        return

    print(f"Using SFT dataset: {latest_sft}")
    samples = load_sft_dataset(latest_sft)
    print(f"Total SFT samples: {len(samples)}")

    by_topic: Counter = Counter(s.get("topic", "unknown") for s in samples)
    by_ctx: Counter = Counter(s.get("context_type", "unknown") for s in samples)

    print("\nSFT samples by context_type:")
    for ctx, n in sorted(by_ctx.items()):
        print(f"  {ctx:<10} {n:5d} samples")

    print("\nNumber of distinct topics in SFT dataset:")
    print(f"  {len(by_topic)} topics")

    example = samples[0] if samples else None
    if example:
        print("\nExample SFT pair:")
        print(f"  topic        : {example.get('topic')}")
        print(f"  context_type : {example.get('context_type')}")
        print(f"  prompt words : {len(example.get('prompt', '').split())}")
        print(f"  response words: {len(example.get('response', '').split())}")
        print("  system prompt:")
        print(f"    {example.get('system')}")


def describe_grpo_groups(rewards_dir: str) -> None:
    """Print how GRPO groups and completions are structured."""
    print("\n=== GRPO DATA (PHASE 2) ===")
    latest_grpo = find_latest(os.path.join(rewards_dir, "grpo_training_*.json"))
    if not latest_grpo:
        print(f"No GRPO training dataset found under {rewards_dir}")
        return

    print(f"Using GRPO dataset: {latest_grpo}")
    groups = load_grpo_training_data(latest_grpo)
    print(f"Total GRPO groups: {len(groups)}")

    completions_per_group: Counter = Counter(len(g.get("completions", [])) for g in groups)
    print("\nCompletions per group distribution:")
    for k, n in sorted(completions_per_group.items()):
        print(f"  {k:2d} completions → {n:4d} groups")

    pressure_counts: Counter = Counter()
    context_type_counts: Counter = Counter()
    for g in groups:
        for c in g.get("completions", []):
            pressure_counts[c.get("pressure_level", "unknown")] += 1
            context_type_counts[c.get("context_type", "unknown")] += 1

    print("\nGRPO completions by pressure_level:")
    for lvl, n in sorted(pressure_counts.items()):
        print(f"  {lvl:<10} {n:5d} completions")

    print("\nGRPO completions by context_type:")
    for ctx, n in sorted(context_type_counts.items()):
        print(f"  {ctx:<10} {n:5d} completions")

    if groups:
        g0 = groups[0]
        print("\nExample GRPO group (first group):")
        print(f"  topic        : {g0.get('topic')}")
        print(f"  question     : {g0.get('question')}")
        print(f"  opinion      : {g0.get('opinion')}")
        print(f"  baseline_orig words: {len(g0.get('baseline_orig', '').split()):4d}")
        print(f"  baseline_opp  words: {len(g0.get('baseline_opp', '').split()):4d}")
        print(f"  num completions    : {len(g0.get('completions', []))}")

        by_pair: defaultdict[tuple[str, str], int] = defaultdict(int)
        for c in g0.get("completions", []):
            key = (c.get("pressure_level", "unknown"), c.get("context_type", "unknown"))
            by_pair[key] += 1
        print("  completions by (pressure_level, context_type):")
        for key, n in sorted(by_pair.items()):
            print(f"    {key}: {n}")

        if g0.get("completions"):
            c0 = g0["completions"][0]
            print("\n  Example completion meta (first in group):")
            print(f"    sample_id     : {c0.get('sample_id')}")
            print(f"    pressure_level: {c0.get('pressure_level')}")
            print(f"    context_type  : {c0.get('context_type')}")
            ctx = c0.get("context", "")
            print(f"    context words : {len(ctx.split()):4d}")


def main() -> None:
    """Entry point for inspecting dataset and training data artifacts."""
    parser = argparse.ArgumentParser(description="Inspect SFT/GRPO dataset characteristics.")
    parser.add_argument("--dataset-dir", default="dataset_generation/output")
    parser.add_argument("--rewards-dir", default="training/output/rewards")
    args = parser.parse_args()

    print("DATASET AND TRAINING ARTIFACT INSPECTOR")
    print(f"  dataset dir : {args.dataset_dir}")
    print(f"  rewards dir : {args.rewards_dir}")

    describe_raw_datasets(args.dataset_dir)
    describe_sft_pairs(args.rewards_dir)
    describe_grpo_groups(args.rewards_dir)


if __name__ == "__main__":
    main()

