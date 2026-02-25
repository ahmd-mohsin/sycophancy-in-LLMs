import json
import os
from collections import defaultdict
from typing import Optional

from metrics.sycophancy_metrics import SampleGroup


def load_sample_groups(
    filepath: str,
    category_filter: Optional[str] = None,
) -> list[SampleGroup]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", [])

    if category_filter:
        samples = [s for s in samples if s["category"] == category_filter]

    samples = [s for s in samples if not s["response"].startswith("ERROR")]

    topic_map: dict[str, dict] = defaultdict(lambda: {
        "topic": "",
        "question": "",
        "opinion": "",
        "baseline_orig": None,
        "baseline_opp": None,
        "pressured": [],
    })

    for sample in samples:
        key = (sample["topic"], sample["components"]["question"])
        entry = topic_map[key]
        entry["topic"] = sample["topic"]
        entry["question"] = sample["components"]["question"]
        entry["opinion"] = sample["opinion"]

        is_baseline = sample.get("is_baseline") or sample.get("pressure_level") == "none"

        if is_baseline:
            if sample["context_type"] == "original":
                entry["baseline_orig"] = sample["response"]
            else:
                entry["baseline_opp"] = sample["response"]
        else:
            entry["pressured"].append(sample)

    groups = []
    for entry in topic_map.values():
        if entry["baseline_orig"] is None or entry["baseline_opp"] is None:
            continue
        if not entry["pressured"]:
            continue

        groups.append(SampleGroup(
            topic=entry["topic"],
            question=entry["question"],
            opinion=entry["opinion"],
            baseline_orig=entry["baseline_orig"],
            baseline_opp=entry["baseline_opp"],
            pressured=entry["pressured"],
        ))

    return groups


def load_all_groups(output_dir: str) -> list[SampleGroup]:
    import glob
    files = glob.glob(os.path.join(output_dir, "*_with_baselines.json"))
    groups = []
    for f in files:
        groups.extend(load_sample_groups(f))
    return groups