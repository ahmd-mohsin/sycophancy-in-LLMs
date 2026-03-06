"""
metrics/dataset_loader.py

Loads dataset files into SampleGroup objects for reward scoring.

Handles all three file formats produced by the pipeline:
  1. Checkpoint format  — raw list of group dicts, each with nested "samples",
                          "baseline_orig", "baseline_opp" as top-level group fields.
                          Files: *_checkpoint.json
  2. Dataset format     — dict with {"metadata": ..., "samples": [flat list]}
                          where baselines are stored separately per-group.
                          Files: *_dataset_TIMESTAMP.json
  3. Baselines format   — dict with {"metadata": ..., "groups": [group list]}
                          Files: *_with_baselines.json (if present)

The original loader only handled format 2 (flat sample list with is_baseline flags)
and used a glob that matched none of the actual files. Both issues are fixed here.
"""

import glob
import json
import os
from collections import defaultdict
from typing import Optional

from metrics.sycophancy_metrics import SampleGroup


# ── Format detection ──────────────────────────────────────────────────────────

def _detect_format(data) -> str:
    """Return 'checkpoint', 'groups', 'nested_samples', or 'flat_samples'.

    'nested_samples' is the v2 dataset format:
        {"metadata": ..., "samples": [<group dicts with baseline_orig/baseline_opp>]}
    Each item in "samples" is a GROUP (not a pressured sample), identifiable by the
    presence of "baseline_orig" directly on the dict.
    """
    if isinstance(data, list):
        return "checkpoint"
    if isinstance(data, dict):
        if "groups" in data:
            return "groups"
        if "samples" in data:
            items = data["samples"]
            # v2 format: "samples" holds group dicts that carry baseline_orig/baseline_opp
            if items and isinstance(items[0], dict) and "baseline_orig" in items[0]:
                return "nested_samples"
            return "flat_samples"
    return "unknown"


# ── Format 1: checkpoint — list of group dicts ────────────────────────────────

def _load_checkpoint_format(groups_list: list, category_filter: Optional[str]) -> list[SampleGroup]:
    """
    Checkpoint files are a raw list of group dicts:
    [
      {
        "topic": "...",
        "question": "...",
        "opinion": "...",
        "opposite_opinion": "...",   # optional
        "baseline_orig": "...",      # committed baseline for original context
        "baseline_opp":  "...",      # committed baseline for opposite context
        "nli_contra": 0.999,
        "samples": [                 # 6 pressured samples
          {
            "id": "...",
            "category": "political",
            "pressure_level": "low",
            "context_type": "original",
            "response": "...",
            "opinion": "...",
            "components": {"context": "...", "question": "...", "pressure_prompt": "..."},
            "full_prompt": "..."
          },
          ...
        ]
      },
      ...
    ]
    """
    result = []
    for group in groups_list:
        baseline_orig = group.get("baseline_orig")
        baseline_opp  = group.get("baseline_opp")

        if not baseline_orig or not baseline_opp:
            continue

        pressured = group.get("samples", [])

        # Filter out error responses
        pressured = [s for s in pressured if not s.get("response", "").startswith("ERROR")]

        if not pressured:
            continue

        # Apply category filter using the category field on the first sample
        if category_filter:
            sample_cat = pressured[0].get("category", "") if pressured else ""
            if sample_cat != category_filter:
                continue

        result.append(SampleGroup(
            topic=group.get("topic", ""),
            question=group.get("question", ""),
            opinion=group.get("opinion", ""),
            baseline_orig=baseline_orig,
            baseline_opp=baseline_opp,
            pressured=pressured,
        ))

    return result


# ── Format 2: flat sample list ────────────────────────────────────────────────

def _load_flat_samples_format(samples: list, category_filter: Optional[str]) -> list[SampleGroup]:
    """
    Flat sample list where baselines are identified by is_baseline flag
    or pressure_level == 'none'. Each group is reconstructed by grouping
    on (topic, question).
    """
    if category_filter:
        samples = [s for s in samples if s.get("category") == category_filter]

    samples = [s for s in samples if not s.get("response", "").startswith("ERROR")]

    topic_map: dict[tuple, dict] = defaultdict(lambda: {
        "topic": "",
        "question": "",
        "opinion": "",
        "baseline_orig": None,
        "baseline_opp": None,
        "pressured": [],
    })

    for sample in samples:
        components = sample.get("components", {})
        question   = components.get("question", sample.get("question", ""))
        key        = (sample["topic"], question)
        entry      = topic_map[key]

        entry["topic"]    = sample["topic"]
        entry["question"] = question
        entry["opinion"]  = sample.get("opinion", "")

        is_baseline = (
            sample.get("is_baseline")
            or sample.get("pressure_level") == "none"
        )

        if is_baseline:
            ctx = sample.get("context_type", "")
            if ctx == "original":
                entry["baseline_orig"] = sample["response"]
            elif ctx == "opposite":
                entry["baseline_opp"] = sample["response"]
        else:
            entry["pressured"].append(sample)

    result = []
    for entry in topic_map.values():
        if entry["baseline_orig"] is None or entry["baseline_opp"] is None:
            continue
        if not entry["pressured"]:
            continue
        result.append(SampleGroup(
            topic=entry["topic"],
            question=entry["question"],
            opinion=entry["opinion"],
            baseline_orig=entry["baseline_orig"],
            baseline_opp=entry["baseline_opp"],
            pressured=entry["pressured"],
        ))

    return result


# ── Format 3: groups list (with_baselines format) ─────────────────────────────

def _load_groups_format(groups_list: list, category_filter: Optional[str]) -> list[SampleGroup]:
    """
    Groups-format files (add_baselines_v2 output):
    {"metadata": ..., "groups": [{topic, baseline_orig, baseline_opp, samples: [...]}]}
    Same structure as checkpoint, just wrapped in a dict.
    """
    return _load_checkpoint_format(groups_list, category_filter)


# ── Public API ────────────────────────────────────────────────────────────────

def load_sample_groups(
    filepath: str,
    category_filter: Optional[str] = None,
) -> list[SampleGroup]:
    """Load SampleGroups from any supported file format."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    fmt = _detect_format(data)

    if fmt == "checkpoint":
        return _load_checkpoint_format(data, category_filter)

    if fmt == "groups":
        return _load_groups_format(data["groups"], category_filter)

    if fmt == "nested_samples":
        # v2 format: {"metadata": ..., "samples": [<group dicts>]}
        # The group dicts have the same structure as checkpoint groups.
        return _load_checkpoint_format(data["samples"], category_filter)

    if fmt == "flat_samples":
        return _load_flat_samples_format(data["samples"], category_filter)

    raise ValueError(
        f"Unrecognised dataset format in {filepath}. "
        f"Expected a list of groups, a dict with 'groups', or a dict with 'samples'."
    )


def load_all_groups(
    output_dir: str,
    category_filter: Optional[str] = None,
    prefer_checkpoint: bool = True,
) -> list[SampleGroup]:
    """
    Load SampleGroups from all dataset files in output_dir.

    By default loads from checkpoint files (*_checkpoint.json) which are
    always up-to-date. Falls back to *_dataset_*.json if no checkpoints found.
    Also loads *_with_baselines.json if present.

    Setting prefer_checkpoint=False loads from *_dataset_*.json instead
    (use this if you want the finalised timestamped output rather than the
    live checkpoint).

    Deduplicates by (topic, question) across files so loading both checkpoint
    and dataset files for the same category doesn't double-count.
    """
    seen: set[tuple[str, str]] = set()
    groups: list[SampleGroup] = []

    def _add(new_groups: list[SampleGroup]):
        for g in new_groups:
            key = (g.topic, g.question)
            if key not in seen:
                seen.add(key)
                groups.append(g)

    # Primary source
    primary_pattern = "*_checkpoint.json" if prefer_checkpoint else "*_dataset_*.json"
    primary_files = sorted(glob.glob(os.path.join(output_dir, primary_pattern)))

    if not primary_files:
        # Fallback to the other pattern
        fallback_pattern = "*_dataset_*.json" if prefer_checkpoint else "*_checkpoint.json"
        primary_files = sorted(glob.glob(os.path.join(output_dir, fallback_pattern)))

    for f in primary_files:
        loaded = load_sample_groups(f, category_filter)
        print(f"  Loaded {len(loaded):4d} groups from {os.path.basename(f)}")
        _add(loaded)

    # Also pick up any with_baselines files (may have better baselines from v2 regeneration)
    for f in sorted(glob.glob(os.path.join(output_dir, "*_with_baselines.json"))):
        loaded = load_sample_groups(f, category_filter)
        before = len(groups)
        _add(loaded)
        added = len(groups) - before
        if added:
            print(f"  Loaded {added:4d} new groups from {os.path.basename(f)} (with_baselines)")

    if not groups:
        raise RuntimeError(
            f"load_all_groups found 0 groups in '{output_dir}'.\n"
            f"Files found: {glob.glob(os.path.join(output_dir, '*.json'))}\n"
            f"Expected files matching: *_checkpoint.json or *_dataset_*.json"
        )

    print(f"  Total: {len(groups)} unique groups loaded")
    return groups