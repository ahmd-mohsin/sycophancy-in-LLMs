import argparse
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.dataset_loader import load_all_groups
from rewards.reward_aggregator import RewardWeights, compute_group_rewards


def _latest(path_pattern: str) -> str | None:
    matches = glob.glob(path_pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _flatten_samples(reward_groups: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for g in reward_groups:
        for s in g.get("samples", []):
            row = s.copy()
            row["topic"] = g.get("topic")
            row["question"] = g.get("question")
            rows.append(row)
    return rows


def describe_reward_dataset(dataset_dir: str, rewards_dir: str) -> None:
    print("\n=== REWARD DATASET (NLI-BASED SCORES) ===")
    latest_reward = _latest(os.path.join(rewards_dir, "reward_dataset_*.json"))
    if not latest_reward:
        print(f"No reward_dataset_*.json found under {rewards_dir}")
        return

    print(f"Using reward dataset: {latest_reward}")
    with open(latest_reward, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    meta = data.get("metadata", {})
    reward_groups = data.get("groups", [])
    print(f"Total GRPO-style groups in reward dataset: {len(reward_groups)}")
    print(f"Weights: {meta.get('weights')}")
    print(f"Judge model: {meta.get('judge_model')}")

    samples = _flatten_samples(reward_groups)
    print(f"Total scored samples (pressured): {len(samples)}")

    by_pressure = Counter(s["pressure_level"] for s in samples)
    by_context = Counter(s["context_type"] for s in samples)

    print("\nSamples by pressure_level:")
    for lvl, n in sorted(by_pressure.items()):
        print(f"  {lvl:<10} {n:5d}")

    print("\nSamples by context_type:")
    for ctx, n in sorted(by_context.items()):
        print(f"  {ctx:<10} {n:5d}")

    def stat(field: str) -> tuple[float, float, float]:
        vals = [float(s[field]) for s in samples]
        return min(vals), sum(vals) / len(vals), max(vals)

    for name in ["rq", "rc", "rp", "rpos", "rg", "total", "advantage"]:
        if name not in samples[0]:
            continue
        vmin, vmean, vmax = stat(name)
        print(f"\n{name} stats:")
        print(f"  min  : {vmin: .4f}")
        print(f"  mean : {vmean: .4f}")
        print(f"  max  : {vmax: .4f}")

    print("\nExample scored sample (first row):")
    ex = samples[0]
    for k in ["topic", "pressure_level", "context_type", "rq", "rc", "rp", "rpos", "rg", "total", "advantage"]:
        if k in ex:
            print(f"  {k:<14}: {ex[k]}")

    print("\n=== CONSISTENCY CHECK WITH LIVE NLI PIPELINE ===")
    groups = load_all_groups(dataset_dir)
    keyed = {(g.topic, g.question): g for g in groups}

    weights = RewardWeights(**meta.get("weights", {})) if meta.get("weights") else RewardWeights()
    max_checks = 3
    checked = 0

    for rg in reward_groups:
        if checked >= max_checks:
            break
        key = (rg["topic"], rg["question"])
        if key not in keyed:
            continue

        print(f"\nRecomputing rewards for topic='{key[0]}'")
        g = keyed[key]
        category = "math"
        if "math" not in rg["topic"].lower() and "physics" not in rg["topic"].lower():
            category = "opinion"

        live = compute_group_rewards(g, weights=weights, judge_model=meta.get("judge_model"), category=category)
        live_by_id = {r.sample_id: r for r in live}

        for stored in rg["samples"]:
            sid = stored["sample_id"]
            if sid not in live_by_id:
                continue
            r_live = live_by_id[sid]
            print(f"  sample_id={sid}")
            print(f"    stored total: {stored['total']:.4f}, live total: {r_live.total:.4f}")

            # Older reward datasets (v2) do not have rpos; handle both cases.
            has_rpos = "rpos" in stored
            if has_rpos:
                print(
                    "    stored rq/rc/rp/rpos/rg: "
                    f"{stored['rq']:.3f}, {stored['rc']:.3f}, {stored['rp']:.3f}, "
                    f"{stored['rpos']:.3f}, {stored['rg']:.3f}"
                )
                print(
                    "    live   rq/rc/rp/rpos/rg: "
                    f"{r_live.rq:.3f}, {r_live.rc:.3f}, {r_live.rp:.3f}, "
                    f"{r_live.rpos:.3f}, {r_live.rg:.3f}"
                )
            else:
                print(
                    "    stored rq/rc/rp/rg    : "
                    f"{stored['rq']:.3f}, {stored['rc']:.3f}, "
                    f"{stored['rp']:.3f}, {stored['rg']:.3f}"
                )
                print(
                    "    live   rq/rc/rp/rg    : "
                    f"{r_live.rq:.3f}, {r_live.rc:.3f}, "
                    f"{r_live.rp:.3f}, {r_live.rg:.3f}"
                )

        checked += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect reward dataset and verify NLI-based scores.")
    parser.add_argument("--dataset-dir", default="dataset_generation/output")
    parser.add_argument("--rewards-dir", default="training/output/rewards")
    args = parser.parse_args()

    print("REWARD DATASET AND NLI PIPELINE INSPECTOR")
    print(f"  dataset dir : {args.dataset_dir}")
    print(f"  rewards dir : {args.rewards_dir}")

    describe_reward_dataset(args.dataset_dir, args.rewards_dir)


if __name__ == "__main__":
    main()

