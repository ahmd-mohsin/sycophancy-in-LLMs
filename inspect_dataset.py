"""
inspect_dataset.py

Validates and analyses the generated dataset BEFORE training.
Runs directly on the raw checkpoint/dataset JSON files.

Usage:
    # Inspect all categories in output dir
    python inspect_dataset.py --dataset-dir dataset_generation/output

    # Inspect a single file
    python inspect_dataset.py --file dataset_generation/output/political_checkpoint.json

    # Full NLI re-validation (slow, requires GPU)
    python inspect_dataset.py --dataset-dir dataset_generation/output --rerun-nli

    # Save a detailed report
    python inspect_dataset.py --dataset-dir dataset_generation/output --save-report
"""

import argparse
import glob
import json
import os
import sys
import re
from collections import defaultdict, Counter
from typing import Optional

# ── ANSI colours ──────────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(s):    return f"{GREEN}✓ {s}{RESET}"
def warn(s):  return f"{YELLOW}⚠ {s}{RESET}"
def fail(s):  return f"{RED}✗ {s}{RESET}"
def header(s):return f"\n{BOLD}{CYAN}{'═'*60}\n  {s}\n{'═'*60}{RESET}"
def sub(s):   return f"{BOLD}{s}{RESET}"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_file(path: str) -> list[dict]:
    """Load either a checkpoint (list) or dataset (dict with 'samples' or 'groups')."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if "groups" in data:
        return data["groups"]
    if "samples" in data:
        items = data["samples"]
        if not items:
            return []
        # v2 format: top-level "samples" is actually a list of GROUP dicts,
        # each with keys: topic, question, opinion, baseline_orig, baseline_opp,
        # nli_contra, samples (list of 6 pressured sample dicts).
        if isinstance(items[0], dict) and "baseline_orig" in items[0]:
            return items
        # Legacy flat format: each item is a pressured sample — reconstruct groups.
        groups: dict[tuple, dict] = {}
        for s in items:
            key = (s.get("topic", "unknown"), s.get("components", {}).get("question", ""))
            if key not in groups:
                groups[key] = {
                    "topic":         s.get("topic", "unknown"),
                    "question":      s.get("components", {}).get("question", ""),
                    "opinion":       s.get("opinion", ""),
                    "baseline_orig": "",
                    "baseline_opp":  "",
                    "nli_contra":    None,
                    "samples":       [],
                }
            groups[key]["samples"].append(s)
        return list(groups.values())
    return []


def find_datasets(directory: str) -> list[str]:
    patterns = ["*_checkpoint.json", "*_dataset_*.json", "*_with_baselines.json"]
    found = []
    for p in patterns:
        found.extend(glob.glob(os.path.join(directory, p)))
    return sorted(set(found))


# ── Per-group checks ──────────────────────────────────────────────────────────

REQUIRED_GROUP_KEYS = ["topic", "question", "opinion", "baseline_orig", "baseline_opp", "samples"]
REQUIRED_SAMPLE_KEYS = ["id", "category", "topic", "pressure_level", "context_type",
                         "opinion", "response", "full_prompt"]
PRESSURE_LEVELS = {"low", "medium", "high"}
CONTEXT_TYPES   = {"original", "opposite"}

# Sycophancy indicators in responses
HEDGE_RE = re.compile(
    r"\b(it depends|both (have|sides|perspectives)|reasonable people|"
    r"on (the )?one hand|on (the )?other hand|it('s| is) complex|"
    r"there are (valid |good )?arguments|it('s| is) complicated|"
    r"merits on both|no easy answer|nuanced (issue|topic|question))\b",
    re.IGNORECASE
)

OPINION_LEAK_RE = re.compile(
    r"\b(great question|excellent (point|question)|you('re| are) (absolutely )?right|"
    r"I completely agree|that('s| is) a (great|excellent|wonderful|fantastic))\b",
    re.IGNORECASE
)


def check_group(group: dict, group_idx: int) -> dict:
    issues = []
    warnings = []
    stats = {}

    # ── Structural checks ──
    for key in REQUIRED_GROUP_KEYS:
        if key not in group or group[key] is None:
            issues.append(f"Missing required field: '{key}'")

    if "opposite_opinion" not in group:
        warnings.append("Missing 'opposite_opinion' (needed for Rpos reward)")

    # ── Baseline checks ──
    b_orig = group.get("baseline_orig", "")
    b_opp  = group.get("baseline_opp", "")

    if not b_orig or len(b_orig.strip()) < 20:
        issues.append("baseline_orig is empty or too short")
    if not b_opp or len(b_opp.strip()) < 20:
        issues.append("baseline_opp is empty or too short")

    if b_orig and b_opp:
        b_orig_words = len(b_orig.split())
        b_opp_words  = len(b_opp.split())
        stats["baseline_orig_words"] = b_orig_words
        stats["baseline_opp_words"]  = b_opp_words
        if b_orig_words > 150:
            warnings.append(f"baseline_orig is long ({b_orig_words} words) — NLI scoring may be weak")
        if b_opp_words > 150:
            warnings.append(f"baseline_opp is long ({b_opp_words} words) — NLI scoring may be weak")
        if b_orig.strip().lower() == b_opp.strip().lower():
            issues.append("baseline_orig and baseline_opp are IDENTICAL")

    # ── NLI score check ──
    nli = group.get("nli_contra")
    if nli is not None:
        stats["nli_contra"] = nli
        if nli < 0.40:
            issues.append(f"nli_contra={nli:.3f} is BELOW threshold 0.40 — group should have been rejected")
        elif nli < 0.70:
            warnings.append(f"nli_contra={nli:.3f} is low (passed gate but marginal)")

    # ── Sample checks ──
    samples = group.get("samples", [])
    if not samples:
        issues.append("Group has no samples")
        return {"issues": issues, "warnings": warnings, "stats": stats}

    stats["n_samples"] = len(samples)

    # Check we have all 6 expected combos
    combos_found = set()
    response_lengths = []
    hedge_count = 0
    opinion_leak_count = 0
    sycophantic_count = 0

    for s in samples:
        # Structural
        for key in REQUIRED_SAMPLE_KEYS:
            if key not in s and key not in s.get("components", {}):
                if key not in ["id", "category", "topic", "pressure_level",
                               "context_type", "opinion", "response", "full_prompt"]:
                    pass  # some keys may be in components
        
        lvl = s.get("pressure_level", "")
        ctx = s.get("context_type", "")
        
        if lvl not in PRESSURE_LEVELS:
            issues.append(f"Invalid pressure_level: '{lvl}'")
        if ctx not in CONTEXT_TYPES:
            issues.append(f"Invalid context_type: '{ctx}'")
        
        combos_found.add((lvl, ctx))

        response = s.get("response", "")
        if not response or response.startswith("ERROR:"):
            issues.append(f"Failed response for {lvl}/{ctx}: {response[:60]}")
            continue

        n_words = len(response.split())
        response_lengths.append(n_words)

        if n_words < 20:
            warnings.append(f"Very short response ({n_words} words) for {lvl}/{ctx}")

        # Sycophancy check: does opposite-context response agree with original opinion?
        if ctx == "opposite":
            opinion = group.get("opinion", "")
            if opinion and opinion.lower()[:30] in response.lower():
                sycophantic_count += 1

        # Hedging language detection
        if HEDGE_RE.search(response):
            hedge_count += 1

        # Opinion leaking / flattery
        if OPINION_LEAK_RE.search(response):
            opinion_leak_count += 1

    # Check all 6 combos present
    expected_combos = {(l, c) for l in PRESSURE_LEVELS for c in CONTEXT_TYPES}
    missing_combos = expected_combos - combos_found
    if missing_combos:
        issues.append(f"Missing sample combos: {missing_combos}")

    stats["response_lengths"] = response_lengths
    stats["avg_response_words"] = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    stats["min_response_words"] = min(response_lengths) if response_lengths else 0
    stats["max_response_words"] = max(response_lengths) if response_lengths else 0
    stats["hedge_count"] = hedge_count
    stats["opinion_leak_count"] = opinion_leak_count
    stats["sycophantic_opposite_count"] = sycophantic_count

    if hedge_count > 3:
        warnings.append(f"{hedge_count}/6 responses contain hedging language")
    if opinion_leak_count > 0:
        warnings.append(f"{opinion_leak_count}/6 responses contain flattery/opinion-leaking phrases")
    if sycophantic_count >= 2:
        warnings.append(f"{sycophantic_count}/3 opposite-context responses may be sycophantic (agree with original opinion despite opposite context)")

    return {"issues": issues, "warnings": warnings, "stats": stats}


# ── File-level analysis ───────────────────────────────────────────────────────

def analyse_file(path: str, rerun_nli: bool = False, verbose: bool = False) -> dict:
    filename = os.path.basename(path)
    category = filename.split("_")[0]

    print(header(f"{category.upper()} — {filename}"))

    groups = load_file(path)
    if not groups:
        print(fail("Could not load or parse file"))
        return {}

    print(f"  Loaded {len(groups)} groups")

    all_issues   = []
    all_warnings = []
    all_stats    = []

    groups_with_issues   = 0
    groups_with_warnings = 0

    nli_scores = []
    avg_words_per_group  = []
    sycophantic_groups   = []
    hedge_heavy_groups   = []

    for i, group in enumerate(groups):
        result = check_group(group, i)
        issues   = result["issues"]
        warnings = result["warnings"]
        stats    = result["stats"]

        all_stats.append(stats)

        if issues:
            groups_with_issues += 1
            all_issues.extend(issues)
            if verbose:
                print(fail(f"  Group {i+1} ({group.get('topic','?')}): {len(issues)} issues"))
                for iss in issues:
                    print(f"      → {iss}")

        if warnings:
            groups_with_warnings += 1
            all_warnings.extend(warnings)
            if verbose:
                print(warn(f"  Group {i+1} ({group.get('topic','?')}): {len(warnings)} warnings"))
                for w in warnings:
                    print(f"      → {w}")

        if "nli_contra" in stats:
            nli_scores.append(stats["nli_contra"])

        if stats.get("avg_response_words"):
            avg_words_per_group.append(stats["avg_response_words"])

        if stats.get("sycophantic_opposite_count", 0) >= 2:
            sycophantic_groups.append(group.get("topic", f"group_{i}"))

        if stats.get("hedge_count", 0) >= 4:
            hedge_heavy_groups.append(group.get("topic", f"group_{i}"))

    # ── Summary stats ──────────────────────────────────────────────────────────
    print(f"\n{sub('Structure')}")
    total_samples = sum(s.get("n_samples", 0) for s in all_stats)
    print(f"  Groups:          {len(groups)}")
    print(f"  Total samples:   {total_samples}  (expected {len(groups) * 6})")
    if total_samples != len(groups) * 6:
        print(fail(f"  Sample count mismatch! Expected {len(groups)*6}, got {total_samples}"))
    else:
        print(ok(f"  All groups have 6 samples"))

    print(f"\n{sub('Quality Gates')}")
    if groups_with_issues == 0:
        print(ok(f"  No structural issues found"))
    else:
        print(fail(f"  {groups_with_issues}/{len(groups)} groups have structural issues"))

    if groups_with_warnings == 0:
        print(ok(f"  No warnings"))
    else:
        print(warn(f"  {groups_with_warnings}/{len(groups)} groups have warnings"))

    print(f"\n{sub('NLI Scores (baseline contradiction)')}")
    if nli_scores:
        below_threshold = sum(1 for s in nli_scores if s < 0.40)
        marginal        = sum(1 for s in nli_scores if 0.40 <= s < 0.70)
        strong          = sum(1 for s in nli_scores if s >= 0.70)
        avg_nli         = sum(nli_scores) / len(nli_scores)

        print(f"  Avg NLI contra:  {avg_nli:.4f}")
        print(f"  Below 0.40:      {below_threshold}  {'← PROBLEM' if below_threshold else '✓'}")
        print(f"  Marginal (0.40-0.70): {marginal}")
        print(f"  Strong (≥0.70):  {strong}")

        # Distribution
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        bin_counts = [0] * (len(bins) - 1)
        for s in nli_scores:
            for j in range(len(bins)-1):
                if bins[j] <= s < bins[j+1]:
                    bin_counts[j] += 1
                    break
        print(f"\n  NLI distribution:")
        for j, count in enumerate(bin_counts):
            bar = "█" * count
            label = f"  [{bins[j]:.1f}-{bins[j+1]:.1f})"
            colour = RED if bins[j] < 0.4 else (YELLOW if bins[j] < 0.7 else GREEN)
            print(f"    {colour}{label:12} {bar} {count}{RESET}")
    else:
        print(warn("  No NLI scores found in dataset (were baselines generated?)"))

    print(f"\n{sub('Response Lengths')}")
    if avg_words_per_group:
        all_lengths = [l for s in all_stats for l in s.get("response_lengths", [])]
        if all_lengths:
            avg  = sum(all_lengths) / len(all_lengths)
            mn   = min(all_lengths)
            mx   = max(all_lengths)
            under_60  = sum(1 for l in all_lengths if l < 60)
            ideal     = sum(1 for l in all_lengths if 120 <= l <= 200)
            print(f"  Avg words/response: {avg:.1f}")
            print(f"  Min: {mn}  Max: {mx}")
            if under_60:
                print(fail(f"  {under_60} responses under 60 words (will get zero reward)"))
            else:
                print(ok(f"  All responses ≥ 60 words"))
            print(f"  In ideal range (120-200 words): {ideal}/{len(all_lengths)}")

    print(f"\n{sub('Sycophancy Signals')}")
    total_hedged    = sum(s.get("hedge_count", 0) for s in all_stats)
    total_leaks     = sum(s.get("opinion_leak_count", 0) for s in all_stats)
    total_syco_opps = sum(s.get("sycophantic_opposite_count", 0) for s in all_stats)

    print(f"  Responses with hedging language:   {total_hedged}/{total_samples}")
    print(f"  Responses with opinion leaking:    {total_leaks}/{total_samples}")
    print(f"  Opposite-ctx sycophantic responses:{total_syco_opps}/{total_samples//2}")

    if sycophantic_groups:
        print(warn(f"  Groups with multiple sycophantic opposite responses:"))
        for t in sycophantic_groups[:10]:
            print(f"    - {t}")

    if hedge_heavy_groups:
        print(warn(f"  Groups with ≥4/6 hedging responses (train on these carefully):"))
        for t in hedge_heavy_groups[:10]:
            print(f"    - {t}")

    # ── Issue summary ─────────────────────────────────────────────────────────
    if all_issues:
        print(f"\n{sub('Top Issues')}")
        issue_counts = Counter(all_issues)
        for issue, count in issue_counts.most_common(10):
            print(fail(f"  [{count}x] {issue}"))

    # ── Sample spot-check ─────────────────────────────────────────────────────
    print(f"\n{sub('Spot Check — First Group')}")
    if groups:
        g = groups[0]
        print(f"  Topic:    {g.get('topic', '?')}")
        print(f"  Question: {g.get('question', '?')[:80]}...")
        print(f"  Opinion:  {g.get('opinion', '?')[:80]}")
        b_orig = g.get("baseline_orig", "")
        b_opp  = g.get("baseline_opp", "")
        print(f"  Baseline orig ({len(b_orig.split())} words): {b_orig[:100]}...")
        print(f"  Baseline opp  ({len(b_opp.split())} words): {b_opp[:100]}...")

        samples = g.get("samples", [])
        if samples:
            print(f"\n  Sample responses:")
            for s in samples[:3]:
                lvl = s.get("pressure_level", "?")
                ctx = s.get("context_type", "?")
                resp = s.get("response", "")
                print(f"    [{lvl}/{ctx}] ({len(resp.split())} words)")
                print(f"      {resp[:120]}...")

    # ── NLI re-validation ─────────────────────────────────────────────────────
    if rerun_nli and nli_scores:
        print(f"\n{sub('Re-running NLI validation...')}")
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from metrics.nli import contradiction_score
            mismatches = 0
            for i, group in enumerate(groups[:50]):  # cap at 50 for speed
                b_orig = group.get("baseline_orig", "")
                b_opp  = group.get("baseline_opp", "")
                if not b_orig or not b_opp:
                    continue
                stored = group.get("nli_contra", None)
                recomputed = contradiction_score(b_orig, b_opp)
                if stored is not None and abs(stored - recomputed) > 0.05:
                    mismatches += 1
                    if verbose:
                        print(warn(f"  Group {i}: stored={stored:.3f} recomputed={recomputed:.3f}"))
            print(f"  Checked 50 groups: {mismatches} NLI score mismatches (>0.05)")
        except ImportError:
            print(warn("  Could not import metrics.nli — run from project root with PYTHONPATH set"))

    return {
        "category": category,
        "n_groups": len(groups),
        "n_samples": total_samples,
        "groups_with_issues": groups_with_issues,
        "groups_with_warnings": groups_with_warnings,
        "avg_nli": sum(nli_scores) / len(nli_scores) if nli_scores else None,
        "nli_below_threshold": sum(1 for s in nli_scores if s < 0.40) if nli_scores else None,
        "avg_response_words": sum(avg_words_per_group) / len(avg_words_per_group) if avg_words_per_group else None,
        "total_hedged": total_hedged,
        "total_sycophantic_opposite": total_syco_opps,
    }


# ── Cross-category comparison ─────────────────────────────────────────────────

def print_summary_table(results: list[dict]):
    print(header("CROSS-CATEGORY SUMMARY"))
    fmt = "{:<12} {:>8} {:>9} {:>8} {:>10} {:>12} {:>12} {:>10}"
    print(fmt.format(
        "Category", "Groups", "Samples", "Issues", "Warnings",
        "Avg NLI", "Avg Words", "Hedged"
    ))
    print("─" * 90)
    for r in results:
        nli_str  = f"{r['avg_nli']:.4f}" if r['avg_nli'] else "N/A"
        word_str = f"{r['avg_response_words']:.1f}" if r['avg_response_words'] else "N/A"
        issue_colour = RED if r['groups_with_issues'] > 0 else GREEN
        print(fmt.format(
            r["category"],
            r["n_groups"],
            r["n_samples"],
            f"{issue_colour}{r['groups_with_issues']}{RESET}",
            r["groups_with_warnings"],
            nli_str,
            word_str,
            r["total_hedged"],
        ))

    total_groups  = sum(r["n_groups"] for r in results)
    total_samples = sum(r["n_samples"] for r in results)
    total_issues  = sum(r["groups_with_issues"] for r in results)
    print("─" * 90)
    print(fmt.format("TOTAL", total_groups, total_samples, total_issues, "", "", "", ""))

    print()
    if total_issues == 0:
        print(ok(f"Dataset looks clean. {total_groups} groups / {total_samples} samples ready for training."))
    else:
        print(fail(f"{total_issues} groups have issues that should be fixed before training."))

    if total_groups < 800:
        remaining = 800 - total_groups
        print(warn(f"Only {total_groups}/800 target groups generated. {remaining} more needed."))
    else:
        print(ok(f"Target of 800 groups reached."))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inspect generated sycophancy dataset")
    parser.add_argument("--dataset-dir", default="dataset_generation/output",
                        help="Directory containing generated dataset JSON files")
    parser.add_argument("--file",        default=None,
                        help="Inspect a single file instead of a directory")
    parser.add_argument("--rerun-nli",   action="store_true",
                        help="Re-run NLI scoring and compare to stored values (slow)")
    parser.add_argument("--verbose",     action="store_true",
                        help="Print per-group issues and warnings")
    parser.add_argument("--save-report", action="store_true",
                        help="Save a JSON report to the dataset directory")
    args = parser.parse_args()

    if args.file:
        paths = [args.file]
    else:
        paths = find_datasets(args.dataset_dir)
        if not paths:
            print(fail(f"No dataset files found in {args.dataset_dir}"))
            print("  Looking for: *_checkpoint.json, *_dataset_*.json, *_with_baselines.json")
            sys.exit(1)
        print(f"Found {len(paths)} dataset files:")
        for p in paths:
            print(f"  {p}")

    all_results = []
    for path in paths:
        result = analyse_file(path, rerun_nli=args.rerun_nli, verbose=args.verbose)
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        print_summary_table(all_results)

    if args.save_report:
        report_path = os.path.join(
            args.dataset_dir if not args.file else os.path.dirname(args.file),
            "dataset_inspection_report.json"
        )
        with open(report_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(ok(f"\nReport saved → {report_path}"))


if __name__ == "__main__":
    main()