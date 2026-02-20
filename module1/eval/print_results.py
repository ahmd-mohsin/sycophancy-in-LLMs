"""
Print evaluation results for LaTeX inclusion.

Usage:
    python -m module1.eval.print_results \
        --input self_curated_eval_results.json \
        --show-shifts          # only show samples where shift occurred
        --show-all             # show all samples
        --dataset claude       # filter to one dataset (optional)
        --latex                # output LaTeX table format
"""

import json
import argparse
import textwrap


def load_results(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def print_text_format(results: dict, only_shifts: bool, dataset_filter: str):
    for ds_name, ds_data in results.items():
        if ds_name == "aggregate":
            continue
        if dataset_filter and ds_name != dataset_filter:
            continue

        samples = ds_data.get("results", [])
        print(f"\n{'='*80}")
        print(f"DATASET: {ds_name.upper()} ({len(samples)} samples)")
        print(f"{'='*80}")

        for i, r in enumerate(samples):
            shift = r.get("shift_degree_final", "none")
            if only_shifts and shift == "none":
                continue

            sid = r.get("id", f"#{i+1}")
            syc_type = r.get("sycophancy_type", "?")
            q_type = r.get("question_type", "?")
            clean_ans = r.get("clean_answer", "?")
            biased_ans = r.get("biased_answer", "?")
            gt = r.get("ground_truth", "?")
            clean_correct = r.get("clean_correct")
            biased_correct = r.get("biased_correct")
            m1 = r.get("m1_detection", {})
            score = m1.get("sycophancy_score", 0)
            risk = m1.get("sycophancy_risk", "?")
            detected = m1.get("detected_types", [])
            judge = r.get("judge", {})
            judge_shift = judge.get("shift_degree", "?")
            severity = judge.get("pressure_severity", "?")
            reasoning = judge.get("reasoning", "")

            clean_resp = r.get("clean_response", "")[:200]
            biased_resp = r.get("biased_response", "")[:200]

            print(f"\n--- [{sid}] {syc_type} | {q_type} | shift={shift} ---")
            print(f"  Clean answer:   {clean_ans} {'✓' if clean_correct else '✗' if clean_correct is not None else ''}")
            print(f"  Biased answer:  {biased_ans} {'✓' if biased_correct else '✗' if biased_correct is not None else ''}")
            print(f"  Ground truth:   {gt}")
            print(f"  Shift degree:   {shift} (judge: {judge_shift})")
            print(f"  Syc score:      {score} | risk: {risk} | severity: {severity}/5")
            print(f"  Detected types: {', '.join(detected) if detected else 'none'}")
            print(f"  Clean response: {clean_resp}...")
            print(f"  Biased response:{biased_resp}...")
            if reasoning:
                print(f"  Judge reasoning: {reasoning}")


def print_latex_table(results: dict, only_shifts: bool, dataset_filter: str):
    """Output a LaTeX longtable with key columns."""

    print(r"\begin{longtable}{|c|l|l|c|c|c|c|c|}")
    print(r"\hline")
    print(r"\textbf{ID} & \textbf{Syc. Type} & \textbf{Q. Type} & "
          r"\textbf{Clean} & \textbf{Biased} & \textbf{GT} & "
          r"\textbf{Shift} & \textbf{Score} \\")
    print(r"\hline")
    print(r"\endfirsthead")
    print(r"\hline")
    print(r"\textbf{ID} & \textbf{Syc. Type} & \textbf{Q. Type} & "
          r"\textbf{Clean} & \textbf{Biased} & \textbf{GT} & "
          r"\textbf{Shift} & \textbf{Score} \\")
    print(r"\hline")
    print(r"\endhead")

    for ds_name, ds_data in results.items():
        if ds_name == "aggregate":
            continue
        if dataset_filter and ds_name != dataset_filter:
            continue

        samples = ds_data.get("results", [])

        # Dataset header row
        print(r"\multicolumn{8}{|c|}{\textbf{" + ds_name.upper() + r"}} \\")
        print(r"\hline")

        for i, r in enumerate(samples):
            shift = r.get("shift_degree_final", "none")
            if only_shifts and shift == "none":
                continue

            sid = r.get("id", f"{i+1}").replace("_", r"\_")
            syc_type = r.get("sycophancy_type", "?").replace("_", r"\_").replace("-", r"-")
            q_type = r.get("question_type", "?").replace("_", r"\_")
            clean_ans = r.get("clean_answer", "?")
            biased_ans = r.get("biased_answer", "?")
            gt = r.get("ground_truth", "?")
            score = r.get("m1_detection", {}).get("sycophancy_score", 0)

            # Color shift column
            if shift == "complete":
                shift_cell = r"\cellcolor{red!25}" + "Complete"
            elif shift == "partial":
                shift_cell = r"\cellcolor{yellow!25}" + "Partial"
            else:
                shift_cell = r"\cellcolor{green!25}" + "None"

            print(f"  {sid} & {syc_type} & {q_type} & {clean_ans} & "
                  f"{biased_ans} & {gt} & {shift_cell} & {score:.3f} \\\\")

        print(r"\hline")

    print(r"\end{longtable}")


def print_shift_examples_latex(results: dict, dataset_filter: str, max_examples: int = 10):
    """Print detailed shift examples for LaTeX — shows the actual responses."""

    print(r"% Shift examples for qualitative analysis")
    print(r"\begin{enumerate}")

    count = 0
    for ds_name, ds_data in results.items():
        if ds_name == "aggregate":
            continue
        if dataset_filter and ds_name != dataset_filter:
            continue

        for r in ds_data.get("results", []):
            shift = r.get("shift_degree_final", "none")
            if shift == "none":
                continue
            if count >= max_examples:
                break

            sid = r.get("id", "?")
            syc_type = r.get("sycophancy_type", "?")
            clean_ans = r.get("clean_answer", "?")
            biased_ans = r.get("biased_answer", "?")
            gt = r.get("ground_truth", "?")
            score = r.get("m1_detection", {}).get("sycophancy_score", 0)
            detected = r.get("m1_detection", {}).get("detected_types", [])

            clean_resp = r.get("clean_response", "")[:300].replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
            biased_resp = r.get("biased_response", "")[:300].replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")

            print(f"  \\item \\textbf{{{sid}}} ({syc_type}, shift=\\textbf{{{shift}}}, "
                  f"score={score:.3f}) \\\\")
            print(f"  Detected: {', '.join(detected) if detected else 'none'} \\\\")
            print(f"  Clean={clean_ans}, Biased={biased_ans}, GT={gt} \\\\")
            print(f"  \\textit{{Clean response:}} {clean_resp}... \\\\")
            print(f"  \\textit{{Biased response:}} {biased_resp}...")

            count += 1

    print(r"\end{enumerate}")


def print_summary_latex(results: dict):
    """Print the summary table for all datasets."""

    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Module 1 Sycophancy Detection Results on Self-Curated Datasets}")
    print(r"\label{tab:m1-self-curated}")
    print(r"\begin{tabular}{l|r|r|r|r|r|r|r}")
    print(r"\hline")
    print(r"\textbf{Dataset} & \textbf{N} & \textbf{Complete} & \textbf{Partial} & "
          r"\textbf{None} & \textbf{Shift\%} & \textbf{Avg Score} & \textbf{Acc Drop} \\")
    print(r"\hline")

    for ds_name, ds_data in results.items():
        if ds_name == "aggregate":
            continue
        s = ds_data.get("summary", {})
        n = s.get("total", 0)
        complete = s.get("shift_counts", {}).get("complete", 0)
        partial = s.get("shift_counts", {}).get("partial", 0)
        none_count = s.get("shift_counts", {}).get("none", 0)
        valid_rate = s.get("valid_rate", 0)
        avg_score = s.get("avg_sycophancy_score", 0)
        acc_drop = s.get("accuracy_drop", 0)

        name_tex = ds_name.capitalize()
        print(f"  {name_tex} & {n} & {complete} & {partial} & {none_count} & "
              f"{valid_rate}\\% & {avg_score:.3f} & {acc_drop:+.1f}\\% \\\\")

    # Aggregate
    if "aggregate" in results:
        s = results["aggregate"].get("summary", {})
        n = s.get("total", 0)
        complete = s.get("shift_counts", {}).get("complete", 0)
        partial = s.get("shift_counts", {}).get("partial", 0)
        none_count = s.get("shift_counts", {}).get("none", 0)
        valid_rate = s.get("valid_rate", 0)
        avg_score = s.get("avg_sycophancy_score", 0)
        acc_drop = s.get("accuracy_drop", 0)

        print(r"\hline")
        print(f"  \\textbf{{Aggregate}} & {n} & {complete} & {partial} & {none_count} & "
              f"{valid_rate}\\% & {avg_score:.3f} & {acc_drop:+.1f}\\% \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Per-sycophancy-type table
    print()
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Shift Rate by Sycophancy Type (Aggregate)}")
    print(r"\label{tab:m1-by-type}")
    print(r"\begin{tabular}{l|r|r|r}")
    print(r"\hline")
    print(r"\textbf{Sycophancy Type} & \textbf{Samples} & \textbf{Shift Rate} & \textbf{Avg Score} \\")
    print(r"\hline")

    if "aggregate" in results:
        by_type = results["aggregate"].get("summary", {}).get("by_sycophancy_type", {})
        for syc_type, stats in sorted(by_type.items(), key=lambda x: x[1]["shift_rate"], reverse=True):
            name_tex = syc_type.replace("_", " ").replace("-", " ").title()
            print(f"  {name_tex} & {stats['total']} & {stats['shift_rate']}\\% & {stats['avg_score']:.3f} \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to eval results JSON")
    parser.add_argument("--dataset", default="", help="Filter to one dataset (claude/chatgpt/perplexity)")
    parser.add_argument("--show-shifts", action="store_true", help="Only show samples with shifts")
    parser.add_argument("--show-all", action="store_true", help="Show all samples")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX table")
    parser.add_argument("--latex-examples", action="store_true", help="Output LaTeX shift examples")
    parser.add_argument("--latex-summary", action="store_true", help="Output LaTeX summary tables")
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()

    results = load_results(args.input)

    if args.latex_summary:
        print_summary_latex(results)
    elif args.latex_examples:
        print_shift_examples_latex(results, args.dataset, args.max_examples)
    elif args.latex:
        print_latex_table(results, args.show_shifts, args.dataset)
    elif args.show_shifts:
        print_text_format(results, only_shifts=True, dataset_filter=args.dataset)
    elif args.show_all:
        print_text_format(results, only_shifts=False, dataset_filter=args.dataset)
    else:
        # Default: show shifts only
        print_text_format(results, only_shifts=True, dataset_filter=args.dataset)


if __name__ == "__main__":
    main()