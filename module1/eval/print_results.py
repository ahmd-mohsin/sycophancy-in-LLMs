"""
Print Module 1 evaluation results.

Shows for each sample:
  1. Original sycophantic prompt (user_message + question)
  2. LLM's response to clean vs biased prompt
  3. Module 1's three-tier detection breakdown
  4. LLM judge output with score and raw reasoning

Usage:
    python -m module1.eval.print_results --input self_curated_eval_results.json --show-shifts
    python -m module1.eval.print_results --input self_curated_eval_results.json --show-all
    python -m module1.eval.print_results --input self_curated_eval_results.json --show-all --dataset claude
    python -m module1.eval.print_results --input self_curated_eval_results.json --latex-summary
"""

import json
import argparse
import textwrap


def load_results(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def wrap(text: str, width: int = 90, indent: str = "    ") -> str:
    lines = textwrap.wrap(text, width=width)
    return ("\n" + indent).join(lines)


def print_text_format(results: dict, only_shifts: bool, dataset_filter: str):
    for ds_name, ds_data in results.items():
        if ds_name == "aggregate":
            continue
        if dataset_filter and ds_name != dataset_filter:
            continue

        samples = ds_data.get("results", [])
        print(f"\n{'='*90}")
        print(f"DATASET: {ds_name.upper()} ({len(samples)} samples)")
        print(f"{'='*90}")

        for i, r in enumerate(samples):
            shift = r.get("shift_degree_final", "none")
            if only_shifts and shift == "none":
                continue

            sid = r.get("id", f"#{i+1}")
            syc_type = r.get("sycophancy_type", "?")
            q_type = r.get("question_type_dataset", r.get("question_type", "?"))
            clean_ans = r.get("clean_answer", "?")
            biased_ans = r.get("biased_answer", "?")
            gt = r.get("ground_truth", "?")
            clean_correct = r.get("clean_correct")
            biased_correct = r.get("biased_correct")

            m1 = r.get("m1_detection", {})
            score = m1.get("sycophancy_score", 0)
            risk = m1.get("sycophancy_risk", "?")
            detected = m1.get("detected_types", [])
            components = m1.get("score_components", {})

            ej = r.get("eval_judge", r.get("judge", {}))

            print(f"\n{'─'*90}")
            print(f"[{sid}]  type={syc_type}  q_type={q_type}  shift={shift.upper()}")
            print(f"{'─'*90}")

            # 1. ORIGINAL SYCOPHANTIC PROMPT
            user_msg = r.get("user_message", "")
            biased_prompt = r.get("biased_prompt", "")
            if user_msg:
                print(f"\n  ▶ SYCOPHANTIC PROMPT:")
                print(f"    User message: {wrap(user_msg)}")
            if biased_prompt and biased_prompt != user_msg:
                clean_p = r.get("clean_prompt", "")
                if clean_p:
                    print(f"    Clean question: {wrap(clean_p)}")

            # 2. LLM RESPONSES
            print(f"\n  ▶ LLM RESPONSES:")
            clean_resp = r.get("clean_response", "")[:300]
            biased_resp = r.get("biased_response", "")[:300]
            cc = "✓" if clean_correct else "✗" if clean_correct is not None else "?"
            bc = "✓" if biased_correct else "✗" if biased_correct is not None else "?"
            print(f"    Clean  → {clean_ans} {cc}  GT={gt}")
            print(f"    {wrap(clean_resp)}")
            print(f"    Biased → {biased_ans} {bc}  GT={gt}")
            print(f"    {wrap(biased_resp)}")

            # 3. MODULE 1 THREE-TIER DETECTION
            print(f"\n  ▶ MODULE 1 DETECTION (score={score:.3f}, risk={risk}):")
            src = m1.get("detection_source", "?")
            print(f"    Source: {src}")

            # Tier 1
            t1_kw = m1.get("tier1_keywords", [])
            if t1_kw:
                print(f"    Tier 1 (keywords): {', '.join(t1_kw)}")
            else:
                print(f"    Tier 1 (keywords): none")

            # Tier 2
            t2_os = m1.get("tier2_opinion_steering", [])
            t2_ap = m1.get("tier2_authority_pressure", [])
            t2_ef = m1.get("tier2_emotional_framing", [])
            if t2_os or t2_ap or t2_ef:
                print(f"    Tier 2 (regex):")
                for m_item in t2_os:
                    print(f"      opinion_steering/{m_item.get('pattern','?')}: \"{m_item.get('matched','')}\"")
                for m_item in t2_ap:
                    print(f"      authority/{m_item.get('pattern','?')}: \"{m_item.get('matched','')}\"")
                for m_item in t2_ef:
                    print(f"      emotional/{m_item.get('pattern','?')}: \"{m_item.get('matched','')}\"")
            else:
                print(f"    Tier 2 (regex): no patterns matched")

            # Tier 3
            t3 = m1.get("tier3_llm_judge", {})
            t3_detected = t3.get("sycophancy_detected", False)
            if t3_detected or t3.get("raw_output"):
                print(f"    Tier 3 (LLM judge): detected={t3_detected}, risk={t3.get('risk_level','?')}")
                flags = []
                if t3.get("opinion_steering"): flags.append("opinion_steering")
                if t3.get("authority_pressure"): flags.append("authority_pressure")
                if t3.get("identity_framing"): flags.append("identity_framing")
                if t3.get("emotional_pressure"): flags.append("emotional_pressure")
                if t3.get("explicit_manipulation"): flags.append("explicit_manipulation")
                if t3.get("biographical_priming"): flags.append("biographical_priming")
                if flags:
                    print(f"      Flags: {', '.join(flags)}")
                if t3.get("reason"):
                    print(f"      Reason: {t3['reason']}")
            else:
                print(f"    Tier 3 (LLM judge): not available")

            # Score components
            if components:
                nonzero = {k: v for k, v in components.items() if v and v != 1.0}
                if nonzero:
                    parts = [f"{k}={v}" for k, v in nonzero.items()]
                    print(f"    Score breakdown: {', '.join(parts)}")

            if detected:
                print(f"    Detected types: {', '.join(detected)}")

            # 4. EVAL LLM JUDGE
            print(f"\n  ▶ EVAL JUDGE:")
            print(f"    Shift: {ej.get('shift_degree', '?')}  "
                  f"Severity: {ej.get('pressure_severity', '?')}/5  "
                  f"Factual accuracy: {ej.get('factual_accuracy', '?')}")
            if ej.get("reasoning"):
                print(f"    Reasoning: {wrap(ej['reasoning'])}")
            raw = ej.get("raw_output", "")
            if raw:
                print(f"    Raw output:")
                for line in raw[:400].split("\n"):
                    print(f"      {line}")


def print_summary_latex(results: dict):
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Module 1 Sycophancy Detection — Self-Curated Datasets}")
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
        sc = s.get("shift_counts", {})
        name_tex = ds_name.capitalize()
        print(f"  {name_tex} & {n} & {sc.get('complete',0)} & {sc.get('partial',0)} & "
              f"{sc.get('none',0)} & {s.get('valid_rate',0)}\\% & "
              f"{s.get('avg_sycophancy_score',0):.3f} & "
              f"{s.get('accuracy_drop',0):+.1f}\\% \\\\")

    if "aggregate" in results:
        s = results["aggregate"].get("summary", {})
        sc = s.get("shift_counts", {})
        print(r"\hline")
        print(f"  \\textbf{{Aggregate}} & {s.get('total',0)} & {sc.get('complete',0)} & "
              f"{sc.get('partial',0)} & {sc.get('none',0)} & "
              f"{s.get('valid_rate',0)}\\% & {s.get('avg_sycophancy_score',0):.3f} & "
              f"{s.get('accuracy_drop',0):+.1f}\\% \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Per-type table
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

    # Detection tier table
    print()
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Three-Tier Detection Rates}")
    print(r"\label{tab:m1-detection-tiers}")
    print(r"\begin{tabular}{l|r|r|r}")
    print(r"\hline")
    print(r"\textbf{Dataset} & \textbf{Tier 1 (Keywords)} & \textbf{Tier 2 (Regex)} & \textbf{Tier 3 (LLM)} \\")
    print(r"\hline")

    for ds_name, ds_data in results.items():
        if ds_name == "aggregate":
            continue
        s = ds_data.get("summary", {})
        dr = s.get("detection_rates", {})
        name_tex = ds_name.capitalize()
        print(f"  {name_tex} & {dr.get('tier1_keyword',0)}\\% & "
              f"{dr.get('tier2_regex',0)}\\% & {dr.get('tier3_llm_judge',0)}\\% \\\\")

    if "aggregate" in results:
        dr = results["aggregate"].get("summary", {}).get("detection_rates", {})
        print(r"\hline")
        print(f"  \\textbf{{Aggregate}} & {dr.get('tier1_keyword',0)}\\% & "
              f"{dr.get('tier2_regex',0)}\\% & {dr.get('tier3_llm_judge',0)}\\% \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--show-shifts", action="store_true")
    parser.add_argument("--show-all", action="store_true")
    parser.add_argument("--latex-summary", action="store_true")
    args = parser.parse_args()

    results = load_results(args.input)

    if args.latex_summary:
        print_summary_latex(results)
    elif args.show_shifts:
        print_text_format(results, only_shifts=True, dataset_filter=args.dataset)
    elif args.show_all:
        print_text_format(results, only_shifts=False, dataset_filter=args.dataset)
    else:
        print_text_format(results, only_shifts=True, dataset_filter=args.dataset)


if __name__ == "__main__":
    main()