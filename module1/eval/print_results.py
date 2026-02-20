"""
Print Module 1 evaluation results.

For each sample shows:
  1. Original sycophantic prompt (user_message)
  2. Module 1 three-tier detection (keywords, regex matches, LLM judge raw output)
  3. Sycophancy score + risk level + detected pressure types
  4. Extracted unbiased core claim

Usage:
    python -m module1.eval.print_results --input m1_eval_results.json --show-all
    python -m module1.eval.print_results --input m1_eval_results.json --dataset claude
    python -m module1.eval.print_results --input m1_eval_results.json --latex-summary
"""

import json
import argparse
import textwrap


def load_results(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def wrap(text: str, width: int = 88, indent: str = "    ") -> str:
    lines = textwrap.wrap(text, width=width)
    return ("\n" + indent).join(lines)


def print_text_format(results: dict, dataset_filter: str):
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
            sid = r.get("id", f"#{i+1}")
            syc_type = r.get("sycophancy_type", "?")
            q_type_ds = r.get("question_type_dataset", "?")
            q_type_m1 = r.get("question_type_classified", "?")
            d = r.get("detection", {})
            score = d.get("sycophancy_score", 0)
            risk = d.get("sycophancy_risk", "?")
            detected = d.get("detected_types", [])

            print(f"\n{'─'*90}")
            print(f"[{sid}]  syc_type={syc_type}  q_type={q_type_ds}  "
                  f"classified={q_type_m1}  score={score:.3f}  risk={risk}")
            print(f"{'─'*90}")

            # 1. ORIGINAL SYCOPHANTIC PROMPT
            user_msg = r.get("user_message", "")
            question = r.get("question", "")
            print(f"\n  ▶ SYCOPHANTIC PROMPT:")
            print(f"    {wrap(user_msg)}")
            if question:
                print(f"    Question: {wrap(question)}")

            # 2. MODULE 1 — THREE-TIER DETECTION
            print(f"\n  ▶ DETECTION (source={d.get('source', '?')}):")

            # Tier 1
            t1 = d.get("tier1", {})
            t1_kw = t1.get("keywords_found", [])
            if t1_kw:
                print(f"    Tier 1 (keywords): {', '.join(t1_kw)}  "
                      f"density={t1.get('bias_density', 0):.3f}")
            else:
                print(f"    Tier 1 (keywords): none")

            # Tier 2
            t2 = d.get("tier2", {})
            t2_os = t2.get("opinion_steering", [])
            t2_ap = t2.get("authority_pressure", [])
            t2_ef = t2.get("emotional_framing", [])
            has_t2 = t2_os or t2_ap or t2_ef
            if has_t2:
                print(f"    Tier 2 (regex):")
                for m in t2_os:
                    print(f"      [opinion] {m.get('pattern','?')}: \"{m.get('matched','')}\"")
                for m in t2_ap:
                    print(f"      [authority] {m.get('pattern','?')}: \"{m.get('matched','')}\"")
                for m in t2_ef:
                    print(f"      [emotional] {m.get('pattern','?')}: \"{m.get('matched','')}\"")
            else:
                print(f"    Tier 2 (regex): no matches")

            # Tier 3
            t3 = d.get("tier3", {})
            t3_det = t3.get("sycophancy_detected", False)
            print(f"    Tier 3 (LLM judge): detected={t3_det}  risk={t3.get('risk_level', '?')}")
            flags = []
            for flag in ["opinion_steering", "authority_pressure", "identity_framing",
                         "emotional_pressure", "explicit_manipulation", "biographical_priming"]:
                if t3.get(flag):
                    flags.append(flag)
            if flags:
                print(f"      Flags: {', '.join(flags)}")
            if t3.get("reason"):
                print(f"      Reason: {t3['reason']}")
            raw = t3.get("raw_output", "")
            if raw:
                print(f"      Raw LLM output:")
                for line in raw[:500].split("\n"):
                    line = line.strip()
                    if line:
                        print(f"        {line}")

            # Score components
            components = d.get("score_components", {})
            nonzero = {k: v for k, v in components.items() if v and v != 1.0}
            if nonzero:
                parts = [f"{k}={v}" for k, v in nonzero.items()]
                print(f"    Score breakdown: {', '.join(parts)}")

            if detected:
                print(f"    All detected types: {', '.join(detected)}")

            # 3. EXTRACTED CORE CLAIM
            cc = r.get("core_claim", {})
            if cc.get("was_modified"):
                print(f"\n  ▶ CORE CLAIM EXTRACTION:")
                print(f"    Original:    {wrap(cc.get('original', '')[:300])}")
                print(f"    Neutralized: {wrap(cc.get('neutralized', '')[:300])}")
            else:
                print(f"\n  ▶ CORE CLAIM: no modification needed (no bias detected in user_message)")
                neutralized_full = cc.get("full_prompt_neutralized", "")
                if neutralized_full:
                    print(f"    Full prompt neutralized: {wrap(neutralized_full[:300])}")


def print_summary_latex(results: dict):
    # Main results table
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Module 1 Sycophancy Detection — Self-Curated Datasets}")
    print(r"\label{tab:m1-detection}")
    print(r"\begin{tabular}{l|r|r|r|r|r|r}")
    print(r"\hline")
    print(r"\textbf{Dataset} & \textbf{N} & \textbf{Avg Score} & "
          r"\textbf{Tier 1} & \textbf{Tier 2} & \textbf{Tier 3} & \textbf{Any} \\")
    print(r"\hline")

    for ds_name, ds_data in results.items():
        if ds_name == "aggregate":
            continue
        s = ds_data.get("summary", {})
        dr = s.get("detection_rates", {})
        name_tex = ds_name.capitalize()
        print(f"  {name_tex} & {s.get('total',0)} & {s.get('avg_sycophancy_score',0):.3f} & "
              f"{dr.get('tier1_keyword',0)}\\% & {dr.get('tier2_regex',0)}\\% & "
              f"{dr.get('tier3_llm_judge',0)}\\% & {dr.get('any_tier',0)}\\% \\\\")

    if "aggregate" in results:
        s = results["aggregate"].get("summary", {})
        dr = s.get("detection_rates", {})
        print(r"\hline")
        print(f"  \\textbf{{Aggregate}} & {s.get('total',0)} & "
              f"{s.get('avg_sycophancy_score',0):.3f} & "
              f"{dr.get('tier1_keyword',0)}\\% & {dr.get('tier2_regex',0)}\\% & "
              f"{dr.get('tier3_llm_judge',0)}\\% & {dr.get('any_tier',0)}\\% \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Per sycophancy type
    print()
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Detection Rate by Sycophancy Type (Aggregate)}")
    print(r"\label{tab:m1-by-type}")
    print(r"\begin{tabular}{l|r|r|r}")
    print(r"\hline")
    print(r"\textbf{Sycophancy Type} & \textbf{Samples} & \textbf{Detection Rate} & \textbf{Avg Score} \\")
    print(r"\hline")

    if "aggregate" in results:
        by_type = results["aggregate"].get("summary", {}).get("by_sycophancy_type", {})
        for syc_type, stats in sorted(by_type.items(), key=lambda x: x[1]["avg_score"], reverse=True):
            name_tex = syc_type.replace("_", " ").replace("-", " ").title()
            print(f"  {name_tex} & {stats['total']} & {stats['detection_rate']}\\% & "
                  f"{stats['avg_score']:.3f} \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Risk distribution
    print()
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Risk Level Distribution}")
    print(r"\label{tab:m1-risk}")
    print(r"\begin{tabular}{l|r|r|r|r}")
    print(r"\hline")
    print(r"\textbf{Dataset} & \textbf{None} & \textbf{Low} & \textbf{Medium} & \textbf{High} \\")
    print(r"\hline")

    for ds_name, ds_data in results.items():
        if ds_name == "aggregate":
            continue
        s = ds_data.get("summary", {})
        rd = s.get("risk_distribution", {})
        name_tex = ds_name.capitalize()
        print(f"  {name_tex} & {rd.get('none',0)} & {rd.get('low',0)} & "
              f"{rd.get('medium',0)} & {rd.get('high',0)} \\\\")

    if "aggregate" in results:
        rd = results["aggregate"].get("summary", {}).get("risk_distribution", {})
        print(r"\hline")
        print(f"  \\textbf{{Aggregate}} & {rd.get('none',0)} & {rd.get('low',0)} & "
              f"{rd.get('medium',0)} & {rd.get('high',0)} \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--show-all", action="store_true")
    parser.add_argument("--latex-summary", action="store_true")
    args = parser.parse_args()

    results = load_results(args.input)

    if args.latex_summary:
        print_summary_latex(results)
    else:
        print_text_format(results, dataset_filter=args.dataset)


if __name__ == "__main__":
    main()