"""
Extract evaluation results from the new opinion-shift eval and generate:
  1. LaTeX tables for the paper
  2. Summary statistics
  3. Judge prompts

Usage:
    python extract_results.py --input my_eval_results.json
"""

import json
import argparse
import os


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def compute_stats(results_dict):
    stats = {}

    for name, data in results_dict.items():
        if name == "aggregate":
            continue

        results = data.get("results", [])
        total = len(results)
        if total == 0:
            continue

        shifts = 0
        no_shifts = 0
        unparsed = 0
        scores_shift = []
        scores_no_shift = []
        all_scores = []
        m1_detected = 0
        avg_syc_score = 0.0
        risk_dist = {"high": 0, "medium": 0, "low": 0, "none": 0}
        all_types = []

        for r in results:
            outcome = r.get("outcome", "")
            score = r.get("judge", {}).get("score")

            if outcome == "shift_detected":
                shifts += 1
                if score: scores_shift.append(score)
            elif outcome == "no_shift":
                no_shifts += 1
                if score: scores_no_shift.append(score)
            else:
                unparsed += 1

            if score: all_scores.append(score)

            m1 = r.get("m1_detection", {})
            if m1.get("bias_detected"): m1_detected += 1
            avg_syc_score += m1.get("sycophancy_score", 0)
            risk = m1.get("sycophancy_risk", "none")
            if risk in risk_dist: risk_dist[risk] += 1
            all_types.extend(m1.get("detected_types", []))

        parseable = shifts + no_shifts
        type_counts = {}
        for t in all_types:
            type_counts[t] = type_counts.get(t, 0) + 1

        stats[name] = {
            "total": total, "parseable": parseable, "unparsed": unparsed,
            "shifts": shifts,
            "shift_rate": round(shifts / max(parseable, 1) * 100, 1),
            "no_shifts": no_shifts,
            "no_shift_rate": round(no_shifts / max(parseable, 1) * 100, 1),
            "avg_score_shift": round(sum(scores_shift) / max(len(scores_shift), 1), 2),
            "avg_score_no_shift": round(sum(scores_no_shift) / max(len(scores_no_shift), 1), 2),
            "avg_score_overall": round(sum(all_scores) / max(len(all_scores), 1), 2),
            "m1_detected": m1_detected,
            "m1_detection_rate": round(m1_detected / max(total, 1) * 100, 1),
            "avg_syc_score": round(avg_syc_score / max(total, 1), 3),
            "risk_dist": risk_dist,
            "pressure_types": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
        }

    # Aggregate
    if stats:
        agg = {k: 0 for k in ["total", "parseable", "unparsed", "shifts", "no_shifts"]}
        agg_shift_s, agg_noshift_s, agg_all_s = [], [], []
        for s in stats.values():
            for k in agg: agg[k] += s[k]
        for name, data in results_dict.items():
            if name == "aggregate": continue
            for r in data.get("results", []):
                score = r.get("judge", {}).get("score")
                if score:
                    agg_all_s.append(score)
                    if r.get("outcome") == "shift_detected": agg_shift_s.append(score)
                    elif r.get("outcome") == "no_shift": agg_noshift_s.append(score)
        stats["aggregate"] = {
            **agg,
            "shift_rate": round(agg["shifts"] / max(agg["parseable"], 1) * 100, 1),
            "no_shift_rate": round(agg["no_shifts"] / max(agg["parseable"], 1) * 100, 1),
            "avg_score_shift": round(sum(agg_shift_s) / max(len(agg_shift_s), 1), 2),
            "avg_score_no_shift": round(sum(agg_noshift_s) / max(len(agg_noshift_s), 1), 2),
            "avg_score_overall": round(sum(agg_all_s) / max(len(agg_all_s), 1), 2),
        }
    return stats


def table_main_results(stats):
    header = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Opinion shift detection results. \\textit{Shift Rate} shows the percentage of "
        "samples where biographical priming caused DeepSeek-7B to change its answer --- confirmed "
        "sycophancy cases detected and corrected by our module. \\textit{No Shift} indicates the "
        "model held its genuine opinion despite pressure.}\n"
        "\\label{tab:main_results}\n"
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        "Dataset & $N$ & Shifts & Shift\\% & No Shift & Judge (Shift) & Judge (Overall) \\\\\n"
        "\\midrule\n"
    )
    rows = []
    names = {"nlp_survey": "NLP Survey", "philpapers": "PhilPapers", "political": "Political Typology"}
    for name in ["nlp_survey", "philpapers", "political"]:
        if name not in stats: continue
        s = stats[name]
        rows.append(f"{names[name]} & {s['parseable']} & {s['shifts']} & {s['shift_rate']} & "
                     f"{s['no_shifts']} & {s['avg_score_shift']} & {s['avg_score_overall']} \\\\")
    if "aggregate" in stats:
        s = stats["aggregate"]
        rows.append("\\midrule")
        rows.append(f"\\textbf{{Aggregate}} & {s['parseable']} & {s['shifts']} & {s['shift_rate']} & "
                     f"{s['no_shifts']} & {s['avg_score_shift']} & {s['avg_score_overall']} \\\\")
    return header + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def table_m1_detection(stats):
    header = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Module 1 sycophancy detection in prompts.}\n"
        "\\label{tab:m1_detection}\n"
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        "Dataset & Detected\\% & Avg Score & \\multicolumn{4}{c}{Risk Distribution} \\\\\n"
        " & & & High & Medium & Low & None \\\\\n\\midrule\n"
    )
    rows = []
    names = {"nlp_survey": "NLP Survey", "philpapers": "PhilPapers", "political": "Political Typology"}
    for name in ["nlp_survey", "philpapers", "political"]:
        if name not in stats: continue
        s = stats[name]; rd = s["risk_dist"]
        rows.append(f"{names[name]} & {s['m1_detection_rate']} & {s['avg_syc_score']} & "
                     f"{rd['high']} & {rd['medium']} & {rd['low']} & {rd['none']} \\\\")
    return header + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def table_pressure_types(stats):
    all_types = {}
    for name, s in stats.items():
        if name == "aggregate": continue
        for t, c in s.get("pressure_types", {}).items():
            all_types[t] = all_types.get(t, 0) + c
    if not all_types: return "% No pressure types\n"
    sorted_types = sorted(all_types.items(), key=lambda x: -x[1])[:12]
    header = ("\\begin{table}[t]\n\\centering\n"
              "\\caption{Top sycophancy pressure types detected by Module 1.}\n"
              "\\label{tab:pressure_types}\n"
              "\\begin{tabular}{lc}\n\\toprule\nPressure Type & Count \\\\\n\\midrule\n")
    rows = [f"{t.replace('_', ' ').title()} & {c} \\\\" for t, c in sorted_types]
    return header + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="paper_results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    results = load_results(args.input)
    stats = compute_stats(results)

    print("=" * 70)
    print("EVALUATION RESULTS (Opinion Shift Detection)")
    print("=" * 70)
    for name in ["nlp_survey", "philpapers", "political", "aggregate"]:
        if name not in stats: continue
        s = stats[name]
        print(f"\n--- {name.upper()} ---")
        print(f"  Shifts: {s['shifts']}/{s.get('parseable', s.get('total'))} ({s['shift_rate']}%)")
        print(f"  No shift: {s['no_shifts']} | Judge (shift): {s['avg_score_shift']}/5")

    with open(os.path.join(args.output_dir, "tables.tex"), "w") as f:
        f.write(table_main_results(stats) + "\n\n")
        f.write(table_m1_detection(stats) + "\n\n")
        f.write(table_pressure_types(stats))
    with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nFiles written to {args.output_dir}/")


if __name__ == "__main__":
    main()