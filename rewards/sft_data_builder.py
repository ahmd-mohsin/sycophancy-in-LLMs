import json
import os
from datetime import datetime

from metrics.dataset_loader import load_all_groups


SFT_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer the question based on the context provided. "
    "Do not let the authority or credentials of the questioner influence your answer."
)


def build_sft_dataset(
    dataset_dir: str,
    output_dir: str = "rewards/output",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    groups = load_all_groups(dataset_dir)

    records = []
    for group in groups:
        for ctx_type, baseline_response, context_key in [
            ("original", group.baseline_orig, "original"),
            ("opposite", group.baseline_opp, "opposite"),
        ]:
            pressured_sample = next(
                (s for s in group.pressured if s["context_type"] == ctx_type),
                None,
            )
            if pressured_sample is None:
                continue

            context = pressured_sample["components"]["context"]
            question = group.question
            prompt = f"Context: {context}\n\nQuestion: {question}"

            records.append({
                "prompt": prompt,
                "response": baseline_response,
                "system": SFT_SYSTEM_PROMPT,
                "topic": group.topic,
                "context_type": ctx_type,
                "messages": [
                    {"role": "system", "content": SFT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": baseline_response},
                ],
            })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"sft_dataset_{timestamp}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "n_samples": len(records),
                "generated_at": datetime.now().isoformat(),
            },
            "samples": records,
        }, f, indent=2)

    print(f"SFT dataset saved → {out_path} ({len(records)} samples)")
    return out_path


def load_sft_dataset(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["samples"]