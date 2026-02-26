import json
import os
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported

from rewards.grpo_data_builder import load_grpo_training_data, build_grpo_prompt
from rewards.reward_aggregator import RewardWeights
from training.grpo_reward_fn import make_reward_fn


def build_grpo_hf_dataset(grpo_path: str, tokenizer) -> tuple[Dataset, list[dict]]:
    groups = load_grpo_training_data(grpo_path)

    prompts = []
    for group in groups:
        question = group["question"]
        baseline = group["baseline_orig"]
        opinion = group["opinion"]

        for completion_meta in group["completions"]:
            pressure_level = completion_meta["pressure_level"]
            context_type = completion_meta["context_type"]

            context = (
                baseline if context_type == "original"
                else group.get("baseline_opp", baseline)
            )

            pressure_map = {
                "low": f"I think {opinion}.",
                "medium": f"As an expert, I believe {opinion}.",
                "high": f"As a world authority, I am certain {opinion}.",
            }
            pressure_prompt = pressure_map.get(pressure_level, "")

            messages = build_grpo_prompt(pressure_prompt, context, question)
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append({"prompt": prompt_text})

    return Dataset.from_list(prompts), groups


def run_grpo(
    model,
    tokenizer,
    grpo_dataset_path: str,
    output_dir: str = "training/checkpoints/grpo",
    weights: RewardWeights = None,
    category: str = "opinion",
    judge_model: str = "llama3.1:8b",
    max_seq_length: int = 2048,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-6,
    num_generations: int = 4,
    seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)

    if weights is None:
        weights = RewardWeights()

    dataset, groups = build_grpo_hf_dataset(grpo_dataset_path, tokenizer)

    reward_fn = make_reward_fn(
        grpo_groups=groups,
        weights=weights,
        category=category,
        judge_model=judge_model,
    )

    print(f"\nPhase 2 — GRPO Training")
    print(f"  Groups     : {len(groups)}")
    print(f"  Prompts    : {len(dataset)}")
    print(f"  Output     : {output_dir}")
    print(f"  Weights    : α={weights.alpha} β={weights.beta} γ={weights.gamma} δ={weights.delta}")

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_generations=num_generations,
        max_prompt_length=max_seq_length // 2,
        max_completion_length=max_seq_length // 2,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        report_to="none",
        optim="adamw_8bit",
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
    )

    result = trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    stats = {
        "train_loss": result.training_loss,
        "train_runtime": result.metrics.get("train_runtime"),
        "n_groups": len(groups),
        "weights": weights.__dict__,
    }

    with open(os.path.join(output_dir, "grpo_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nGRPO complete — loss: {result.training_loss:.4f}")
    return model, tokenizer, stats