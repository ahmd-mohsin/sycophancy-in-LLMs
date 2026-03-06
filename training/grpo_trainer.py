import json
import os
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported

from rewards.grpo_data_builder import load_grpo_training_data, build_grpo_prompt
from rewards.reward_aggregator import RewardWeights
from training.grpo_reward_fn import make_reward_fn

PRESSURE_MAP = {
    "low":    "I think {opinion}.",
    "medium": "As an expert, I believe {opinion}.",
    "high":   "As a world authority, I am absolutely certain that {opinion}.",
}


def build_grpo_hf_dataset(grpo_path: str, tokenizer) -> tuple[Dataset, list[dict]]:
    groups = load_grpo_training_data(grpo_path)

    prompts = []
    for group in groups:
        question = group["question"]
        opinion  = group["opinion"]

        for completion_meta in group["completions"]:
            pressure_level = completion_meta["pressure_level"]
            context_type   = completion_meta["context_type"]

            context = completion_meta.get("context", "")
            if not context:
                fallback = (
                    group["baseline_orig"] if context_type == "original"
                    else group.get("baseline_opp", group["baseline_orig"])
                )
                context = fallback[:600]

            pressure_prompt = PRESSURE_MAP.get(pressure_level, "").format(opinion=opinion)
            messages = build_grpo_prompt(pressure_prompt, context, question)
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            idx = len(prompts)
            prompt_text_tagged = f"<!-- __GRPO_IDX_{idx}__ -->\n" + prompt_text
            prompts.append({"prompt": prompt_text_tagged})

    return Dataset.from_list(prompts), groups


def run_grpo(
    model,
    tokenizer,
    grpo_dataset_path: str,
    output_dir: str = "training/checkpoints/grpo",
    weights: RewardWeights = None,
    category: str = "opinion",
    judge_model: str = None,
    max_seq_length: int = 4096,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-6,
    num_generations: int = 4,
    seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)

    if weights is None:
        weights = RewardWeights()

    if weights.delta < 0.5:
        raise ValueError(
            f"weights.delta={weights.delta} is too low. "
            f"GAS penalty should be >= 0.5 to prevent hedging. "
            f"Pass --delta 0.6 (or set it explicitly)."
        )
    if weights.epsilon <= 0:
        raise ValueError(f"weights.epsilon={weights.epsilon} must be > 0 for Rpos to have effect.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset, groups = build_grpo_hf_dataset(grpo_dataset_path, tokenizer)
    reward_fn = make_reward_fn(groups, weights=weights, category=category, judge_model=judge_model)

    print(f"\nPhase 2 — GRPO Training")
    print(f"  Groups          : {len(groups)}")
    print(f"  Prompts         : {len(dataset)}")
    print(f"  num_generations : {num_generations}  (completions per prompt)")
    print(f"  Effective batch : {per_device_batch_size * gradient_accumulation_steps}")
    print(f"  Output          : {output_dir}")
    print(f"  Weights         : α={weights.alpha} β={weights.beta} γ={weights.gamma} "
          f"ε={weights.epsilon} δ={weights.delta}")

    max_completion_length = 1024
    max_prompt_length = max_seq_length - max_completion_length

    # Probe which GRPOConfig params this TRL version accepts
    import inspect
    grpo_src = inspect.getsource(GRPOConfig)
    use_generation_batch_size = "generation_batch_size" in grpo_src
    use_steps_per_generation  = "steps_per_generation"  in grpo_src
    both_present = use_generation_batch_size and use_steps_per_generation

    grpo_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        temperature=0.9,
        beta=0.2,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        report_to="none",
        optim="adamw_8bit",
        use_vllm=False,
        remove_unused_columns=False,
        dataloader_drop_last=True,
    )

    # TRL versions differ on how to set generation batch size:
    # - Older: steps_per_generation only
    # - Newer: generation_batch_size only (mutually exclusive with steps_per_generation)
    # - Some: neither param exists, num_generations handles it
    if both_present:
        # Mutually exclusive — use generation_batch_size only
        grpo_kwargs["generation_batch_size"] = num_generations
        print(f"  TRL config      : generation_batch_size={num_generations}")
    elif use_steps_per_generation:
        grpo_kwargs["steps_per_generation"] = 1
        print(f"  TRL config      : steps_per_generation=1")
    elif use_generation_batch_size:
        grpo_kwargs["generation_batch_size"] = num_generations
        print(f"  TRL config      : generation_batch_size={num_generations}")
    else:
        print(f"  TRL config      : no explicit generation batch param (using defaults)")

    grpo_config = GRPOConfig(**grpo_kwargs)

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
    )

    trainer.model.generation_config.max_new_tokens  = max_completion_length
    trainer.model.generation_config.min_new_tokens  = 100
    trainer.model.generation_config.do_sample       = True
    trainer.model.generation_config.temperature     = 0.9
    trainer.model.generation_config.top_p           = 0.9

    result = trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    stats = {
        "train_loss":    result.training_loss,
        "train_runtime": result.metrics.get("train_runtime"),
        "n_groups":      len(groups),
        "weights":       weights.__dict__,
    }

    with open(os.path.join(output_dir, "grpo_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nGRPO complete — loss: {result.training_loss:.4f}")
    return model, tokenizer, stats