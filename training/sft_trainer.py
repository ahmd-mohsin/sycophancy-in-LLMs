import json
import os
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

from rewards.sft_data_builder import load_sft_dataset


def _format_chat(record: dict, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        record["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def build_sft_hf_dataset(sft_path: str, tokenizer) -> Dataset:
    records = load_sft_dataset(sft_path)
    texts = [_format_chat(r, tokenizer) for r in records]
    return Dataset.from_dict({"text": texts})


def run_sft(
    model,
    tokenizer,
    sft_dataset_path: str,
    output_dir: str = "training/checkpoints/sft",
    max_seq_length: int = 2048,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)

    dataset = build_sft_hf_dataset(sft_dataset_path, tokenizer)

    print(f"\nPhase 1 — SFT Warmup")
    print(f"  Samples : {len(dataset)}")
    print(f"  Output  : {output_dir}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        report_to="none",
        optim="adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=training_args,
    )

    result = trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    stats = {
        "train_loss": result.training_loss,
        "train_runtime": result.metrics.get("train_runtime"),
        "train_samples_per_second": result.metrics.get("train_samples_per_second"),
        "n_samples": len(dataset),
    }

    with open(os.path.join(output_dir, "sft_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSFT complete — loss: {result.training_loss:.4f}")
    return model, tokenizer, stats