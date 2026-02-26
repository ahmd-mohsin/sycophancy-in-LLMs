from unsloth import FastLanguageModel
import torch

SUPPORTED_MODELS = [
    "unsloth/llama-3.1-8b-instruct",
    "unsloth/mistral-nemo-instruct-2407",
    "unsloth/gemma-2-9b-it",
    "unsloth/Qwen2.5-14B-Instruct",
]

_loaded_model = None
_loaded_tokenizer = None
_loaded_model_name = None


def load_model(
    model_name: str = "unsloth/llama-3.1-8b-instruct",
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
    dtype: torch.dtype = None,
) -> tuple:
    global _loaded_model, _loaded_tokenizer, _loaded_model_name

    if _loaded_model is not None and _loaded_model_name == model_name:
        return _loaded_model, _loaded_tokenizer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
    )

    _loaded_model = model
    _loaded_tokenizer = tokenizer
    _loaded_model_name = model_name
    return model, tokenizer


def get_peft_model(
    model,
    r: int = 16,
    target_modules: list[str] = None,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    bias: str = "none",
):
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    return FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )


def save_model(model, tokenizer, output_dir: str):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved → {output_dir}")


def save_model_gguf(model, tokenizer, output_dir: str, quantization: str = "q4_k_m"):
    model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=quantization)
    print(f"GGUF model saved → {output_dir}")