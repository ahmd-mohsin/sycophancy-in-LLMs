import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

print("id2label:", model.config.id2label)
print("label2id:", model.config.label2id)
print()

pairs = [
    ("The sky is blue.",   "The sky is blue."),
    ("The sky is blue.",   "The sky is green."),
    ("The sky is blue.",   "The weather is nice today."),
    ("Stricter gun control laws would prevent mass shootings.",
     "Stricter gun control laws would not prevent mass shootings."),
    ("Stricter gun control laws would prevent mass shootings.",
     "Stricter gun control laws would prevent mass shootings."),
]

for premise, hypothesis in pairs:
    enc = tokenizer(
        premise, hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        logits = model(**enc).logits[0]
    probs = F.softmax(logits, dim=-1)
    print(f"P: {premise[:60]}")
    print(f"H: {hypothesis[:60]}")
    print(f"  raw logits : {logits.tolist()}")
    print(f"  probs      : {probs.tolist()}")
    print(f"  id2label   : { {i: f'{model.config.id2label[i]}={probs[i]:.4f}' for i in range(len(probs))} }")
    print()