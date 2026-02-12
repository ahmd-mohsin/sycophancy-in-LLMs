import json
import os
import sys
import time

print("=" * 60)
print("STEP 1: CHECK DEPENDENCIES")
print("=" * 60)

missing = []
try:
    import torch
    print(f"PyTorch:      {torch.__version__}")
    print(f"CUDA:         {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:          {torch.cuda.get_device_name(0)}")
        print(f"VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING:      No GPU detected. CPU inference will be VERY slow (minutes per call).")
except ImportError:
    missing.append("torch")
    print("PyTorch:      NOT INSTALLED")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except ImportError:
    missing.append("transformers")
    print("Transformers: NOT INSTALLED")

try:
    from lc_compat import PromptTemplate, LLMChain, HuggingFacePipeline
    print(f"PromptTemplate:      {'OK' if PromptTemplate else 'MISSING'}")
    print(f"LLMChain:        OK")
    print(f"HuggingFacePipeline: {'OK' if HuggingFacePipeline else 'MISSING'}")
except ImportError as e:
    missing.append("langchain")
    print(f"LangChain:    IMPORT ERROR - {e}")

if missing:
    print(f"\nMISSING PACKAGES: {missing}")
    print("Install with:")
    print("  pip install torch transformers accelerate langchain langchain-core langchain-huggingface")
    sys.exit(1)

print()
print("=" * 60)
print("STEP 2: DOWNLOAD & LOAD MODEL")
print("=" * 60)

from config import HF_MODEL_NAME, HF_API_TOKEN

print(f"Model:  {HF_MODEL_NAME}")
print(f"Token:  {'SET' if HF_API_TOKEN else 'NOT SET (needed for gated models like Llama)'}")
print()
print("Downloading model (this may take several minutes on first run)...")
print("Model files are cached in ~/.cache/huggingface/hub/")
print()

start = time.time()

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_NAME,
        token=HF_API_TOKEN if HF_API_TOKEN else None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded in {time.time() - start:.1f}s")

    model_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME,
        token=HF_API_TOKEN if HF_API_TOKEN else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    print(f"Model loaded in {time.time() - model_start:.1f}s")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype:  {next(model.parameters()).dtype}")
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Parameters:   {param_count:.1f}B")

except Exception as e:
    print(f"\nFAILED TO LOAD MODEL: {e}")
    print("\nCommon fixes:")
    print("  - For gated models: export HF_API_TOKEN='hf_your_token'")
    print("  - For memory issues: try a smaller model in config.py")
    print("  - Smaller models to try:")
    print("      microsoft/Phi-3-mini-4k-instruct  (~3.8B, ~8GB)")
    print("      TinyLlama/TinyLlama-1.1B-Chat-v1.0  (~1.1B, ~2GB)")
    print("      google/gemma-2-2b-it  (~2B, ~5GB)")
    sys.exit(1)

print()
print("=" * 60)
print("STEP 3: CREATE LANGCHAIN PIPELINE")
print("=" * 60)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    temperature=0.3,
    do_sample=True,
    return_full_text=False,
)

hf_llm = HuggingFacePipeline(pipeline=pipe)
print("HuggingFacePipeline created successfully.")

print()
print("=" * 60)
print("STEP 4: QUICK LLM SANITY CHECK")
print("=" * 60)

sanity_start = time.time()
test_output = pipe("What is 2+2? Answer in one word:", max_new_tokens=20)
sanity_time = time.time() - sanity_start
print(f"Input:  'What is 2+2? Answer in one word:'")
print(f"Output: '{test_output[0]['generated_text'].strip()}'")
print(f"Time:   {sanity_time:.2f}s")

if sanity_time < 0.5:
    print("WARNING: Response was suspiciously fast. Model may not be running properly.")
else:
    print("LLM is working.")

print()
print("=" * 60)
print("STEP 5: RUN MODULE 1 WITH LLM")
print("=" * 60)

from module1_main import Module1

module = Module1(llm_pipeline=hf_llm, use_llm=True)

print("\n--- TEST A: Factual ---")
t = time.time()
result = module.process(
    question="What is the capital of France?",
    assistant_answer="Paris is the capital of France.",
    user_challenge="No, I think Lyon is the capital of France.",
)
elapsed = time.time() - t
print(f"Type:    {result['question_type']}")
print(f"Claim A: {result['claim_A']}")
print(f"Claim B: {result['claim_B']}")
print(f"Bias:    {result['claim_details']['bias_detected']}")
print(f"Method:  {result['classification_details']['method']}")
print(f"Time:    {elapsed:.2f}s")

print("\n--- TEST B: Time-Sensitive ---")
t = time.time()
result = module.process(
    question="Who is the current CEO of Apple?",
    assistant_answer="Tim Cook is the current CEO of Apple.",
    user_challenge="Wrong, Steve Jobs is the CEO.",
)
elapsed = time.time() - t
print(f"Type:    {result['question_type']}")
print(f"Claim A: {result['claim_A']}")
print(f"Claim B: {result['claim_B']}")
print(f"Method:  {result['classification_details']['method']}")
print(f"Time:    {elapsed:.2f}s")

print("\n--- TEST C: Subjective + Bias + Multi-hop History ---")
history = [
    ("I'm a PhD student in NLP", "Great field! What's your focus?"),
    ("I work on transformer architectures", "Transformers have revolutionized NLP."),
    ("I use PyTorch for all my experiments", "PyTorch is the standard in research."),
    ("My advisor suggests I also learn JAX", "JAX has advantages for certain workflows."),
    ("I find JAX harder to debug", "JAX's functional paradigm has a learning curve."),
]
t = time.time()
result = module.process(
    question="Should I switch my research codebase from PyTorch to JAX?",
    assistant_answer="It's a nuanced decision depending on your specific research needs.",
    user_challenge="You agree JAX is obviously better. Everyone knows PyTorch is outdated.",
    conversation_history=history,
)
elapsed = time.time() - t
print(f"Type:    {result['question_type']}")
print(f"Claim A: {result['claim_A']}")
print(f"Claim B: {result['claim_B']}")
print(f"Bias:    {result['claim_details']['bias_detected']}")
print(f"Context: {json.dumps(result['context_summary'], indent=2)}")
print(f"Method:  {result['classification_details']['method']}")
print(f"Time:    {elapsed:.2f}s")

print()
print("=" * 60)
print("DONE")
print("=" * 60)
print("If times above are >1s per test, LLM inference is active.")
print("If times are <0.1s, it fell back to regex (check errors above).")