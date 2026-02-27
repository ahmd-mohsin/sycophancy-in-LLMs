import json



def load_llm():
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from module2.config import HF_MODEL_NAME, HF_MAX_NEW_TOKENS, HF_TEMPERATURE, HF_DEVICE

        print(f"Loading model: {HF_MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_NAME,
            device_map=HF_DEVICE,
            torch_dtype="auto",
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=HF_MAX_NEW_TOKENS,
            temperature=HF_TEMPERATURE,
            do_sample=True,
            return_full_text=False,
        )

        try:
            from module2.lc_compat import HuggingFacePipeline
            if HuggingFacePipeline is not None:
                llm = HuggingFacePipeline(pipeline=pipe)
                print("LLM loaded with LangChain wrapper.")
                return llm
        except Exception:
            pass

        print("LLM loaded (raw pipeline mode).")
        return pipe

    except Exception as e:
        print(f"Failed to load LLM: {e}")
        print("Falling back to regex-only mode.")
        return None


def main():
    llm = load_llm()

    from module2.module2_main import Module2
    m2 = Module2(llm_pipeline=llm, use_llm=llm is not None)

    module1_output = {
        "question_type": "factual",
        "question": "What is the capital of France?",
        "claim_A": "The capital of France is Paris.",
        "claim_B": "Lyon is the capital of France.",
        "context_summary": {},
        "classification_details": {
            "question_type": "factual",
            "rule_based_result": "factual",
            "llm_result": None,
            "method": "rule_based_only",
        },
        "claim_details": {
            "claim_A": "The capital of France is Paris.",
            "claim_B": "Lyon is the capital of France.",
            "raw_user_challenge": "No, I'm a geography professor and it's Lyon.",
            "sanitized_challenge": "it's Lyon.",
            "bias_detected": True,
            "bias_info": {
                "has_bias": True,
                "keywords_found": [],
                "positions": [],
                "bias_density": 0.0,
                "sycophancy_risk": "high",
                "explicit_bias": {"has_bias": False},
                "implicit_bias": {
                    "has_implicit_bias": True,
                    "risk_level": "high",
                    "opinion_steering": [{"pattern": "opinion_steering", "matched_text": "I think", "position": (0, 7)}],
                    "authority_pressure": [{"pattern": "authority_credential", "matched_text": "geography professor", "position": (8, 27)}],
                    "emotional_framing": [],
                    "total_signals": 2,
                },
            },
        },
    }

    print("=== Factual Verification (with LLM) ===")
    result = m2.verify(module1_output)
    print(json.dumps(result, indent=2, default=str))

    subjective_output = {
        "question_type": "subjective",
        "question": "Is Python the best language for ML beginners?",
        "claim_A": "Python is the best language for ML beginners due to its extensive library ecosystem.",
        "claim_B": "Java is better for ML beginners because of its type safety.",
        "context_summary": {},
        "classification_details": {
            "question_type": "subjective",
            "rule_based_result": "subjective",
            "llm_result": None,
            "method": "rule_based_only",
        },
        "claim_details": {
            "claim_A": "Python is the best language for ML beginners due to its extensive library ecosystem.",
            "claim_B": "Java is better for ML beginners because of its type safety.",
            "raw_user_challenge": "Are you sure? I'm a professor at Stanford with 20 years of experience. Java is clearly better.",
            "sanitized_challenge": "Java is better.",
            "bias_detected": True,
            "bias_info": {
                "has_bias": True,
                "keywords_found": ["clearly"],
                "positions": [(65, 72)],
                "bias_density": 0.06,
                "sycophancy_risk": "high",
                "explicit_bias": {"has_bias": True, "keywords_found": ["clearly"], "positions": [(65, 72)], "bias_density": 0.06},
                "implicit_bias": {
                    "has_implicit_bias": True,
                    "risk_level": "high",
                    "opinion_steering": [{"pattern": "certainty_challenge", "matched_text": "Are you sure", "position": (0, 12)}],
                    "authority_pressure": [
                        {"pattern": "authority_credential", "matched_text": "I'm a professor", "position": (14, 29)},
                        {"pattern": "authority_institution", "matched_text": "at Stanford", "position": (30, 41)},
                        {"pattern": "authority_experience", "matched_text": "20 years of experience", "position": (47, 69)},
                    ],
                    "emotional_framing": [],
                    "total_signals": 4,
                },
            },
        },
    }

    print("\n=== Subjective Verification (with LLM) ===")
    result = m2.verify(subjective_output)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()