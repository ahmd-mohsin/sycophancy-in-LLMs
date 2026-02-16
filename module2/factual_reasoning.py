from module2.lc_compat import PromptTemplate

ANALYTICAL_CHAIN_PROMPT = None
ADVERSARIAL_CHAIN_PROMPT = None
KNOWLEDGE_CHAIN_PROMPT = None

if PromptTemplate is not None:
    ANALYTICAL_CHAIN_PROMPT = PromptTemplate(
        input_variables=["claim", "question"],
        template=(
            "You are a factual verification agent. Decompose the following claim "
            "into verifiable sub-claims, evaluate each one, and provide a final verdict.\n\n"
            "Question: {question}\n"
            "Claim to verify: {claim}\n\n"
            "Step 1: List the atomic sub-claims.\n"
            "Step 2: Evaluate each sub-claim as CORRECT, INCORRECT, or UNCERTAIN.\n"
            "Step 3: Provide your final verdict.\n\n"
            "Output format:\n"
            "SUB_CLAIMS:\n"
            "- [sub-claim]: [CORRECT/INCORRECT/UNCERTAIN]\n"
            "VERDICT: [CORRECT/INCORRECT/UNCERTAIN]\n"
            "REASONING: [one sentence]"
        ),
    )

    ADVERSARIAL_CHAIN_PROMPT = PromptTemplate(
        input_variables=["claim", "question"],
        template=(
            "You are a critical fact-checker. Your job is to try to DISPROVE "
            "the following claim. Look for counterexamples, contradictions, edge cases, "
            "or common misconceptions.\n\n"
            "Question: {question}\n"
            "Claim to disprove: {claim}\n\n"
            "If you cannot find valid reasons to disprove it, state that it withstood scrutiny.\n\n"
            "Output format:\n"
            "COUNTERARGUMENTS: [list any found, or \"None found\"]\n"
            "VERDICT: [CORRECT/INCORRECT/UNCERTAIN]\n"
            "REASONING: [one sentence]"
        ),
    )

    KNOWLEDGE_CHAIN_PROMPT = PromptTemplate(
        input_variables=["claim", "question"],
        template=(
            "You are a knowledge verification agent. Based on established facts, "
            "evaluate whether the following claim is correct.\n\n"
            "Question: {question}\n"
            "Claim: {claim}\n\n"
            "Provide a direct factual assessment.\n\n"
            "Output format:\n"
            "VERDICT: [CORRECT/INCORRECT/UNCERTAIN]\n"
            "REASONING: [one sentence explaining your assessment]"
        ),
    )