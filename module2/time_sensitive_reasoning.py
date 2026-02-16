from module2.lc_compat import PromptTemplate

DIRECT_KNOWLEDGE_PROMPT = None
REVERSE_VERIFICATION_PROMPT = None
CROSS_REFERENCE_PROMPT = None

if PromptTemplate is not None:
    DIRECT_KNOWLEDGE_PROMPT = PromptTemplate(
        input_variables=["claim", "question"],
        template=(
            "You are a knowledge recall agent. Based on your training data, provide "
            "the most recent information you have about the following topic.\n\n"
            "Question: {question}\n"
            "Claim to evaluate: {claim}\n\n"
            "What do you know about this topic? Is the claim consistent with your "
            "most recent information?\n\n"
            "Output format:\n"
            "KNOWN_INFO: [what you know about this topic]\n"
            "SUPPORTS_CLAIM: [YES/NO/UNCERTAIN]\n"
            "REASONING: [one sentence]"
        ),
    )

    REVERSE_VERIFICATION_PROMPT = PromptTemplate(
        input_variables=["claim", "question"],
        template=(
            "You are a consistency checker. Evaluate whether the following claim "
            "is internally consistent with known facts. Identify any contradictions "
            "or inconsistencies.\n\n"
            "Question: {question}\n"
            "Claim: {claim}\n\n"
            "What would need to be true for this claim to be correct? Is anything "
            "contradictory or inconsistent?\n\n"
            "Output format:\n"
            "CONTRADICTIONS: [list any found, or \"None found\"]\n"
            "SUPPORTS_CLAIM: [YES/NO/UNCERTAIN]\n"
            "REASONING: [one sentence]"
        ),
    )

    CROSS_REFERENCE_PROMPT = PromptTemplate(
        input_variables=["claim", "question", "entity"],
        template=(
            "You are a cross-reference agent. Verify the following claim by checking "
            "it against related known facts about the entities involved.\n\n"
            "Question: {question}\n"
            "Claim: {claim}\n"
            "Primary entity: {entity}\n\n"
            "What is this entity primarily known for or associated with? Does the "
            "claim match what is known?\n\n"
            "Output format:\n"
            "ENTITY_KNOWN_FOR: [primary associations]\n"
            "SUPPORTS_CLAIM: [YES/NO/UNCERTAIN]\n"
            "REASONING: [one sentence]"
        ),
    )