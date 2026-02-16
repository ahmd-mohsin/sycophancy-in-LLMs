try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    try:
        from langchain.prompts import PromptTemplate
    except ImportError:
        PromptTemplate = None

try:
    from langchain_huggingface import HuggingFacePipeline
except ImportError:
    try:
        from langchain_community.llms import HuggingFacePipeline
    except ImportError:
        HuggingFacePipeline = None


class LLMChain:

    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def invoke(self, inputs: dict) -> dict:
        formatted = self.prompt.format(**inputs)
        if hasattr(self.llm, "invoke"):
            result = self.llm.invoke(formatted)
        elif hasattr(self.llm, "predict"):
            result = self.llm.predict(formatted)
        elif callable(self.llm):
            result = self.llm(formatted)
        else:
            result = str(self.llm.generate([formatted]))

        if isinstance(result, str):
            return {"text": result}
        if hasattr(result, "content"):
            return {"text": result.content}
        return {"text": str(result)}