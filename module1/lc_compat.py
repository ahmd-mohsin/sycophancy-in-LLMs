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

    def __init__(self, llm=None, prompt=None, **kwargs):
        self.llm = llm
        self.prompt = prompt

    def invoke(self, inputs: dict) -> dict:
        if self.llm is None or self.prompt is None:
            return {"text": ""}

        formatted = self.prompt.format(**inputs)

        try:
            result = self.llm.invoke(formatted)
        except Exception:
            try:
                result = self.llm(formatted)
            except Exception:
                return {"text": ""}

        if isinstance(result, str):
            return {"text": result}
        if isinstance(result, list) and len(result) > 0:
            item = result[0]
            if isinstance(item, dict) and "generated_text" in item:
                return {"text": item["generated_text"]}
            return {"text": str(item)}
        return {"text": str(result)}