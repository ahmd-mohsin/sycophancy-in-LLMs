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

        # Extract text from various return formats
        if isinstance(result, str):
            text = result
        elif isinstance(result, list) and len(result) > 0:
            item = result[0]
            if isinstance(item, dict) and "generated_text" in item:
                text = item["generated_text"]
            else:
                text = str(item)
        else:
            text = str(result)

        # CRITICAL FIX: Strip the prompt prefix from the output.
        # HuggingFace text-generation pipelines often return the full
        # prompt + completion.  The HuggingFacePipeline LangChain wrapper
        # is supposed to strip it, but doesn't always.
        # If the output starts with the prompt, remove it.
        if text.startswith(formatted):
            text = text[len(formatted):]
        # Also try partial match — sometimes whitespace differs
        elif formatted.rstrip() and text.startswith(formatted.rstrip()):
            text = text[len(formatted.rstrip()):]

        return {"text": text.strip()}