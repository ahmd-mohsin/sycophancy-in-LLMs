try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    try:
        from langchain.prompts import PromptTemplate
    except ImportError:
        PromptTemplate = None

try:
    from langchain.chains import LLMChain
except ImportError:
    try:
        from langchain_classic.chains import LLMChain
    except ImportError:
        LLMChain = None

try:
    from langchain_huggingface import HuggingFacePipeline
except ImportError:
    try:
        from langchain_community.llms import HuggingFacePipeline
    except ImportError:
        HuggingFacePipeline = None