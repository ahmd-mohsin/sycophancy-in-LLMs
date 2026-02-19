class HFModelWrapper:
    """Thin wrapper for HuggingFace model loading."""

    def __init__(self, model_name: str = None, pipeline=None):
        self.model_name = model_name
        self.pipeline = pipeline