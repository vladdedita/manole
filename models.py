"""ModelManager: single LFM2.5-1.2B-Instruct model with generate()."""
from pathlib import Path


class ModelManager:
    """Single GGUF model loaded via llama-cpp-python."""

    DEFAULT_MODEL_PATH = "models/LFM2.5-1.2B-Instruct-Q4_0.gguf"

    def __init__(self, model_path: str | None = None, n_threads: int = 4):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.n_threads = n_threads
        self.model = None

    def load(self):
        from llama_cpp import Llama
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=8192,
            n_threads=self.n_threads,
            verbose=False,
        )

    def generate(self, messages: list[dict], max_tokens: int = 1024) -> str:
        self.model.reset()
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
            top_k=50,
            top_p=0.1,
            repeat_penalty=1.05,
        )
        return response["choices"][0]["message"]["content"]
