"""ModelManager: loads LFM2-350M-Extract and LFM2-1.2B-RAG concurrently."""
from pathlib import Path


class ModelManager:
    """Manages two GGUF models loaded via llama-cpp-python."""

    DEFAULT_PLANNER_PATH = "models/LFM2-350M-Extract-Q4_0.gguf"
    DEFAULT_RAG_PATH = "models/LFM2-1.2B-RAG-Q4_0.gguf"

    def __init__(
        self,
        planner_path: str | None = None,
        rag_path: str | None = None,
        n_threads: int = 4,
    ):
        self.planner_path = planner_path or self.DEFAULT_PLANNER_PATH
        self.rag_path = rag_path or self.DEFAULT_RAG_PATH
        self.n_threads = n_threads
        self.planner_model = None
        self.rag_model = None

    def load(self):
        """Load both models. Call once at startup."""
        from llama_cpp import Llama

        self.planner_model = Llama(
            model_path=self.planner_path,
            n_ctx=2048,
            n_threads=self.n_threads,
            verbose=False,
        )
        self.rag_model = Llama(
            model_path=self.rag_path,
            n_ctx=4096,
            n_threads=self.n_threads,
            verbose=False,
        )

    def plan(self, prompt: str) -> str:
        """Run prompt through 350M-Extract model. For structured JSON extraction."""
        response = self.planner_model(prompt, max_tokens=256, temperature=0.0)
        return response["choices"][0]["text"]

    def extract(self, prompt: str) -> str:
        """Run prompt through 1.2B-RAG model. For per-chunk fact extraction."""
        response = self.rag_model(prompt, max_tokens=512, temperature=0.0)
        return response["choices"][0]["text"]

    def synthesize(self, prompt: str) -> str:
        """Run prompt through 1.2B-RAG model. For answer synthesis."""
        response = self.rag_model(prompt, max_tokens=1024, temperature=0.1)
        return response["choices"][0]["text"]
