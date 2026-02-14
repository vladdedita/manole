"""ModelManager: loads LFM2.5-1.2B-Instruct and LFM2-350M-Extract."""
from pathlib import Path

# LFM2.5 recommended generation params
_INSTRUCT_PARAMS = dict(temperature=0.1, top_k=50, top_p=0.1, repeat_penalty=1.05)
_EXTRACT_PARAMS = dict(temperature=0.0)


def _messages(system: str, user: str) -> list[dict]:
    """Build chat messages list for create_chat_completion."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class ModelManager:
    """Manages two GGUF models loaded via llama-cpp-python."""

    DEFAULT_EXTRACT_PATH = "models/LFM2-350M-Extract-Q4_0.gguf"
    DEFAULT_INSTRUCT_PATH = "models/LFM2.5-1.2B-Instruct-Q4_0.gguf"

    def __init__(
        self,
        extract_path: str | None = None,
        instruct_path: str | None = None,
        n_threads: int = 4,
    ):
        self.extract_path = extract_path or self.DEFAULT_EXTRACT_PATH
        self.instruct_path = instruct_path or self.DEFAULT_INSTRUCT_PATH
        self.n_threads = n_threads
        self.extract_model = None
        self.instruct_model = None

    def load(self):
        """Load both models. Call once at startup."""
        from llama_cpp import Llama

        self.extract_model = Llama(
            model_path=self.extract_path,
            n_ctx=2048,
            n_threads=self.n_threads,
            verbose=False,
        )
        self.instruct_model = Llama(
            model_path=self.instruct_path,
            n_ctx=4096,
            n_threads=self.n_threads,
            verbose=False,
        )

    def _chat(self, model, system: str, user: str, max_tokens: int, **params) -> str:
        """Run chat completion on a model."""
        model.reset()
        response = model.create_chat_completion(
            messages=_messages(system, user),
            max_tokens=max_tokens,
            **params,
        )
        return response["choices"][0]["message"]["content"]

    def plan(self, system: str, user: str) -> str:
        """LFM2.5-1.2B: query planning."""
        return self._chat(self.instruct_model, system, user, max_tokens=256, **_INSTRUCT_PARAMS)

    def rewrite(self, system: str, user: str) -> str:
        """LFM2.5-1.2B: query rewriting."""
        return self._chat(self.instruct_model, system, user, max_tokens=256, **_INSTRUCT_PARAMS)

    def map_chunk(self, system: str, user: str) -> str:
        """LFM2.5-1.2B: per-chunk relevance + fact extraction."""
        return self._chat(self.instruct_model, system, user, max_tokens=256, **_INSTRUCT_PARAMS)

    def extract(self, system: str, user: str) -> str:
        """350M-Extract: structured data extraction fallback."""
        return self._chat(self.extract_model, system, user, max_tokens=512, **_EXTRACT_PARAMS)

    def synthesize(self, system: str, user: str) -> str:
        """LFM2.5-1.2B: answer synthesis."""
        return self._chat(self.instruct_model, system, user, max_tokens=1024, **_INSTRUCT_PARAMS)
