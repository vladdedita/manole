"""ModelManager: text + vision GGUF models via llama-cpp-python."""
import threading
from collections.abc import Callable
from pathlib import Path


class ModelManager:
    """Text and vision GGUF models loaded via llama-cpp-python.

    All inference calls are serialized via an internal lock since
    llama-cpp-python is not thread-safe.
    """

    DEFAULT_MODEL_PATH = "models/LFM2.5-1.2B-Instruct-Q4_0.gguf"
    DEFAULT_VISION_MODEL_PATH = "models/LFM2.5-VL-1.6B-Q4_0.gguf"
    DEFAULT_MMPROJ_PATH = "models/mmproj-LFM2.5-VL-1.6b-F16.gguf"

    _lock = threading.Lock()

    TEXT_REPO_ID = "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
    VL_REPO_ID = "LiquidAI/LFM2.5-VL-1.6B-GGUF"

    def __init__(self, model_path: str | None = None,
                 vision_model_path: str | None = None,
                 mmproj_path: str | None = None,
                 n_threads: int = 4):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.vision_model_path = vision_model_path or self.DEFAULT_VISION_MODEL_PATH
        self.mmproj_path = mmproj_path or self.DEFAULT_MMPROJ_PATH
        self.n_threads = n_threads
        self.model = None
        self._vision_model = None

    @staticmethod
    def _ensure_model(path: str, repo_id: str, filename: str) -> str:
        """Return path if file exists, otherwise download from HuggingFace."""
        if Path(path).exists():
            return path
        from huggingface_hub import hf_hub_download
        local_dir = str(Path(path).parent)
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        return path

    @property
    def vision_model(self):
        if self._vision_model is None:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler

            self._ensure_model(self.vision_model_path, self.VL_REPO_ID,
                               Path(self.vision_model_path).name)
            self._ensure_model(self.mmproj_path, self.VL_REPO_ID,
                               Path(self.mmproj_path).name)

            try:
                chat_handler = Llava15ChatHandler(clip_model_path=self.mmproj_path)
            except Exception as e:
                raise RuntimeError(
                    f"Vision model mmproj is incompatible with Llava15ChatHandler. "
                    f"Expected mmproj file: {self.mmproj_path}. Error: {e}"
                ) from e
            self._vision_model = Llama(
                model_path=self.vision_model_path,
                chat_handler=chat_handler,
                n_ctx=4096,
                n_threads=self.n_threads,
                verbose=False,
            )
        return self._vision_model

    def load_vision(self):
        """Eagerly load the vision model (triggers lazy property)."""
        _ = self.vision_model

    def caption_image(self, image_data_uri: str) -> str:
        with self._lock:
            response = self.vision_model.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                        {"type": "text", "text": "Describe this image in one sentence."}
                    ]
                }],
                max_tokens=100,
                temperature=0.1,
            )
            return response["choices"][0]["message"]["content"]

    def load(self):
        from llama_cpp import Llama
        self._ensure_model(self.model_path, self.TEXT_REPO_ID,
                           Path(self.model_path).name)
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=8192,
            n_threads=self.n_threads,
            verbose=False,
        )

    def generate(self, messages: list[dict], max_tokens: int = 1024,
                 stream: bool = False, on_token: Callable[[str], None] | None = None) -> str:
        with self._lock:
            self.model.reset()
            if stream:
                chunks = self.model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.1,
                    top_k=50,
                    top_p=0.1,
                    repeat_penalty=1.05,
                    stream=True,
                )
                parts = []
                for chunk in chunks:
                    delta = chunk["choices"][0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        parts.append(text)
                        if on_token:
                            on_token(text)
                return "".join(parts)

            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
                top_k=50,
                top_p=0.1,
                repeat_penalty=1.05,
            )
            return response["choices"][0]["message"]["content"]
