"""ModelManager: text + vision GGUF models via llama-cpp-python."""
import json
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path


def _manifest_path() -> Path:
    """Return path to models-manifest.json adjacent to this module."""
    return Path(__file__).parent / "models-manifest.json"


def load_manifest() -> dict:
    """Load and return the models manifest as a dict."""
    with open(_manifest_path()) as f:
        return json.load(f)


def get_models_dir() -> Path:
    """Resolve the models directory based on platform and runtime context.

    Priority:
    1. MANOLE_MODELS_DIR env var (always wins)
    2. Dev mode (not frozen): ./models/ relative path
    3. Packaged mode (frozen): platform-specific user data dir
    """
    env_override = os.environ.get("MANOLE_MODELS_DIR")
    if env_override:
        return Path(env_override)

    if not getattr(sys, "frozen", False):
        return Path("models")

    # Packaged mode: platform-specific paths
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Manole" / "models"
    else:
        return Path.home() / ".local" / "share" / "manole" / "models"


def _manifest_lookup() -> dict[str, dict]:
    """Return manifest models keyed by id."""
    manifest = load_manifest()
    return {m["id"]: m for m in manifest["models"]}


class ModelManager:
    """Text and vision GGUF models loaded via llama-cpp-python.

    All inference calls are serialized via an internal lock since
    llama-cpp-python is not thread-safe.
    """

    _lock = threading.Lock()

    # Backward-compat class constants (sourced from manifest at import time)
    _MANIFEST = _manifest_lookup()
    TEXT_REPO_ID = _MANIFEST["text-model"]["repo_id"]
    VL_REPO_ID = _MANIFEST["vision-model"]["repo_id"]
    del _MANIFEST

    def __init__(self, model_path: str | None = None,
                 vision_model_path: str | None = None,
                 mmproj_path: str | None = None,
                 n_threads: int = 4):
        manifest = _manifest_lookup()
        models_dir = get_models_dir()

        self.model_path = model_path or str(
            models_dir / manifest["text-model"]["filename"]
        )
        self.vision_model_path = vision_model_path or str(
            models_dir / manifest["vision-model"]["filename"]
        )
        self.mmproj_path = mmproj_path or str(
            models_dir / manifest["vision-projector"]["filename"]
        )

        self.n_threads = n_threads
        self.model = None
        self._vision_model = None

    @property
    def _repo_ids(self) -> dict[str, str]:
        """Lazy-load repo IDs from manifest (supports __new__ bypass)."""
        if not hasattr(self, "_repo_ids_cache"):
            manifest = _manifest_lookup()
            self._repo_ids_cache = {
                m_id: m["repo_id"] for m_id, m in manifest.items()
            }
        return self._repo_ids_cache

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
            from llama_cpp.llama_chat_format import MoondreamChatHandler

            self._ensure_model(
                self.vision_model_path,
                self._repo_ids["vision-model"],
                Path(self.vision_model_path).name,
            )
            self._ensure_model(
                self.mmproj_path,
                self._repo_ids["vision-projector"],
                Path(self.mmproj_path).name,
            )

            try:
                chat_handler = MoondreamChatHandler(clip_model_path=self.mmproj_path)
            except Exception as e:
                raise RuntimeError(
                    f"Vision model mmproj is incompatible with MoondreamChatHandler. "
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
        self._ensure_model(
            self.model_path,
            self._repo_ids["text-model"],
            Path(self.model_path).name,
        )
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
