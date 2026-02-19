"""Persistent file-based caption cache for image descriptions."""
import hashlib
from pathlib import Path


class CaptionCache:
    """File-based caption cache. Key = sha256(path + mtime). Persistent across sessions."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, image_path: str) -> str | None:
        cache_file = self.cache_dir / f"{self._key(image_path)}.txt"
        if cache_file.exists():
            return cache_file.read_text()
        return None

    def put(self, image_path: str, caption: str) -> None:
        cache_file = self.cache_dir / f"{self._key(image_path)}.txt"
        cache_file.write_text(caption)

    def _key(self, path: str) -> str:
        p = Path(path)
        mtime = str(p.stat().st_mtime) if p.exists() else "0"
        return hashlib.sha256(f"{path}:{mtime}".encode()).hexdigest()
