"""Background image captioner: scans directory, captions via VL model, injects into LEANN index."""
import base64
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections.abc import Callable

from leann import LeannBuilder
from caption_cache import CaptionCache

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.webp',
    '.heic', '.heif', '.bmp', '.tiff', '.tif',
}


class ImageCaptioner:
    """Scans a directory for images, captions them via VL model, injects into LEANN index."""

    def __init__(self, model, index_path: str, cache: CaptionCache,
                 data_dir: str, send_fn: Callable, dir_id: str,
                 debug: bool = False):
        self.model = model
        self.index_path = index_path
        self.cache = cache
        self.data_dir = Path(data_dir)
        self.send_fn = send_fn
        self.dir_id = dir_id
        self.debug = debug

    def run(self) -> None:
        images = self._find_images()
        if not images:
            return

        # Separate cached from uncached
        uncached = []
        cached_captions: list[tuple[Path, str]] = []
        for img in images:
            caption = self.cache.get(str(img))
            if caption is not None:
                cached_captions.append((img, caption))
            else:
                uncached.append(img)

        total = len(uncached)

        # Always notify UI that captioning phase has started
        self.send_fn(None, "status", {"state": "captioning"})

        # Caption uncached images
        new_captions: list[tuple[Path, str]] = []
        if total > 0:
            log.info(f"{total} uncached images to caption")
            if self.debug:
                print(f"[CAPTIONER] {total} uncached images to caption")

            done = 0
            with ThreadPoolExecutor(max_workers=1) as executor:
                # Submit first image load
                next_future = executor.submit(self._load_image_as_data_uri, uncached[0])

                for i, img in enumerate(uncached):
                    try:
                        if self.debug:
                            print(f"[CAPTIONER] Captioning {img.name}...")
                        # Get current image's data URI (already loading or loaded)
                        data_uri = next_future.result()

                        # Pre-load next image while this one is being captioned
                        if i + 1 < len(uncached):
                            next_future = executor.submit(self._load_image_as_data_uri, uncached[i + 1])

                        caption = self.model.caption_image(data_uri)
                        self.cache.put(str(img), caption)
                        new_captions.append((img, caption))
                        done += 1
                        if self.debug:
                            print(f"[CAPTIONER] {done}/{total} done: {img.name} -> {caption[:60]}")
                        self.send_fn(None, "captioning_progress", {
                            "directoryId": self.dir_id,
                            "done": done,
                            "total": total,
                        })
                    except Exception as exc:
                        log.warning(f"Error captioning {img.name}: {exc}")
                        if self.debug:
                            print(f"[CAPTIONER] Error captioning {img.name}: {exc}")
                        # If preload failed for next image, we need to re-submit
                        if i + 1 < len(uncached):
                            next_future = executor.submit(self._load_image_as_data_uri, uncached[i + 1])
                        continue

        # Only inject when there are new captions to add
        if new_captions:
            try:
                self._inject_captions(new_captions)
                if self.debug:
                    print(f"[CAPTIONER] Injected {len(new_captions)} new captions into index")
            except Exception as exc:
                log.warning(f"Failed to inject captions into index: {exc}")
                if self.debug:
                    print(f"[CAPTIONER] Failed to inject captions: {exc}")

        if self.debug:
            if total > 0:
                print(f"[CAPTIONER] Complete: {len(new_captions)}/{total} images captioned")
            elif cached_captions:
                print(f"[CAPTIONER] All {len(cached_captions)} images already cached, injected into index")
        self.send_fn(None, "captioning_progress", {
            "directoryId": self.dir_id,
            "done": total,
            "total": total,
            "state": "complete",
        })

    def _find_images(self) -> list[Path]:
        images = []
        for f in self.data_dir.rglob("*"):
            if f.is_symlink():
                continue
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(f)
        return sorted(images)

    def _load_image_as_data_uri(self, path: Path) -> str:
        from PIL import Image

        if path.suffix.lower() in ('.heic', '.heif'):
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except ImportError:
                raise RuntimeError("pillow-heif required for HEIC support")

        img = Image.open(path)
        img = img.convert("RGB")
        img.thumbnail((768, 768), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    def _inject_captions(self, captions: list[tuple[Path, str]]) -> None:
        """Batch-inject all captions into the LEANN index in a single update."""
        import hashlib
        builder = LeannBuilder(backend_name="hnsw", embedding_model="facebook/contriever", is_recompute=False)
        for image_path, caption in captions:
            # Use a unique ID based on image path hash to avoid collision
            # with existing passage IDs (which start from "0")
            chunk_id = f"img_{hashlib.sha256(str(image_path).encode()).hexdigest()[:12]}"
            builder.add_text(
                text=f"Photo description: {caption}",
                metadata={
                    "file_path": str(image_path),
                    "file_name": image_path.name,
                    "file_type": "image",
                    "path": str(image_path),
                    "id": chunk_id,
                },
            )
        builder.update_index(self.index_path)
