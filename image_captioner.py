"""Background image captioner: scans directory, captions via VL model, injects into LEANN index."""
import base64
import io
import logging
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

        # Caption uncached images
        new_captions: list[tuple[Path, str]] = []
        if total > 0:
            log.info(f"{total} uncached images to caption")
            if self.debug:
                print(f"[CAPTIONER] {total} uncached images to caption")

            done = 0
            for img in uncached:
                try:
                    if self.debug:
                        print(f"[CAPTIONER] Captioning {img.name}...")
                    data_uri = self._load_image_as_data_uri(img)
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
                    continue

        # Inject ALL captions (cached + new) into the index
        all_captions = cached_captions + new_captions
        if all_captions:
            try:
                self._inject_captions(all_captions)
                if self.debug:
                    print(f"[CAPTIONER] Injected {len(all_captions)} captions into index ({len(cached_captions)} cached, {len(new_captions)} new)")
            except Exception as exc:
                log.warning(f"Failed to inject captions into index: {exc}")
                if self.debug:
                    print(f"[CAPTIONER] Failed to inject captions: {exc}")

        if total > 0:
            if self.debug:
                print(f"[CAPTIONER] Complete: {len(new_captions)}/{total} images captioned")
            self.send_fn(None, "captioning_progress", {
                "directoryId": self.dir_id,
                "done": len(new_captions),
                "total": total,
                "state": "complete",
            })
        elif cached_captions and self.debug:
            print(f"[CAPTIONER] All {len(cached_captions)} images already cached, injected into index")

    def _find_images(self) -> list[Path]:
        images = []
        for f in self.data_dir.rglob("*"):
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
            chunk_id = f"img_{hashlib.md5(str(image_path).encode()).hexdigest()[:12]}"
            builder.add_text(
                text=f"Photo description: {caption}",
                metadata={
                    "file_name": image_path.name,
                    "file_type": "image",
                    "path": str(image_path),
                    "id": chunk_id,
                },
            )
        builder.update_index(self.index_path)
