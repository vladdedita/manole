# Vision Pipeline: Architecture Design

**Date**: 2026-02-17
**Status**: Draft
**Wave**: DESIGN
**Input**: `docs/requirements/requirements.md`, `docs/requirements/acceptance-criteria.md`

## 1. Architecture Overview

The vision pipeline adds background image captioning to manole's existing indexing flow.
No new agent tools. No new query paths. Captions become searchable chunks in the LEANN
index like any other document content.

```
┌──────────────────────────────────────────────────────────────────┐
│                          server.py                               │
│                                                                  │
│  handle_init()                                                   │
│    │                                                             │
│    ├─ 1. Load text model (existing)                              │
│    ├─ 2. Build LEANN index via `leann build` CLI (existing)      │
│    ├─ 3. Wire components (existing)                              │
│    ├─ 4. Send "ready" to UI (existing)                           │
│    ├─ 5. Generate summary (existing)                             │
│    └─ 6. NEW: Start background image captioning thread ──────┐   │
│                                                               │   │
│  ┌────────────────────────────────────────────────────────────┘   │
│  │  ImageCaptioner (background thread)                           │
│  │    ├─ Scan directory for image files                          │
│  │    ├─ Filter out cached images (CaptionCache)                 │
│  │    ├─ Lazy-load VL model (ModelManager.vision_model)          │
│  │    ├─ For each uncaptioned image:                             │
│  │    │    ├─ Convert if HEIC → JPEG                             │
│  │    │    ├─ Caption via VL model                               │
│  │    │    ├─ Store in CaptionCache                              │
│  │    │    ├─ Inject into LEANN index (LeannBuilder.update_index)│
│  │    │    └─ Send captioning_progress NDJSON message            │
│  │    └─ Send final completion message                           │
│  └───────────────────────────────────────────────────────────────│
│                                                                  │
│  handle_query() ← unchanged, searches LEANN index as before     │
└──────────────────────────────────────────────────────────────────┘
```

## 2. New Modules

### 2.1 `caption_cache.py` — Persistent Caption Cache

```python
class CaptionCache:
    """File-based caption cache. Key = md5(path + mtime). Persistent across sessions."""

    def __init__(self, cache_dir: str = ".neurofind/captions")
    def get(self, image_path: str) -> str | None
    def put(self, image_path: str, caption: str) -> None
    def _key(self, path: str) -> str  # md5(path + mtime)
```

- Cache dir: `.neurofind/captions/` (inside the indexed directory)
- One `.txt` file per caption, keyed by hash
- Invalidation: mtime changes → different hash → cache miss → re-caption

### 2.2 `image_captioner.py` — Background Captioning Orchestrator

```python
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp',
                    '.heic', '.heif', '.bmp', '.tiff', '.tif'}

class ImageCaptioner:
    """Scans a directory for images, captions them via VL model, injects into index."""

    def __init__(self, model: ModelManager, index_path: str, cache: CaptionCache,
                 data_dir: str, send_fn: Callable, dir_id: str, debug: bool = False)

    def run(self) -> None
        """Main loop: scan → filter cached → caption → inject → report progress."""

    def _find_images(self) -> list[Path]
        """Recursively find image files in data_dir."""

    def _load_image(self, path: Path) -> bytes
        """Load image as JPEG bytes. Converts HEIC/HEIF via Pillow."""

    def _caption(self, image_bytes: bytes) -> str
        """Run VL model inference on image bytes."""

    def _inject_caption(self, image_path: Path, caption: str) -> None
        """Add caption chunk to LEANN index via LeannBuilder.add_text + update_index."""
```

## 3. Modified Modules

### 3.1 `models.py` — ModelManager Changes

Add vision model support:

```python
class ModelManager:
    DEFAULT_MODEL_PATH = "models/LFM2.5-1.2B-Instruct-Q4_0.gguf"
    DEFAULT_VISION_MODEL_PATH = "models/LFM2.5-VL-1.6B-Q4_0.gguf"  # NEW

    def __init__(self, model_path=None, vision_model_path=None, n_threads=4):
        # ... existing ...
        self.vision_model_path = vision_model_path or self.DEFAULT_VISION_MODEL_PATH  # NEW
        self._vision_model = None  # NEW

    @property
    def vision_model(self):  # NEW — lazy loaded
        """Load VL model on first access."""
        if self._vision_model is None:
            from llama_cpp import Llama
            self._vision_model = Llama(
                model_path=self.vision_model_path,
                n_ctx=4096,
                n_threads=self.n_threads,
                verbose=False,
            )
        return self._vision_model

    def caption_image(self, image_data_uri: str) -> str:  # NEW
        """Caption an image using the VL model. Input: base64 data URI."""
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
```

### 3.2 `server.py` — Integration Point

In `handle_init()`, after the existing "send ready + generate summary" block:

```python
# After summary generation, start background image captioning
import threading
from image_captioner import ImageCaptioner
from caption_cache import CaptionCache

cache = CaptionCache(str(data_dir_path / ".neurofind" / "captions"))
captioner = ImageCaptioner(
    model=self.model,
    index_path=index_path,
    cache=cache,
    data_dir=str(data_dir_path),
    send_fn=send,
    dir_id=dir_id,
    debug=self.debug,
)

thread = threading.Thread(target=captioner.run, daemon=True)
thread.start()
self.directories[dir_id]["captioner_thread"] = thread
```

## 4. LEANN Index Injection Strategy

The key technical challenge: appending captions to an existing LEANN index at runtime.

### Approach: LeannBuilder.update_index

```python
from leann import LeannBuilder

# Create a builder with the same embedding settings as the original index
builder = LeannBuilder(backend_name="hnsw", embedding_model="facebook/contriever")

# Add caption text with metadata
builder.add_text(
    text=f"Photo description: {caption}",
    metadata={"file_name": image_name, "file_type": "image", "path": str(image_path)}
)

# Append to existing index
builder.update_index(index_path)
```

**Thread Safety Concern**: `update_index` modifies the HNSW index files on disk.
`LeannSearcher.search()` reads from the same files. We need to verify:

1. Whether LeannSearcher caches the index in memory (likely yes — HNSW is memory-mapped)
2. Whether update_index is safe to call while searches are running
3. If not safe: use a `threading.Lock` around index writes, or batch updates

**Mitigation**: Start with a lock around `update_index` calls. If LeannSearcher needs
to be refreshed after updates, we may need to periodically re-initialize it.

```python
# In ImageCaptioner
self._index_lock = threading.Lock()

def _inject_caption(self, image_path, caption):
    builder = LeannBuilder(backend_name="hnsw", embedding_model="facebook/contriever")
    builder.add_text(
        text=f"Photo description: {caption}",
        metadata={"file_name": image_path.name, "file_type": "image", "path": str(image_path)}
    )
    with self._index_lock:
        builder.update_index(self.index_path)
```

**Alternative if update_index is too disruptive**: Batch all captions, then do a single
`update_index` at the end. Trades immediacy for simplicity.

## 5. HEIC Conversion

```python
def _load_image(self, path: Path) -> bytes:
    """Load image as JPEG bytes. Converts HEIC/HEIF."""
    import io
    from PIL import Image

    if path.suffix.lower() in ('.heic', '.heif'):
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            raise RuntimeError("pillow-heif required for HEIC support")

    img = Image.open(path)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()
```

## 6. Progress Messages

NDJSON protocol extension:

```json
// During captioning (after each image or every N images):
{"id": null, "type": "captioning_progress", "data": {"directoryId": "abc123", "done": 5, "total": 50}}

// On completion:
{"id": null, "type": "captioning_progress", "data": {"directoryId": "abc123", "done": 50, "total": 50, "state": "complete"}}

// On error (captioner failed completely, e.g. VL model won't load):
{"id": null, "type": "captioning_progress", "data": {"directoryId": "abc123", "state": "error", "message": "..."}}
```

## 7. File Changes Summary

| File | Change | Lines (est.) |
|------|--------|-------------|
| `caption_cache.py` | **New** — persistent file-based cache | ~40 |
| `image_captioner.py` | **New** — background captioning orchestrator | ~100 |
| `models.py` | Add `vision_model` property, `caption_image()` | ~30 |
| `server.py` | Start captioner thread in `handle_init()` | ~15 |
| `pyproject.toml` | Add `pillow-heif` dependency | ~1 |

**No changes to**: agent.py, searcher.py, router.py, rewriter.py, tools.py, toolbox.py, chat.py

## 8. Dependencies

| Dependency | Purpose | Already Installed? |
|------------|---------|-------------------|
| `llama-cpp-python` | VL model inference | Yes (v0.3.16) |
| `Pillow` | Image loading, HEIC conversion | Check |
| `pillow-heif` | HEIC/HEIF format support | No — add to deps |
| `leann` | Index update (LeannBuilder) | Yes |

## 9. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| LEANN update_index not thread-safe | High | Lock around writes; verify with test |
| LeannSearcher doesn't see new chunks after update | High | May need periodic searcher re-init; test first |
| VL model GGUF doesn't include mmproj | Medium | Verify model file before building; fallback to separate mmproj |
| llama-cpp-python multimodal API quirks | Medium | Spike test with actual model before implementation |
| Background thread crash kills captioning silently | Low | Wrap run() in try/except, send error message |
| Large directories with 1000+ images | Low | Captioning is O(n) but non-blocking; progress messages keep user informed |

## 10. Implementation Order (for DISTILL/DELIVER waves)

1. **Spike**: Verify VL model + llama-cpp-python multimodal works end-to-end
2. **CaptionCache**: TDD — simple, no external deps
3. **ModelManager extensions**: Add vision_model + caption_image
4. **ImageCaptioner**: Core captioning loop with LEANN injection
5. **Server integration**: Thread start in handle_init, progress messages
6. **HEIC support**: pillow-heif integration
7. **Integration test**: Full flow with real images

## 11. Open Questions for Spike

- [ ] Does `LeannSearcher.search()` see chunks added via `update_index` without re-initialization?
- [ ] Does LFM2.5-VL-1.6B GGUF include the vision projector, or is a separate mmproj file needed?
- [ ] What's the actual per-image captioning latency on the target hardware?
- [ ] Does `pillow-heif` work on macOS ARM64?
