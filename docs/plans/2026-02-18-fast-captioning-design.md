# Design: Fast Image Captioning (5 Root Cause Fixes)

## Problem
Image captioning takes ~245s for 20 uncached images. Five root causes identified in `docs/analysis/root-cause-analysis-slow-captioning.md`.

## Fix A: Switch to Q4_0 Vision Model (highest impact)

**Change**: `models.py` — replace moondream2 f16 (2.6 GB) with LFM2.5-VL Q4_0 (664 MB).

```python
# Before
DEFAULT_VISION_MODEL_PATH = "models/moondream2-text-model-f16_ct-vicuna.gguf"
DEFAULT_MMPROJ_PATH = "models/moondream2-mmproj-f16-20250414.gguf"
VL_REPO_ID = "ggml-org/moondream2-20250414-GGUF"

# After
DEFAULT_VISION_MODEL_PATH = "models/LFM2.5-VL-1.6B-Q4_0.gguf"
DEFAULT_MMPROJ_PATH = "models/mmproj-LFM2.5-VL-1.6b-F16.gguf"
VL_REPO_ID = "LiquidAI/LFM2.5-VL-1.6B-GGUF"
```

**Chat handler change**: MoondreamChatHandler → Llava15ChatHandler (generic llama-cpp-python vision handler). Both use same API: `handler = Handler(clip_model_path=...)`. The error message and test assertions need updating.

**Risk**: Caption quality may differ. Both are ~1.6B VL models. LFM2.5-VL is from Liquid AI (same vendor as the text model already in use).

**Files**: `models.py`, `tests/test_vision_models.py`

## Fix B: Pipeline I/O with Inference

**Change**: `image_captioner.py` — use a producer-consumer pattern where one thread pre-loads images while the main thread runs inference.

```python
# In run(), replace sequential loop with:
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

def _preload(img_path):
    return (img_path, self._load_image_as_data_uri(img_path))

with ThreadPoolExecutor(max_workers=1) as executor:
    # Submit first image load
    futures = [executor.submit(_preload, uncached[0])]

    for i, img in enumerate(uncached):
        img_path, data_uri = futures[0].result()
        futures.pop(0)

        # Pre-load next image while this one is being captioned
        if i + 1 < len(uncached):
            futures.append(executor.submit(_preload, uncached[i + 1]))

        caption = self.model.caption_image(data_uri)
        # ... rest of per-image logic
```

**Files**: `image_captioner.py`, `tests/test_image_captioner.py`

## Fix C: Image Downscaling

**Change**: `image_captioner.py` `_load_image_as_data_uri()` — add `thumbnail()` before JPEG encoding.

```python
# After img = Image.open(path), add:
img = img.convert("RGB")
img.thumbnail((768, 768), Image.LANCZOS)
# Then encode to JPEG as before
```

768px is generous — VL models typically use 384-512px input. This reduces a 12MP photo from ~3MB base64 to ~100KB.

**Files**: `image_captioner.py`, `tests/test_image_captioner.py`

## Fix D: Eliminate Duplicate Filesystem Work

**Change**: Split `ImageCaptioner.run()` into `scan()` + `run(scan_result)`.

```python
class ImageCaptioner:
    def scan(self) -> tuple[list[Path], int]:
        """Return (all_images, uncached_count) without captioning."""
        images = self._find_images()
        uncached_count = sum(1 for img in images if self.cache.get(str(img)) is None)
        return images, uncached_count

    def run(self) -> None:
        # Existing logic, but uses internal scan only (no change to external API)
        ...
```

Then `server.py` calls `captioner.scan()` to decide whether to send "captioning" status, then `captioner.run()`. But since `run()` still does its own internal scan, we need to pass the pre-scanned results. Simpler: just let `run()` return (or set) the uncached count, and have server.py always call `run()` — if no uncached images, `run()` returns instantly. The only thing to gate is the "captioning" status event.

**Simplest approach**: Add a `has_uncached()` method that does the scan once and caches the result internally, then `run()` uses the cached scan. Or: just accept the duplicate scan — it's < 1s overhead.

**Decision**: Keep it simple. The duplicate scan adds maybe 0.5s for 20 images. Not worth the API complexity. Instead, just remove the duplicate scan from `server.py` and send the "captioning" status unconditionally if images exist (or skip it if `_find_images()` returns empty). The UI already handles the case gracefully.

Actually, simplest fix: move the uncached count check INTO `ImageCaptioner.run()` and have it send the status event itself. Then server.py just calls `captioner.run()` and doesn't need to pre-scan.

**Files**: `image_captioner.py`, `server.py`, `tests/test_server.py`

## Fix E: Eager Vision Model Loading

**Change**: `models.py` — add `load_vision()` method. `server.py` — call it during model load phase.

```python
# models.py
def load_vision(self):
    """Eagerly load the vision model (triggers lazy property)."""
    _ = self.vision_model

# server.py, in handle_init, after self.model.load():
send(None, "status", {"state": "loading_model"})
self.model.load()
self.model.load_vision()  # Load VL model during "Loading model" phase
```

**Files**: `models.py`, `server.py`, `tests/test_vision_models.py`, `tests/test_server.py`

## Implementation Order

| Step | Fix | Impact | Complexity | Files |
|------|-----|--------|-----------|-------|
| 01-01 | A: Q4_0 model + Llava15ChatHandler | ~3-5x per image | Low | models.py, tests/test_vision_models.py |
| 01-02 | E: Eager vision model loading | Eliminates cold start | Low | models.py, server.py, tests |
| 02-01 | C: Image downscaling (thumbnail 768px) | ~1.5x per image | Low | image_captioner.py, tests |
| 02-02 | D: Eliminate duplicate scan | Clean API | Low | image_captioner.py, server.py, tests |
| 02-03 | B: I/O pipeline | ~10-20% further | Medium | image_captioner.py, tests |

## Expected Result

For 20 uncached images:
- **Before**: ~245s (f16 model, full-res images, cold start, sequential)
- **After**: ~30-40s (Q4_0 model, thumbnails, warm start, pipelined I/O)
- **Improvement**: ~6-8x
