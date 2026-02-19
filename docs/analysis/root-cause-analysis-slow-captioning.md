# Root Cause Analysis: Slow Image Captioning During App Loading

**Date:** 2026-02-18
**Analyst:** Rex (Root Cause Analysis Specialist)
**Status:** Complete
**Severity:** High -- blocks user interaction during app initialization

---

## Problem Statement

After moving image captioning from background to foreground (inline in `handle_init`), users see the captioning step in the LoadingScreen and it takes "a very very long time." The app is unusable until captioning completes because `handle_init` is synchronous.

**Scope:** Image captioning pipeline during `server.py:handle_init` -> `ImageCaptioner.run()` -> `ModelManager.caption_image()`.

**Affected files:**
- `/Users/ded/Projects/assist/manole/server.py` (lines 254-282)
- `/Users/ded/Projects/assist/manole/image_captioner.py`
- `/Users/ded/Projects/assist/manole/models.py`
- `/Users/ded/Projects/assist/manole/caption_cache.py`

---

## Toyota 5 Whys Analysis

### Branch A: Model Inference Is Slow Per Image

**WHY 1 (Symptom):** Each image takes a long time to caption.
- **Evidence:** `caption_image()` calls `self.vision_model.create_chat_completion()` with a full VLM inference pass per image (`models.py:72-84`). Local VLM inference on CPU is inherently slow (typically 5-15 seconds per image for moondream2 f16).

**WHY 2 (Context):** The vision model is running full f16 precision inference on CPU.
- **Evidence:** The model file is `moondream2-text-model-f16_ct-vicuna.gguf` at **2.64 GB** (f16 precision). The text model used for chat is Q4_0 quantized at 664 MB. The vision model is 4x larger due to no quantization. See `models.py:15`: `DEFAULT_VISION_MODEL_PATH = "models/moondream2-text-model-f16_ct-vicuna.gguf"`. File listing confirms: `ls -lh` shows 2.6G for moondream2 vs 664M for LFM2.5 Q4_0.
- **Note:** An alternative quantized VL model already exists on disk: `LFM2.5-VL-1.6B-Q4_0.gguf` (664 MB) with its mmproj `mmproj-LFM2.5-VL-1.6b-F16.gguf` (814 MB).

**WHY 3 (System):** No quantized moondream2 variant was selected, and no GPU offloading is configured.
- **Evidence:** `n_gpu_layers` is never set in the Llama constructor (`models.py:62-68`). Only `n_threads=4` is configured. The model runs entirely on CPU with f16 weights, meaning every inference step processes 2.6 GB of weights through CPU without any acceleration.

**WHY 4 (Design):** The model selection prioritized accuracy over inference speed; no performance benchmarking was done for the foreground use case.
- **Evidence:** The moondream2 f16 model was likely chosen for caption quality. The existing alternative `LFM2.5-VL-1.6B-Q4_0.gguf` (Q4_0 quantized, 664 MB) is already downloaded but not wired up for captioning. The `preprocess.py` file references LFM2.5-VL as an alternative VL pipeline.

**WHY 5 (Root Cause):** **ROOT CAUSE A: The vision model uses f16 precision (2.6 GB) without quantization or GPU offloading, making each inference call 3-5x slower than necessary for the foreground captioning use case.**

---

### Branch B: Images Are Processed Sequentially With No Parallelism

**WHY 1 (Symptom):** Total captioning time scales linearly with the number of uncached images (N images x T seconds/image).
- **Evidence:** `image_captioner.py:58`: `for img in uncached:` -- a simple sequential loop over all uncached images.

**WHY 2 (Context):** Each image goes through: file I/O -> PIL open -> RGB convert -> JPEG encode -> base64 encode -> model inference, all in series.
- **Evidence:** `_load_image_as_data_uri()` (lines 111-125) does PIL open, RGB conversion, JPEG save at quality=85, and base64 encoding for every single image. No pre-loading, no batching of the I/O-heavy steps.

**WHY 3 (System):** The model's threading lock (`_lock = threading.Lock()` in `models.py:18`) serializes all inference, but image I/O could be parallelized or pipelined.
- **Evidence:** The lock at `models.py:72` (`with self._lock:`) means only one inference call can run at a time (correct, since llama-cpp is not thread-safe). However, image loading and encoding (the I/O portion) does not need the lock and could be done concurrently while the previous image is being captioned.

**WHY 4 (Design):** The captioner was originally a background task where sequential processing was acceptable. Moving it to foreground changed the latency requirements without changing the architecture.
- **Evidence:** The comment in `server.py:254` says `# --- Inline image captioning ---`. The captioner class itself has no concept of parallelism or batching.

**WHY 5 (Root Cause):** **ROOT CAUSE B: Sequential processing without I/O pipelining means total wall-clock time is sum(image_load_time + inference_time) for all images, rather than overlapping I/O with inference.**

---

### Branch C: No Image Downscaling Before Captioning

**WHY 1 (Symptom):** Large images (e.g., 4000x3000 from a phone camera) are sent to the model at full resolution after JPEG encoding.
- **Evidence:** `_load_image_as_data_uri()` at line 121-124: `img = Image.open(path)` then `img.convert("RGB").save(buf, format="JPEG", quality=85)`. No resize, no thumbnail. A 12MP photo at quality=85 JPEG produces a ~2-4 MB base64 string.

**WHY 2 (Context):** The moondream2 model internally resizes images to its input resolution (typically 384x384 or 512x512), so sending a 4000x3000 image wastes bandwidth and processing time on the base64 encoding/decoding without any quality benefit.
- **Evidence:** VLM models have fixed input resolutions. The extra pixels are discarded during preprocessing inside the chat handler. The large base64 string must still be parsed and decoded by llama-cpp before it can be resized.

**WHY 3 (System):** No image preprocessing pipeline exists between file loading and model inference.
- **Evidence:** Searching for `resize`, `thumbnail`, `MAX_SIZE`, or `downscale` across all Python files yields zero results.

**WHY 4 (Design):** The `_load_image_as_data_uri` method was written for correctness (any image in, data URI out) without considering the model's actual input requirements.
- **Evidence:** The function signature and implementation show a generic image-to-data-uri converter with no model-awareness.

**WHY 5 (Root Cause):** **ROOT CAUSE C: Images are not downscaled before encoding, causing unnecessary I/O overhead (large base64 strings) and wasted decoding time inside the model's preprocessing, especially for high-resolution photos.**

---

### Branch D: Duplicate Work in server.py

**WHY 1 (Symptom):** `_find_images()` is called twice, and cache is checked twice for every image.
- **Evidence:** `server.py:269-270` calls `captioner._find_images()` and iterates all images checking `cache.get(str(img))`. Then `captioner.run()` at line 273 calls `_find_images()` again (line 34) and checks `cache.get()` again for every image (line 42-46).

**WHY 2 (Context):** The duplicate call exists because server.py needs the uncached count to decide whether to send the "captioning" status message, but the ImageCaptioner API does not expose this count without running the scan.
- **Evidence:** Lines 269-272 show the pattern: scan images, count uncached, conditionally send status. Then `run()` repeats the same scan internally.

**WHY 3 (System):** `CaptionCache._key()` calls `Path(path).stat()` for every cache lookup to include mtime in the hash key. This means every duplicate check also duplicates a filesystem stat call.
- **Evidence:** `caption_cache.py:24-25`: `mtime = str(p.stat().st_mtime) if p.exists() else "0"`. Each `cache.get()` call does `p.exists()` + `p.stat()` + MD5 hash + file existence check on the cache file.

**WHY 4 (Design):** The ImageCaptioner was designed as a self-contained unit (scan + caption + inject). The server wraps it but needs pre-scan information the API does not provide.
- **Evidence:** `ImageCaptioner.run()` is a monolithic method that does everything internally. No separate `scan()` or `count_uncached()` method exists.

**WHY 5 (Root Cause):** **ROOT CAUSE D: The ImageCaptioner API does not expose scan results, forcing the server to duplicate filesystem traversal and cache lookups. While this is not the primary bottleneck, it adds unnecessary latency proportional to image count.**

---

### Branch E: Lazy Vision Model Loading Adds Cold-Start Latency

**WHY 1 (Symptom):** The first image captioned takes significantly longer than subsequent images.
- **Evidence:** `models.py:45-69`: The `vision_model` property is lazy-loaded. On first access, it loads the 2.6 GB model file from disk, initializes the MoondreamChatHandler with the 868 MB mmproj file, and creates the Llama instance. This is ~3.5 GB of disk I/O plus model initialization.

**WHY 2 (Context):** The vision model is not loaded during the `model.load()` call in `handle_init`. Only the text model is loaded eagerly.
- **Evidence:** `models.py:86-95`: `load()` only loads the text model (`self.model_path`). The vision model is loaded on first `caption_image()` call via the lazy property.

**WHY 3 (System):** The text and vision models are treated as independent, with vision loaded only on demand.
- **Evidence:** `ModelManager.__init__` sets `self._vision_model = None`. No `load_vision()` method exists.

**WHY 4 (Design):** Lazy loading was appropriate when captioning was background/optional. Now that it runs inline during init, the cold-start penalty is felt directly by the user.

**WHY 5 (Root Cause):** **ROOT CAUSE E: Vision model lazy loading adds a one-time ~3.5 GB disk I/O and initialization penalty to the first captioning call, which is now on the critical path of app startup.**

---

## Backwards Chain Validation

| Root Cause | Forward Trace | Validates? |
|---|---|---|
| A: f16 model, no GPU offload | f16 weights -> 4x memory bandwidth -> slow inference per image | Yes |
| B: Sequential processing | N images x (I/O + inference) -> linear scaling | Yes |
| C: No image downscaling | Large base64 -> extra decode time -> wasted cycles | Yes |
| D: Duplicate scan/cache checks | 2x filesystem traversal + 2x stat per image | Yes |
| E: Lazy vision model loading | First call loads 3.5 GB -> cold-start spike | Yes |

**Cross-validation:** Root causes A, B, C, and E are independent and additive. For a directory with 20 uncached images:
- E adds ~5-10s one-time (model load)
- A makes each image ~5-15s instead of ~1-3s
- C adds ~0.5-1s per large image (encoding overhead)
- B means total = E + 20 * (A + C) = potentially 5 + 20 * 12 = **245 seconds** (4+ minutes)
- D adds minor overhead (~0.5-2s total for 20 images)

---

## Solutions

### Immediate Mitigations (Restore acceptable UX)

**M1: Switch to quantized vision model (Q4_0)**
- Replace moondream2 f16 (2.6 GB) with `LFM2.5-VL-1.6B-Q4_0.gguf` (664 MB) already on disk.
- Expected improvement: 3-5x faster inference per image.
- File: `models.py` -- change `DEFAULT_VISION_MODEL_PATH` and `DEFAULT_MMPROJ_PATH`.
- Risk: Minor quality reduction in captions (acceptable for search indexing).

**M2: Add image downscaling before encoding**
- In `_load_image_as_data_uri()`, add `img.thumbnail((768, 768), Image.LANCZOS)` before JPEG encoding.
- Expected improvement: Reduces base64 payload by 5-10x for large images, saving encoding and model preprocessing time.
- File: `image_captioner.py` line 121-124.

**M3: Eagerly load vision model during init**
- Call `self.model.vision_model` (triggering the lazy load) right after `self.model.load()` in `handle_init`, with a "loading_vision_model" status message.
- Expected improvement: Vision model load happens during the existing "loading_model" phase, not during captioning. Removes cold-start surprise.
- File: `server.py` around line 186.

### Permanent Fixes (Prevent recurrence)

**P1: Pipeline I/O with inference**
- Use a producer-consumer pattern: one thread pre-loads and encodes images into a queue, the main thread consumes encoded images for inference.
- Expected improvement: Overlaps I/O with inference, reducing total time by the I/O fraction (~10-20%).
- File: `image_captioner.py` -- refactor `run()` to use `concurrent.futures.ThreadPoolExecutor` for image loading.

**P2: Expose scan results from ImageCaptioner API**
- Add a `scan()` method that returns `(cached_count, uncached_count, images)` and have `run()` accept pre-scanned results.
- Eliminates duplicate filesystem traversal and cache lookups.
- File: `image_captioner.py` -- split `run()` into `scan()` + `caption(scan_results)`.

**P3: Add progress estimation to UI**
- Since captioning now blocks init, provide accurate time estimates: "Captioning image 3/20 (~45s remaining)".
- Track per-image timing and compute rolling average for estimates.
- File: `image_captioner.py` -- add timing to the progress callback.

**P4: Consider batch captioning support**
- If the VLM supports batch inference (multiple images in one forward pass), implement batch processing.
- Note: moondream2 via llama-cpp may not support this, but LFM2.5-VL via transformers might (see `preprocess.py` for the alternative pipeline).

### Early Detection

**D1: Add timing instrumentation to captioning**
- Log per-image captioning time and total captioning time. Alert if average exceeds threshold.
- This prevents regressions when models or configurations change.

---

## Recommended Priority

| Priority | Action | Impact | Effort |
|---|---|---|---|
| 1 | M1: Switch to Q4_0 vision model | ~3-5x speedup | Low (config change) |
| 2 | M2: Add image downscaling | ~1.5-2x speedup on large images | Low (3 lines) |
| 3 | M3: Eager vision model loading | Eliminates cold-start spike | Low (2 lines) |
| 4 | P2: Fix duplicate scan | Cleaner API, minor speedup | Medium |
| 5 | P1: Pipeline I/O | ~10-20% further improvement | Medium |
| 6 | P3: Progress estimation | UX improvement | Low |

**Combined impact of M1 + M2 + M3:** Estimated reduction from ~245s to ~30-50s for 20 uncached images (5-8x improvement).

---

## Evidence Summary

| Evidence | Location | Finding |
|---|---|---|
| Vision model file size | `models/moondream2-text-model-f16_ct-vicuna.gguf` | 2.64 GB, f16 precision |
| Alternative model on disk | `models/LFM2.5-VL-1.6B-Q4_0.gguf` | 664 MB, Q4_0 quantized |
| Sequential loop | `image_captioner.py:58` | `for img in uncached:` |
| No image resize | `image_captioner.py:111-125` | No `resize`/`thumbnail` call |
| Lazy vision model | `models.py:45-69` | Loaded on first `caption_image()` |
| Duplicate scan | `server.py:269` + `image_captioner.py:34` | `_find_images()` called twice |
| No GPU offload | `models.py:62-68` | `n_gpu_layers` never set |
| Threading lock | `models.py:18,72` | Serializes all inference |
| Cache stat overhead | `caption_cache.py:24-25` | `p.stat()` on every lookup |
