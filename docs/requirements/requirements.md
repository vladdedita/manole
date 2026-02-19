# Vision Pipeline Requirements

**Feature**: LFM2.5-VL-1.6B Image Understanding
**Date**: 2026-02-17
**Status**: Draft (from DISCUSS wave)

## Summary

Add image content understanding to manole by integrating a vision-language model
(LFM2.5-VL-1.6B) that captions images in the background after text indexing completes.
Captions are injected into the LEANN index immediately, making image content searchable
via the existing semantic search pipeline. No new agent tools — images become searchable
like any other indexed content.

## Design Decisions (from discussion)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Captioning strategy | Background after text indexing | Zero impact on indexing speed; queries work immediately |
| Caption storage | Inject into LEANN index immediately | Makes captions searchable via existing semantic search |
| describe_image tool | Not needed | Background captioning covers all images; no on-demand tool |
| HEIC support | Day 1, best effort | Convert via Pillow; skip gracefully if conversion fails |
| Image format support | All common formats | jpg, jpeg, png, gif, webp, heic, heif, bmp, tiff |
| VL model loading | Lazy on first image captioning | No memory cost for text-only directories |
| Caption cache | File-based (caption_cache.py) | Persists across sessions; avoids re-captioning |
| UI status | NDJSON progress messages | Reuses existing `send()` push mechanism |
| Batch limit | No explicit cap | Agent doesn't caption — background process handles all images |
| In-flight queries | Use what's available | Search only already-captioned images; show captioning progress |

## Functional Requirements

### FR-1: Vision Model Management
- ModelManager gains a `vision_model` lazy property
- VL model loads only when first image captioning is requested
- Both models can coexist in memory (~1.4GB total for Q4_0)
- `caption_image(image_path, query=None)` method on ModelManager

### FR-2: Caption Cache
- New `caption_cache.py` module with file-based persistent cache
- Cache key = hash(filepath + mtime) for automatic invalidation
- Cache directory: `.neurofind/captions/` (or configurable)
- `get(image_path) -> str | None`
- `put(image_path, caption)`

### FR-3: Background Image Captioning
- After text indexing completes and directory state is `ready`:
  1. Identify all image files in the indexed directory
  2. Filter out already-cached images
  3. Caption uncaptioned images one by one using VL model
  4. For each caption: store in cache AND inject into LEANN index
- Must not block the main query-handling loop
- Must handle interruption gracefully (partial progress saved via cache)

### FR-4: LEANN Index Injection
- Each caption is added as a searchable chunk with metadata:
  ```
  text: "Photo description: {caption}"
  metadata: { file_name, file_type: "image", path }
  ```
- Injection happens immediately after each caption is generated
- Future semantic searches naturally find these chunks

### FR-5: Image Format Support
- Supported formats: jpg, jpeg, png, gif, webp, heic, heif, bmp, tiff
- HEIC/HEIF files: convert to JPEG in-memory via Pillow before sending to VL model
- Unsupported/corrupted files: skip with warning log, continue processing

### FR-6: Progress Reporting
- NDJSON progress messages sent during background captioning:
  ```json
  {"id": null, "type": "captioning_progress", "data": {"directoryId": "...", "done": 50, "total": 200}}
  ```
- Message sent after each image is captioned (or at reasonable intervals for large sets)
- Final message when captioning is complete:
  ```json
  {"id": null, "type": "captioning_progress", "data": {"directoryId": "...", "done": 200, "total": 200, "state": "complete"}}
  ```

### FR-7: Query Behavior During Captioning
- Queries work normally — semantic search finds whatever captions are already indexed
- No special handling needed in agent/router/searcher
- UI may optionally show "Image captioning in progress (N/M)" based on progress messages

## Non-Functional Requirements

### NFR-1: Memory
- Total RAM with both models: < 2.5 GB (Q4_0 quantization)
- VL model loaded only when images exist in the directory

### NFR-2: Indexing Performance
- Text indexing speed: unchanged (zero overhead)
- Image captioning: ~2-3s per image on CPU (background, non-blocking)

### NFR-3: Resilience
- Session restart resumes captioning (cache persists)
- VL model failure doesn't affect text search
- Individual image failures don't stop batch processing

## Out of Scope
- Real-time image upload/watch (future)
- Video file support
- Multi-turn image conversations ("what else is in that photo?")
- Custom captioning prompts per directory
