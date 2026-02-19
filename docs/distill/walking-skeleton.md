# Vision Pipeline: Walking Skeleton

**Date**: 2026-02-17

## Skeleton Definition

The walking skeleton for the vision pipeline is the thinnest slice that proves
the end-to-end flow works: **one image gets captioned and becomes searchable**.

```
Image file exists in directory
  → CaptionCache says "not cached"
  → ImageCaptioner loads image as JPEG bytes
  → ModelManager.caption_image() returns a caption string
  → CaptionCache stores the caption
  → LeannBuilder.add_text() + update_index() injects into LEANN index
  → Progress message sent via send()
```

## Walking Skeleton Test

The walking skeleton is `test_all_images_captioned` in `test_image_captioner.py`.
When this test passes with real implementations (not mocks), the entire pipeline works.

## Implementation Order

1. **caption_cache.py** — `CaptionCache` class (get/put/key)
2. **models.py** — Add `vision_model` property + `caption_image()` method
3. **image_captioner.py** — `ImageCaptioner` class with `run()`, `_find_images()`, `_load_image()`, `_caption()`, `_inject_caption()`
4. **server.py** — Start captioner thread in `handle_init()`

Each step has its own test file. Green tests at each step before moving to the next.
