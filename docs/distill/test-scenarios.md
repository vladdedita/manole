# Vision Pipeline: Test Scenarios

**Date**: 2026-02-17
**Framework**: pytest (follows existing project conventions)

## Test File Mapping

| Test File | Module Under Test | User Stories | AC Coverage |
|-----------|-------------------|-------------|-------------|
| `test_caption_cache.py` | `caption_cache.py` | US-2 | AC-6, AC-7, AC-8 |
| `test_vision_models.py` | `models.py` (vision extensions) | US-5 | AC-15, AC-16, AC-17 |
| `test_image_captioner.py` | `image_captioner.py` | US-1, US-3, US-4, US-6 | AC-2 through AC-5, AC-9 through AC-14, AC-18, AC-20 |

## Implementation Order (One-at-a-Time)

### Milestone 1: CaptionCache (`test_caption_cache.py`)
Pure Python, no external deps. Foundation for everything else.
- `test_put_then_get_returns_caption`
- `test_cache_persists_across_instances`
- `test_get_returns_none_for_uncached`
- `test_cache_invalidated_when_file_modified`
- `test_different_paths_produce_different_keys`
- `test_cache_creates_directory_if_missing`

### Milestone 2: ModelManager Vision Extensions (`test_vision_models.py`)
Adds vision_model property and caption_image method.
- `test_vision_model_not_loaded_on_init`
- `test_vision_model_lazy_loads_on_first_access`
- `test_vision_model_loaded_only_once`
- `test_default_vision_model_path`
- `test_custom_vision_model_path`
- `test_caption_image_calls_vision_model`
- `test_caption_image_sends_multimodal_message`

### Milestone 3: ImageCaptioner Core (`test_image_captioner.py`)
Background captioning orchestrator with LEANN injection.
- `test_all_images_captioned`
- `test_caption_injected_with_correct_metadata`
- `test_corrupted_image_skipped`
- `test_cached_images_not_recaptioned`
- `test_finds_all_image_formats`
- `test_excludes_non_image_files`

### Milestone 4: Progress Reporting (`test_image_captioner.py`)
NDJSON progress messages.
- `test_progress_messages_sent`
- `test_completion_message_sent`
- `test_no_progress_for_text_only_directory`
- `test_vision_model_not_loaded_for_empty_dir`

### Milestone 5: Server Integration
Wiring in server.py — tested manually or via integration test.
- Background thread starts after handle_init
- Captioning doesn't block query handling

### Not Covered in Unit Tests (Require Real Models)
- AC-17: Both models coexist in memory (< 2.5GB) — manual/integration
- AC-19: HEIC conversion via pillow-heif — requires real HEIC file + pillow-heif
- Real VL model captioning quality — requires LFM2.5-VL GGUF

## Test Conventions

Following existing project patterns:
- Plain pytest functions (no classes, no BDD framework)
- Mocks via `unittest.mock.MagicMock` and `patch`
- Fake implementations (e.g., `FakeSearcher`) for simple stubs
- `tempfile.mkdtemp()` / `TemporaryDirectory` for test fixtures
- No test fixtures in conftest.py (inline per test file)
