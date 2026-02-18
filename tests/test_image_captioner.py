"""Acceptance tests for ImageCaptioner (US-1, US-3, US-4, US-5, US-6).

Milestone 2-5: These tests cover the background captioning orchestrator,
LEANN index injection, progress reporting, and format support.

Tests use mocks for ModelManager and LeannBuilder to avoid loading
real models in unit tests. Integration tests with real models are separate.
"""
import os
import tempfile
import threading
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call


# --- Helpers ---

def _make_test_dir(images=None, texts=None):
    """Create a temp directory with specified image and text files."""
    tmp = tempfile.mkdtemp()
    for name in (images or []):
        (Path(tmp) / name).write_bytes(b"\xff\xd8fake-jpeg-data")
    for name in (texts or []):
        (Path(tmp) / name).write_text("Some text content")
    return tmp


def _make_captioner(data_dir, send_fn=None, cache=None, model=None):
    """Create an ImageCaptioner with mocked dependencies.
    _load_image_as_data_uri is patched to avoid Pillow parsing fake bytes."""
    from image_captioner import ImageCaptioner
    from caption_cache import CaptionCache

    if model is None:
        model = MagicMock()
        model.caption_image.return_value = "A test caption"
        model._vision_model = None

    if cache is None:
        cache = CaptionCache(os.path.join(data_dir, ".neurofind", "captions"))

    if send_fn is None:
        send_fn = MagicMock()

    captioner = ImageCaptioner(
        model=model,
        index_path="/fake/index/path",
        cache=cache,
        data_dir=data_dir,
        send_fn=send_fn,
        dir_id="test-dir-123",
    )
    # Bypass Pillow image loading for unit tests
    captioner._load_image_as_data_uri = MagicMock(
        return_value="data:image/jpeg;base64,/9j/fake"
    )
    return captioner, send_fn, cache


# --- US-1: Background Image Captioning ---

# AC-1: Captioning starts after text indexing
# (Integration-level — tested in test_server.py integration)

# AC-2: All images are captioned
@patch("image_captioner.LeannBuilder")
def test_all_images_captioned(MockBuilder):
    """Given a directory with image files, when captioning runs to completion,
    then all images have captions in the cache and in the index."""
    data_dir = _make_test_dir(images=["cat.jpg", "dog.png", "sunset.gif"])
    captioner, send_fn, cache = _make_captioner(data_dir)

    captioner.run()

    # All 3 images should be cached
    assert cache.get(os.path.join(data_dir, "cat.jpg")) == "A test caption"
    assert cache.get(os.path.join(data_dir, "dog.png")) == "A test caption"
    assert cache.get(os.path.join(data_dir, "sunset.gif")) == "A test caption"

    # LeannBuilder batch: 3 add_text calls, 1 update_index call
    assert MockBuilder.return_value.add_text.call_count == 3
    assert MockBuilder.return_value.update_index.call_count == 1


# AC-3: Captions injected with correct format
@patch("image_captioner.LeannBuilder")
def test_caption_injected_with_correct_metadata(MockBuilder):
    """Given an image is captioned, when injected into the index,
    then text is 'Photo description: ...' and metadata includes file_type='image'."""
    data_dir = _make_test_dir(images=["sunset.png"])
    captioner, _, _ = _make_captioner(data_dir)

    captioner.run()

    add_text_call = MockBuilder.return_value.add_text.call_args
    assert "Photo description:" in add_text_call.kwargs.get("text", add_text_call[0][0] if add_text_call[0] else "")
    metadata = add_text_call.kwargs.get("metadata", {})
    assert metadata.get("file_type") == "image"
    assert metadata.get("file_name") == "sunset.png"


# AC-4: Captioning doesn't block (tested structurally — ImageCaptioner.run is called in a thread)

# AC-5: Captioning handles errors gracefully
@patch("image_captioner.LeannBuilder")
def test_corrupted_image_skipped(MockBuilder):
    """Given a directory with a corrupted image, when captioning encounters it,
    then it's skipped and remaining images are still captioned."""
    data_dir = _make_test_dir(images=["good.jpg", "bad.jpg", "also_good.png"])

    model = MagicMock()
    call_count = [0]
    def side_effect(data_uri):
        call_count[0] += 1
        if call_count[0] == 2:  # second image fails
            raise RuntimeError("Corrupted image data")
        return "Valid caption"
    model.caption_image.side_effect = side_effect
    model._vision_model = None

    captioner, send_fn, cache = _make_captioner(data_dir, model=model)
    captioner.run()

    # Should have attempted all 3, succeeded on 2
    assert model.caption_image.call_count == 3
    # Batch inject: 2 add_text calls (skipped the failed one), 1 update_index
    assert MockBuilder.return_value.add_text.call_count == 2
    assert MockBuilder.return_value.update_index.call_count == 1


# --- US-3: Searchable Image Content ---

# AC-9, AC-10: Tested via AC-2 and AC-3 above (caption format + metadata)

# AC-11: Uncaptioned images not in search
@patch("image_captioner.LeannBuilder")
def test_cached_images_not_recaptioned(MockBuilder):
    """Given some images are already cached, when captioning runs,
    then only uncached images are captioned."""
    data_dir = _make_test_dir(images=["cached.jpg", "new.png"])
    from caption_cache import CaptionCache
    cache = CaptionCache(os.path.join(data_dir, ".neurofind", "captions"))
    cache.put(os.path.join(data_dir, "cached.jpg"), "Already captioned")

    captioner, _, _ = _make_captioner(data_dir, cache=cache)
    captioner.run()

    # Model should only be called for the uncached image
    assert captioner.model.caption_image.call_count == 1
    # Both cached + new captions are injected into the index
    assert MockBuilder.return_value.add_text.call_count == 2
    assert MockBuilder.return_value.update_index.call_count == 1


# --- US-4: Captioning Progress Visibility ---

# AC-12: Progress messages during captioning
@patch("image_captioner.LeannBuilder")
def test_progress_messages_sent(MockBuilder):
    """Given captioning is processing N images, when each completes,
    then a captioning_progress NDJSON message is sent."""
    data_dir = _make_test_dir(images=["a.jpg", "b.jpg", "c.jpg"])
    captioner, send_fn, _ = _make_captioner(data_dir)

    captioner.run()

    # Should have progress messages for each image + completion
    progress_calls = [
        c for c in send_fn.call_args_list
        if c[0][1] == "captioning_progress"
    ]
    assert len(progress_calls) >= 3  # at least one per image

    # Check structure of a progress message
    last_progress = progress_calls[-1]
    data = last_progress[0][2]
    assert "done" in data
    assert "total" in data
    assert data["directoryId"] == "test-dir-123"


# AC-13: Completion message
@patch("image_captioner.LeannBuilder")
def test_completion_message_sent(MockBuilder):
    """Given captioning finishes all images, then a final completion message is sent."""
    data_dir = _make_test_dir(images=["a.jpg", "b.jpg"])
    captioner, send_fn, _ = _make_captioner(data_dir)

    captioner.run()

    progress_calls = [
        c for c in send_fn.call_args_list
        if c[0][1] == "captioning_progress"
    ]
    final = progress_calls[-1]
    data = final[0][2]
    assert data.get("state") == "complete"
    assert data["done"] == data["total"]


# AC-14: No progress messages for text-only directories
@patch("image_captioner.LeannBuilder")
def test_no_progress_for_text_only_directory(MockBuilder):
    """Given a directory with no image files, then no captioning_progress messages are sent."""
    data_dir = _make_test_dir(texts=["readme.txt", "notes.md"])
    captioner, send_fn, _ = _make_captioner(data_dir)

    captioner.run()

    progress_calls = [
        c for c in send_fn.call_args_list
        if c[0][1] == "captioning_progress"
    ]
    assert len(progress_calls) == 0


# --- US-5: Vision Model Lifecycle ---

# AC-15: Lazy loading — VL model accessed only during captioning
@patch("image_captioner.LeannBuilder")
def test_vision_model_not_loaded_for_empty_dir(MockBuilder):
    """Given a directory with no images, when captioning runs,
    then the VL model is never accessed."""
    data_dir = _make_test_dir(texts=["readme.txt"])
    model = MagicMock()
    model._vision_model = None
    captioner, _, _ = _make_captioner(data_dir, model=model)

    captioner.run()

    model.caption_image.assert_not_called()


# --- US-6: Broad Image Format Support ---

# AC-18: Common formats supported
@patch("image_captioner.LeannBuilder")
def test_finds_all_image_formats(MockBuilder):
    """Given images in various formats, when scanning for images,
    then all supported formats are found."""
    data_dir = _make_test_dir(images=[
        "a.jpg", "b.jpeg", "c.png", "d.gif", "e.webp", "f.bmp", "g.tiff",
    ])
    captioner, _, _ = _make_captioner(data_dir)
    images = captioner._find_images()
    assert len(images) == 7


# AC-18: Non-image files excluded
@patch("image_captioner.LeannBuilder")
def test_excludes_non_image_files(MockBuilder):
    """Given a mix of image and non-image files, then only images are found."""
    data_dir = _make_test_dir(
        images=["photo.jpg", "screenshot.png"],
        texts=["readme.txt", "notes.md"],
    )
    captioner, _, _ = _make_captioner(data_dir)
    images = captioner._find_images()
    names = {img.name for img in images}
    assert names == {"photo.jpg", "screenshot.png"}


# AC-20: Unsupported/corrupted format handling (covered by test_corrupted_image_skipped above)


# --- Step 03-01: LeannBuilder is_recompute=False ---

@patch("image_captioner.LeannBuilder")
def test_leann_builder_constructed_with_is_recompute_false(MockBuilder):
    """Given caption injection runs, when LeannBuilder is constructed,
    then is_recompute=False to avoid spawning a duplicate ZMQ embedding server."""
    data_dir = _make_test_dir(images=["photo.jpg"])
    captioner, _, _ = _make_captioner(data_dir)

    captioner.run()

    MockBuilder.assert_called_once_with(
        backend_name="hnsw",
        embedding_model="facebook/contriever",
        is_recompute=False,
    )


# --- Step 02-01: Image downscaling before captioning ---


@pytest.mark.parametrize("input_size,expected_max", [
    ((4000, 3000), (768, 576)),   # large landscape -> downscaled, aspect preserved
    ((3000, 4000), (576, 768)),   # large portrait -> downscaled, aspect preserved
    ((200, 150), (200, 150)),     # small image -> NOT upscaled
    ((768, 768), (768, 768)),     # exact boundary -> unchanged
    ((1000, 500), (768, 384)),    # wide image -> width-limited
])
def test_downscales_large_images_preserving_aspect_ratio(input_size, expected_max, tmp_path):
    """Given an image of various sizes, when loaded as data URI,
    then large images are downscaled to fit 768x768 preserving aspect ratio,
    and small images are not upscaled."""
    from PIL import Image
    from image_captioner import ImageCaptioner
    from caption_cache import CaptionCache
    import base64
    import io

    # Create a real image file
    img = Image.new("RGB", input_size, color=(255, 0, 0))
    img_path = tmp_path / "test.jpg"
    img.save(img_path, format="JPEG")

    # Create captioner with minimal deps (only need _load_image_as_data_uri)
    captioner = ImageCaptioner(
        model=MagicMock(),
        index_path="/fake",
        cache=MagicMock(),
        data_dir=str(tmp_path),
        send_fn=MagicMock(),
        dir_id="test",
    )

    data_uri = captioner._load_image_as_data_uri(img_path)

    # Decode the data URI back to check dimensions
    b64_data = data_uri.split(",")[1]
    result_img = Image.open(io.BytesIO(base64.b64decode(b64_data)))
    assert result_img.size == expected_max


# --- Step 02-02: Captioner sends "captioning" status event ---

@patch("image_captioner.LeannBuilder")
def test_run_sends_captioning_status_when_uncached_images_exist(MockBuilder):
    """Given a directory with uncached images, when run() executes,
    then it sends a status event with state='captioning' exactly once."""
    data_dir = _make_test_dir(images=["a.jpg", "b.png"])
    captioner, send_fn, _ = _make_captioner(data_dir)

    captioner.run()

    status_calls = [
        c for c in send_fn.call_args_list
        if c[0][1] == "status" and c[0][2].get("state") == "captioning"
    ]
    assert len(status_calls) == 1, (
        f"Expected exactly 1 captioning status event, got {len(status_calls)}"
    )


@patch("image_captioner.LeannBuilder")
def test_run_does_not_send_captioning_status_when_all_cached(MockBuilder):
    """Given all images are already cached, when run() executes,
    then no status event with state='captioning' is sent."""
    data_dir = _make_test_dir(images=["cached1.jpg", "cached2.png"])
    from caption_cache import CaptionCache
    cache = CaptionCache(os.path.join(data_dir, ".neurofind", "captions"))
    cache.put(os.path.join(data_dir, "cached1.jpg"), "Already captioned")
    cache.put(os.path.join(data_dir, "cached2.png"), "Already captioned")

    captioner, send_fn, _ = _make_captioner(data_dir, cache=cache)

    captioner.run()

    status_calls = [
        c for c in send_fn.call_args_list
        if c[0][1] == "status" and c[0][2].get("state") == "captioning"
    ]
    assert len(status_calls) == 0, (
        f"Expected no captioning status event, got {len(status_calls)}"
    )
