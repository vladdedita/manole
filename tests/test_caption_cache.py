"""Acceptance tests for CaptionCache (US-2: Caption Persistence).

Milestone 1: CaptionCache — pure Python, no external deps.
All tests here should pass before moving to Milestone 2.
"""
import os
import time
import tempfile
import pytest


# --- AC-6: Captions persist across sessions ---

def test_put_then_get_returns_caption():
    """Given an image was captioned, when we get it, then the caption is returned."""
    from caption_cache import CaptionCache
    with tempfile.TemporaryDirectory() as cache_dir:
        # Create a fake image file so we have a real path with mtime
        img = os.path.join(cache_dir, "photo.jpg")
        with open(img, "wb") as f:
            f.write(b"\xff\xd8fake-jpeg")

        cache = CaptionCache(os.path.join(cache_dir, "captions"))
        cache.put(img, "A tabby cat on a desk")
        assert cache.get(img) == "A tabby cat on a desk"


def test_cache_persists_across_instances():
    """Given captions were stored in session 1, when a new CaptionCache is created
    for session 2, then cached captions are still available."""
    from caption_cache import CaptionCache
    with tempfile.TemporaryDirectory() as cache_dir:
        img = os.path.join(cache_dir, "photo.jpg")
        with open(img, "wb") as f:
            f.write(b"\xff\xd8fake-jpeg")

        caption_dir = os.path.join(cache_dir, "captions")

        # Session 1: store caption
        cache1 = CaptionCache(caption_dir)
        cache1.put(img, "Ocean sunset with orange sky")

        # Session 2: new instance reads same cache dir
        cache2 = CaptionCache(caption_dir)
        assert cache2.get(img) == "Ocean sunset with orange sky"


def test_get_returns_none_for_uncached():
    """Given an image was never captioned, when we get it, then None is returned."""
    from caption_cache import CaptionCache
    with tempfile.TemporaryDirectory() as cache_dir:
        img = os.path.join(cache_dir, "photo.jpg")
        with open(img, "wb") as f:
            f.write(b"\xff\xd8fake-jpeg")

        cache = CaptionCache(os.path.join(cache_dir, "captions"))
        assert cache.get(img) is None


# --- AC-7: Cache invalidation on file change ---

def test_cache_invalidated_when_file_modified():
    """Given image was captioned with mtime T1, when the file is modified (mtime T2),
    then cache returns None."""
    from caption_cache import CaptionCache
    with tempfile.TemporaryDirectory() as cache_dir:
        img = os.path.join(cache_dir, "photo.jpg")
        with open(img, "wb") as f:
            f.write(b"\xff\xd8fake-jpeg-v1")

        cache = CaptionCache(os.path.join(cache_dir, "captions"))
        cache.put(img, "Old caption")
        assert cache.get(img) == "Old caption"

        # Modify the file (change mtime)
        time.sleep(0.1)  # ensure mtime changes
        with open(img, "wb") as f:
            f.write(b"\xff\xd8fake-jpeg-v2")

        assert cache.get(img) is None


# --- AC-8: Cache key uniqueness ---

def test_different_paths_produce_different_keys():
    """Given two different images at different paths, when both are captioned,
    then each has a unique cache entry and captions don't collide."""
    from caption_cache import CaptionCache
    with tempfile.TemporaryDirectory() as cache_dir:
        img_a = os.path.join(cache_dir, "a.jpg")
        img_b = os.path.join(cache_dir, "b.jpg")
        with open(img_a, "wb") as f:
            f.write(b"\xff\xd8fake-a")
        with open(img_b, "wb") as f:
            f.write(b"\xff\xd8fake-b")

        cache = CaptionCache(os.path.join(cache_dir, "captions"))
        cache.put(img_a, "Caption A")
        cache.put(img_b, "Caption B")
        assert cache.get(img_a) == "Caption A"
        assert cache.get(img_b) == "Caption B"


def test_cache_creates_directory_if_missing():
    """Given the cache directory doesn't exist, when CaptionCache is initialized,
    then it creates the directory."""
    from caption_cache import CaptionCache
    with tempfile.TemporaryDirectory() as base:
        cache_dir = os.path.join(base, "nested", "captions")
        cache = CaptionCache(cache_dir)
        img = os.path.join(base, "photo.jpg")
        with open(img, "wb") as f:
            f.write(b"\xff\xd8fake")
        cache.put(img, "test")
        assert cache.get(img) == "test"
