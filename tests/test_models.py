"""Tests for ModelManager single-model interface."""
import json
import os
import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch
from models import ModelManager, get_models_dir, load_manifest


def _mock_chat_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def _mock_stream_chunks(text: str) -> list[dict]:
    """Build a list of streaming chunk dicts, one per character."""
    return [{"choices": [{"delta": {"content": ch}}]} for ch in text]


@pytest.fixture
def mock_model():
    """ModelManager with a mock Llama supporting both regular and streaming calls."""
    mgr = ModelManager.__new__(ModelManager)
    model = MagicMock()
    model.create_chat_completion.side_effect = lambda **kwargs: (
        iter(_mock_stream_chunks("hello")) if kwargs.get("stream") else _mock_chat_response("hello")
    )
    mgr.model = model
    return mgr


def test_generate_calls_model():
    mgr = ModelManager.__new__(ModelManager)
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_chat_response("hello")
    mgr.model = mock_model
    result = mgr.generate([{"role": "user", "content": "hi"}])
    mock_model.create_chat_completion.assert_called_once()
    assert result == "hello"


def test_generate_passes_messages():
    mgr = ModelManager.__new__(ModelManager)
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_chat_response("ok")
    mgr.model = mock_model
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "test"},
    ]
    mgr.generate(messages)
    call_kwargs = mock_model.create_chat_completion.call_args
    passed_messages = call_kwargs.kwargs.get("messages") or call_kwargs[0][0]
    assert len(passed_messages) == 2
    assert passed_messages[0]["role"] == "system"


def test_generate_max_tokens():
    mgr = ModelManager.__new__(ModelManager)
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_chat_response("ok")
    mgr.model = mock_model
    mgr.generate([{"role": "user", "content": "hi"}], max_tokens=256)
    call_kwargs = mock_model.create_chat_completion.call_args
    assert call_kwargs.kwargs.get("max_tokens") == 256


def test_generate_resets_model():
    mgr = ModelManager.__new__(ModelManager)
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_chat_response("ok")
    mgr.model = mock_model
    mgr.generate([{"role": "user", "content": "hi"}])
    mock_model.reset.assert_called_once()


def test_default_model_path():
    mgr = ModelManager()
    assert "LFM2.5-1.2B-Instruct" in mgr.model_path


def test_custom_model_path():
    mgr = ModelManager(model_path="/custom/path.gguf")
    assert mgr.model_path == "/custom/path.gguf"


# --- Platform-aware path resolution ---


class TestGetModelsDir:
    """Test get_models_dir() returns correct path per platform and env."""

    def test_dev_mode_uses_local_models_dir(self):
        """When not frozen (dev mode), uses ./models/ relative to project."""
        with patch.object(sys, "frozen", False, create=True):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MANOLE_MODELS_DIR", None)
                result = get_models_dir()
        assert result == Path("models")

    def test_env_var_overrides_platform_default(self):
        """MANOLE_MODELS_DIR env var overrides any platform default."""
        with patch.dict(os.environ, {"MANOLE_MODELS_DIR": "/custom/models"}):
            with patch.object(sys, "frozen", True, create=True):
                result = get_models_dir()
        assert result == Path("/custom/models")

    @pytest.mark.parametrize("platform,expected_suffix", [
        ("darwin", "Library/Application Support/Manole/models"),
        ("linux", ".local/share/manole/models"),
    ])
    def test_packaged_mode_platform_paths(self, platform, expected_suffix):
        """Frozen/packaged mode resolves to platform-specific user data dir."""
        with patch.object(sys, "frozen", True, create=True):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MANOLE_MODELS_DIR", None)
                with patch("models.sys.platform", platform):
                    result = get_models_dir()
        assert str(result).endswith(expected_suffix)


class TestLoadManifest:
    """Test manifest loading from models-manifest.json."""

    def test_loads_manifest_with_expected_models(self):
        """Manifest contains text-model, vision-model, vision-projector."""
        manifest = load_manifest()
        model_ids = [m["id"] for m in manifest["models"]]
        assert "text-model" in model_ids
        assert "vision-model" in model_ids
        assert "vision-projector" in model_ids

    def test_manifest_models_have_required_fields(self):
        """Each model entry has id, filename, repo_id."""
        manifest = load_manifest()
        for model in manifest["models"]:
            assert "id" in model
            assert "filename" in model
            assert "repo_id" in model


class TestModelManagerManifestIntegration:
    """ModelManager resolves paths from manifest, not hardcoded strings."""

    def test_model_paths_use_manifest_filenames(self):
        """ModelManager default paths contain filenames from manifest."""
        manifest = load_manifest()
        filenames = {m["id"]: m["filename"] for m in manifest["models"]}
        mgr = ModelManager()
        assert Path(mgr.model_path).name == filenames["text-model"]
        assert Path(mgr.vision_model_path).name == filenames["vision-model"]
        assert Path(mgr.mmproj_path).name == filenames["vision-projector"]

    def test_model_paths_use_models_dir(self):
        """ModelManager paths are rooted in get_models_dir()."""
        mgr = ModelManager()
        models_dir = get_models_dir()
        assert Path(mgr.model_path).parent == models_dir
        assert Path(mgr.vision_model_path).parent == models_dir
        assert Path(mgr.mmproj_path).parent == models_dir


def test_generate_stream_calls_on_token(mock_model):
    """Streaming generate calls on_token for each chunk and returns full text."""
    tokens = []
    result = mock_model.generate(
        [{"role": "user", "content": "hi"}],
        stream=True,
        on_token=lambda t: tokens.append(t),
    )
    assert isinstance(result, str)
    assert len(result) > 0
    assert len(tokens) > 0
    assert "".join(tokens) == result


def test_generate_stream_without_callback(mock_model):
    """Streaming without on_token still returns full text."""
    result = mock_model.generate(
        [{"role": "user", "content": "hi"}],
        stream=True,
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_non_stream_unchanged(mock_model):
    """Default non-streaming behavior is unchanged."""
    result = mock_model.generate(
        [{"role": "user", "content": "hi"}],
    )
    assert isinstance(result, str)
    assert len(result) > 0
