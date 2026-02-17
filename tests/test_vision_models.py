"""Acceptance tests for ModelManager vision extensions (US-5).

Milestone 3: Vision model lifecycle in ModelManager.
"""
import pytest
from unittest.mock import MagicMock, patch


# --- AC-15: Lazy loading ---

def test_vision_model_not_loaded_on_init():
    """Given a ModelManager, when initialized, then the VL model is not loaded."""
    from models import ModelManager
    mgr = ModelManager()
    assert mgr._vision_model is None


def test_vision_model_lazy_loads_on_first_access():
    """Given a ModelManager, when vision_model property is accessed,
    then the VL model loads with MoondreamChatHandler."""
    from models import ModelManager
    mgr = ModelManager.__new__(ModelManager)
    mgr.n_threads = 4
    mgr.vision_model_path = "/fake/vl-model.gguf"
    mgr.mmproj_path = "/fake/mmproj.gguf"
    mgr._vision_model = None

    mock_llama = MagicMock()
    mock_handler = MagicMock()
    with patch("llama_cpp.Llama", return_value=mock_llama) as MockLlama, \
         patch("llama_cpp.llama_chat_format.MoondreamChatHandler", return_value=mock_handler), \
         patch.object(ModelManager, "_ensure_model", return_value="/fake/path"):
        result = mgr.vision_model
        MockLlama.assert_called_once()
        # Verify chat_handler was passed to Llama
        call_kwargs = MockLlama.call_args.kwargs
        assert call_kwargs["chat_handler"] is mock_handler
        assert result is mock_llama


def test_vision_model_loaded_only_once():
    """Given vision_model is accessed multiple times, the model loads only once."""
    from models import ModelManager
    mgr = ModelManager.__new__(ModelManager)
    mgr.n_threads = 4
    mgr.vision_model_path = "/fake/vl-model.gguf"
    mgr.mmproj_path = "/fake/mmproj.gguf"
    mgr._vision_model = None

    mock_llama = MagicMock()
    with patch("llama_cpp.Llama", return_value=mock_llama) as MockLlama, \
         patch("llama_cpp.llama_chat_format.MoondreamChatHandler", return_value=MagicMock()), \
         patch.object(ModelManager, "_ensure_model", return_value="/fake/path"):
        _ = mgr.vision_model
        _ = mgr.vision_model
        _ = mgr.vision_model
        assert MockLlama.call_count == 1


# --- AC-16: Text-only directories (VL model never loaded) ---

def test_default_vision_model_path():
    """Given default init, vision_model_path points to the VL GGUF."""
    from models import ModelManager
    mgr = ModelManager()
    assert "moondream2" in mgr.vision_model_path


def test_custom_vision_model_path():
    """Given a custom vision model path, it's stored correctly."""
    from models import ModelManager
    mgr = ModelManager(vision_model_path="/custom/vl.gguf")
    assert mgr.vision_model_path == "/custom/vl.gguf"


def test_default_mmproj_path():
    """Given default init, mmproj_path points to the mmproj GGUF."""
    from models import ModelManager
    mgr = ModelManager()
    assert "mmproj" in mgr.mmproj_path


def test_custom_mmproj_path():
    """Given a custom mmproj path, it's stored correctly."""
    from models import ModelManager
    mgr = ModelManager(mmproj_path="/custom/mmproj.gguf")
    assert mgr.mmproj_path == "/custom/mmproj.gguf"


# --- caption_image method ---

def test_caption_image_calls_vision_model():
    """Given a base64 image URI, caption_image calls the VL model and returns the caption."""
    from models import ModelManager
    mgr = ModelManager.__new__(ModelManager)
    mock_vl = MagicMock()
    mock_vl.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "A tabby cat on a desk"}}]
    }
    mgr._vision_model = mock_vl
    mgr.n_threads = 4
    mgr.vision_model_path = "/fake/vl.gguf"
    mgr.mmproj_path = "/fake/mmproj.gguf"

    result = mgr.caption_image("data:image/jpeg;base64,/9j/4AAQ...")
    assert result == "A tabby cat on a desk"
    mock_vl.create_chat_completion.assert_called_once()


def test_caption_image_sends_multimodal_message():
    """Given caption_image is called, the message sent to VL model includes
    both image_url and text content types."""
    from models import ModelManager
    mgr = ModelManager.__new__(ModelManager)
    mock_vl = MagicMock()
    mock_vl.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "A sunset"}}]
    }
    mgr._vision_model = mock_vl
    mgr.n_threads = 4
    mgr.vision_model_path = "/fake/vl.gguf"
    mgr.mmproj_path = "/fake/mmproj.gguf"

    mgr.caption_image("data:image/jpeg;base64,abc123")

    call_kwargs = mock_vl.create_chat_completion.call_args.kwargs
    messages = call_kwargs["messages"]
    assert len(messages) == 1
    content = messages[0]["content"]
    content_types = {item["type"] for item in content}
    assert "image_url" in content_types
    assert "text" in content_types


# --- _ensure_model auto-download ---

def test_ensure_model_returns_path_when_file_exists(tmp_path):
    """Given a model file that exists, _ensure_model returns its path without downloading."""
    from models import ModelManager
    fake_model = tmp_path / "model.gguf"
    fake_model.write_text("fake")

    result = ModelManager._ensure_model(str(fake_model), "some/repo", "model.gguf")
    assert result == str(fake_model)


def test_ensure_model_downloads_when_missing(tmp_path):
    """Given a model file that doesn't exist, _ensure_model calls hf_hub_download."""
    from models import ModelManager
    missing = tmp_path / "missing.gguf"

    with patch("huggingface_hub.hf_hub_download") as mock_dl:
        result = ModelManager._ensure_model(str(missing), "LiquidAI/SomeRepo", "missing.gguf")
        mock_dl.assert_called_once_with(
            repo_id="LiquidAI/SomeRepo",
            filename="missing.gguf",
            local_dir=str(tmp_path),
        )
        assert result == str(missing)


def test_load_calls_ensure_model():
    """Given load() is called, _ensure_model is invoked for the text model."""
    from models import ModelManager
    mgr = ModelManager()

    mock_llama = MagicMock()
    with patch("llama_cpp.Llama", return_value=mock_llama), \
         patch.object(ModelManager, "_ensure_model", return_value=mgr.model_path) as mock_ensure:
        mgr.load()
        mock_ensure.assert_called_once_with(
            mgr.model_path,
            ModelManager.TEXT_REPO_ID,
            "LFM2.5-1.2B-Instruct-Q4_0.gguf",
        )


def test_vision_model_calls_ensure_model_for_both_files():
    """Given vision_model is accessed, _ensure_model is called for both VL model and mmproj."""
    from models import ModelManager
    mgr = ModelManager()

    mock_llama = MagicMock()
    with patch("llama_cpp.Llama", return_value=mock_llama), \
         patch("llama_cpp.llama_chat_format.MoondreamChatHandler", return_value=MagicMock()), \
         patch.object(ModelManager, "_ensure_model", return_value="/fake/path") as mock_ensure:
        _ = mgr.vision_model
        assert mock_ensure.call_count == 2


# --- AC: Fail-fast on incompatible mmproj ---

def test_vision_model_raises_runtime_error_on_handler_failure():
    """Given MoondreamChatHandler raises ValueError on incompatible mmproj,
    when vision_model is accessed, then RuntimeError is raised with actionable message."""
    from models import ModelManager
    mgr = ModelManager.__new__(ModelManager)
    mgr.n_threads = 4
    mgr.vision_model_path = "/fake/vl-model.gguf"
    mgr.mmproj_path = "/fake/mmproj.gguf"
    mgr._vision_model = None

    with patch("llama_cpp.llama_chat_format.MoondreamChatHandler",
               side_effect=ValueError("Invalid clip model")), \
         patch.object(ModelManager, "_ensure_model", return_value="/fake/path"):
        with pytest.raises(RuntimeError, match="incompatible with MoondreamChatHandler"):
            _ = mgr.vision_model
