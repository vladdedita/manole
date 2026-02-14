"""Tests for ModelManager single-model interface."""
from unittest.mock import MagicMock
from models import ModelManager


def _mock_chat_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


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
