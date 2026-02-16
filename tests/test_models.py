"""Tests for ModelManager single-model interface."""
import pytest
from unittest.mock import MagicMock
from models import ModelManager


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
