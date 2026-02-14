"""Tests for ModelManager dual-model loading."""
from unittest.mock import MagicMock
from models import ModelManager, _messages


def _mock_chat_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def test_model_manager_has_methods():
    mgr = ModelManager.__new__(ModelManager)
    mgr.extract_model = None
    mgr.instruct_model = None
    assert hasattr(mgr, "plan")
    assert hasattr(mgr, "map_chunk")
    assert hasattr(mgr, "extract")
    assert hasattr(mgr, "synthesize")


def test_messages_format():
    msgs = _messages("sys prompt", "user msg")
    assert msgs[0] == {"role": "system", "content": "sys prompt"}
    assert msgs[1] == {"role": "user", "content": "user msg"}


def test_plan_calls_instruct_model():
    mgr = ModelManager.__new__(ModelManager)
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_chat_response('{"keywords": ["test"]}')
    mgr.instruct_model = mock_model
    mgr.extract_model = MagicMock()

    result = mgr.plan("system", "user")
    mock_model.create_chat_completion.assert_called_once()
    assert '{"keywords": ["test"]}' in result


def test_map_chunk_calls_instruct_model():
    mgr = ModelManager.__new__(ModelManager)
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_chat_response('{"relevant": true}')
    mgr.instruct_model = mock_model
    mgr.extract_model = MagicMock()

    result = mgr.map_chunk("system", "user")
    mock_model.create_chat_completion.assert_called_once()


def test_extract_calls_extract_model():
    mgr = ModelManager.__new__(ModelManager)
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_chat_response('{"field": "value"}')
    mgr.extract_model = mock_model
    mgr.instruct_model = MagicMock()

    result = mgr.extract("system", "user")
    mock_model.create_chat_completion.assert_called_once()


def test_synthesize_calls_instruct_model():
    mgr = ModelManager.__new__(ModelManager)
    mgr.extract_model = MagicMock()
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_chat_response("The answer is 42.")
    mgr.instruct_model = mock_model

    result = mgr.synthesize("system", "user")
    mock_model.create_chat_completion.assert_called_once()


def test_messages_passed_correctly():
    mgr = ModelManager.__new__(ModelManager)
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_chat_response("{}")
    mgr.instruct_model = mock_model
    mgr.extract_model = MagicMock()

    mgr.plan("my system", "my user")
    call_kwargs = mock_model.create_chat_completion.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "my system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "my user"


def test_rewrite_calls_instruct_model():
    mgr = ModelManager.__new__(ModelManager)
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_chat_response('{"intent": "factual"}')
    mgr.instruct_model = mock_model
    mgr.extract_model = MagicMock()

    result = mgr.rewrite("system", "user")
    mock_model.create_chat_completion.assert_called_once()
    assert "factual" in result
