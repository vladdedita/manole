"""Tests for ModelManager dual-model loading."""
from unittest.mock import patch, MagicMock
from models import ModelManager


def test_model_manager_has_plan_method():
    mgr = ModelManager.__new__(ModelManager)
    mgr.planner_model = None
    mgr.rag_model = None
    assert hasattr(mgr, "plan")
    assert hasattr(mgr, "extract")
    assert hasattr(mgr, "synthesize")


def test_plan_calls_planner_model():
    mgr = ModelManager.__new__(ModelManager)
    mock_model = MagicMock()
    mock_model.return_value = {"choices": [{"text": '{"keywords": ["test"]}'}]}
    mgr.planner_model = mock_model
    mgr.rag_model = None

    result = mgr.plan("test prompt")
    mock_model.assert_called_once()
    assert '{"keywords": ["test"]}' in result


def test_extract_calls_rag_model():
    mgr = ModelManager.__new__(ModelManager)
    mgr.planner_model = None
    mock_model = MagicMock()
    mock_model.return_value = {"choices": [{"text": '{"relevant": true}'}]}
    mgr.rag_model = mock_model

    result = mgr.extract("test prompt")
    mock_model.assert_called_once()


def test_synthesize_calls_rag_model():
    mgr = ModelManager.__new__(ModelManager)
    mgr.planner_model = None
    mock_model = MagicMock()
    mock_model.return_value = {"choices": [{"text": "The answer is 42."}]}
    mgr.rag_model = mock_model

    result = mgr.synthesize("test prompt")
    mock_model.assert_called_once()


def test_plan_max_tokens_is_256():
    """Planner should use small max_tokens for JSON output."""
    mgr = ModelManager.__new__(ModelManager)
    mock_model = MagicMock()
    mock_model.return_value = {"choices": [{"text": "{}"}]}
    mgr.planner_model = mock_model
    mgr.rag_model = None

    mgr.plan("test")
    call_kwargs = mock_model.call_args
    assert call_kwargs[1].get("max_tokens") == 256 or call_kwargs.kwargs.get("max_tokens") == 256
