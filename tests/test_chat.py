"""Tests for chat.py --pipeline flag integration.

Step 02-01: kreuzberg-integration
Test Budget: 3 behaviors x 2 = 6 max unit tests

Behaviors:
1. pipeline=kreuzberg delegates to KreuzbergIndexer.build()
2. pipeline=leann uses existing subprocess behavior (unchanged)
3. main() parses --pipeline flag from sys.argv
"""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# --- Acceptance test: pipeline flag routes correctly ---


@patch("chat.subprocess")
@patch("indexer.KreuzbergIndexer")
def test_build_index_with_kreuzberg_pipeline_delegates_to_kreuzberg_indexer(
    MockIndexerClass, mock_subprocess
):
    """Given pipeline='kreuzberg',
    when build_index is called,
    then it creates a KreuzbergIndexer and calls build(), not subprocess."""
    from chat import build_index

    mock_indexer = MockIndexerClass.return_value
    data_dir = Path("/tmp/test_data")

    index_name = build_index(data_dir, force=True, pipeline="kreuzberg")

    mock_indexer.build.assert_called_once_with(data_dir, index_name, force=True)
    mock_subprocess.run.assert_not_called()


# --- Unit tests ---


@patch("chat.subprocess")
def test_build_index_with_leann_pipeline_uses_subprocess(mock_subprocess):
    """Given pipeline='leann' (default),
    when build_index is called,
    then it runs leann CLI via subprocess, same as before."""
    from chat import build_index

    mock_subprocess.run.return_value = MagicMock(returncode=0)
    data_dir = Path("/tmp/test_data")

    index_name = build_index(data_dir, force=False, pipeline="leann")

    mock_subprocess.run.assert_called_once()
    cmd = mock_subprocess.run.call_args[0][0]
    assert "build" in cmd
    assert index_name == "test_data"


@patch("chat.subprocess")
def test_build_index_default_pipeline_is_leann(mock_subprocess):
    """Given no pipeline argument,
    when build_index is called,
    then it defaults to leann subprocess behavior."""
    from chat import build_index

    mock_subprocess.run.return_value = MagicMock(returncode=0)
    data_dir = Path("/tmp/test_data")

    build_index(data_dir)

    mock_subprocess.run.assert_called_once()


@patch("chat.build_index")
@patch("chat.chat_loop")
def test_main_parses_pipeline_flag(mock_chat_loop, mock_build_index):
    """Given --pipeline kreuzberg in sys.argv,
    when main() is called,
    then build_index receives pipeline='kreuzberg'."""
    from chat import main

    mock_build_index.return_value = "test_data"

    with patch("sys.argv", ["chat.py", "/tmp/test_data", "--pipeline", "kreuzberg"]), \
         patch("pathlib.Path.is_dir", return_value=True), \
         patch("pathlib.Path.resolve", return_value=Path("/tmp/test_data")):
        main()

    mock_build_index.assert_called_once()
    call_kwargs = mock_build_index.call_args
    # pipeline should be passed as keyword argument
    assert call_kwargs.kwargs.get("pipeline") == "kreuzberg" or \
           (len(call_kwargs.args) >= 3 and call_kwargs.args[2] == "kreuzberg") or \
           call_kwargs[1].get("pipeline") == "kreuzberg"


@patch("indexer.KreuzbergIndexer")
def test_build_index_kreuzberg_returns_compatible_index_name(MockIndexerClass):
    """Given pipeline='kreuzberg',
    when build_index completes,
    then returned index_name matches get_index_name() output (compatible with find_index_path)."""
    from chat import build_index, get_index_name

    mock_indexer = MockIndexerClass.return_value
    data_dir = Path("/tmp/my-documents")

    index_name = build_index(data_dir, pipeline="kreuzberg")

    expected_name = get_index_name(data_dir)
    assert index_name == expected_name
