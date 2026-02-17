"""Tests for the NDJSON server protocol layer."""
import json
import io
import sys
import pytest
from unittest.mock import MagicMock, patch


def make_request(method, params=None, req_id=1):
    return json.dumps({"id": req_id, "method": method, "params": params or {}})


class TestProtocol:
    """Test the NDJSON read/write protocol."""

    def test_parse_valid_request(self):
        from server import parse_request
        req = parse_request('{"id": 1, "method": "ping", "params": {}}')
        assert req["id"] == 1
        assert req["method"] == "ping"

    def test_parse_invalid_json(self):
        from server import parse_request
        req = parse_request("not json")
        assert req is None

    def test_parse_missing_method(self):
        from server import parse_request
        req = parse_request('{"id": 1}')
        assert req is None

    def test_format_result(self):
        from server import format_response
        line = format_response(1, "result", {"status": "ok"})
        parsed = json.loads(line)
        assert parsed == {"id": 1, "type": "result", "data": {"status": "ok"}}

    def test_format_token(self):
        from server import format_response
        line = format_response(2, "token", {"text": "hello"})
        parsed = json.loads(line)
        assert parsed["id"] == 2
        assert parsed["type"] == "token"
        assert parsed["data"]["text"] == "hello"

    def test_format_error(self):
        from server import format_response
        line = format_response(None, "error", {"message": "boom"})
        parsed = json.loads(line)
        assert parsed["id"] is None
        assert parsed["type"] == "error"


class TestPing:
    """Test the ping/health check method."""

    def test_ping_before_init(self):
        from server import Server
        srv = Server()
        result = srv.handle_ping(1)
        assert result["data"]["state"] == "not_initialized"

    def test_ping_after_ready(self):
        from server import Server
        srv = Server()
        srv.state = "ready"
        result = srv.handle_ping(1)
        assert result["data"]["state"] == "ready"
        assert "uptime" in result["data"]


class TestToggleDebug:
    """Test debug toggle."""

    def test_toggle_debug(self):
        from server import Server
        srv = Server()
        assert srv.debug is False
        result = srv.handle_toggle_debug(1)
        assert srv.debug is True
        assert result["data"]["debug"] is True

    def test_toggle_debug_propagates_to_directories(self):
        from server import Server
        srv = Server()
        mock_agent = MagicMock()
        mock_searcher = MagicMock()
        srv.directories["test"] = {
            "agent": mock_agent,
            "searcher": mock_searcher,
            "state": "ready",
            "dir_id": "test",
            "conversation_history": [],
        }
        srv.handle_toggle_debug(1)
        assert mock_agent.debug is True
        assert mock_searcher.debug is True


class TestQueryStreaming:
    """Test streaming tokens through the query handler."""

    def test_query_streams_tokens(self):
        from server import Server
        sent_messages = []

        import server as srv_mod
        original_send = srv_mod.send
        srv_mod.send = lambda rid, rtype, data: sent_messages.append(
            {"id": rid, "type": rtype, "data": data}
        )

        try:
            srv = Server()
            srv.state = "ready"

            mock_agent = MagicMock()
            mock_agent.run.return_value = "hello world"
            srv.directories["test"] = {
                "agent": mock_agent,
                "conversation_history": [],
                "state": "ready",
                "dir_id": "test",
            }

            result = srv.handle_query(1, {"text": "test query", "directoryId": "test"})
            assert result["type"] == "result"
            assert result["data"]["text"] == "hello world"

            # Verify agent.run was called with on_token callback
            call_kwargs = mock_agent.run.call_args
            assert "on_token" in call_kwargs.kwargs
            assert callable(call_kwargs.kwargs["on_token"])

            # Simulate what happens when on_token is called
            on_token_cb = call_kwargs.kwargs["on_token"]
            on_token_cb("hello")
            on_token_cb(" world")
            assert len(sent_messages) == 2
            assert sent_messages[0] == {"id": 1, "type": "token", "data": {"text": "hello"}}
            assert sent_messages[1] == {"id": 1, "type": "token", "data": {"text": " world"}}
        finally:
            srv_mod.send = original_send


class TestShutdown:
    """Test clean shutdown."""

    def test_shutdown_sets_running_false(self):
        from server import Server
        srv = Server()
        srv.running = True
        result = srv.handle_shutdown(1)
        assert srv.running is False


class TestMultiDirectoryInit:
    """Test multi-directory initialization."""

    def test_directories_dict_exists(self):
        from server import Server
        srv = Server()
        assert isinstance(srv.directories, dict)
        assert len(srv.directories) == 0

    def test_make_dir_id_simple(self):
        from server import make_dir_id
        assert make_dir_id("/home/user/documents") == "documents"

    def test_make_dir_id_with_spaces(self):
        from server import make_dir_id
        assert make_dir_id("/home/user/my documents") == "my_documents"

    def test_no_old_attributes(self):
        from server import Server
        srv = Server()
        assert not hasattr(srv, "agent")
        assert not hasattr(srv, "searcher")
        assert not hasattr(srv, "data_dir")
        assert not hasattr(srv, "index_name")
        assert not hasattr(srv, "conversation_history")


class TestQueryRouting:
    """Test directoryId routing in queries."""

    def test_query_unknown_directory_errors(self):
        from server import Server
        srv = Server()
        srv.state = "ready"
        result = srv.handle_query(1, {"text": "hello", "directoryId": "nonexistent"})
        assert result["type"] == "error"
        assert "Unknown directory" in result["data"]["message"]

    def test_query_not_ready_directory_errors(self):
        from server import Server
        srv = Server()
        srv.state = "ready"
        srv.directories["test"] = {
            "dir_id": "test",
            "state": "indexing",
            "agent": MagicMock(),
            "conversation_history": [],
        }
        result = srv.handle_query(1, {"text": "hello", "directoryId": "test"})
        assert result["type"] == "error"
        assert "not ready" in result["data"]["message"]

    def test_query_falls_back_to_first_ready(self):
        from server import Server
        import server as srv_mod
        original_send = srv_mod.send
        srv_mod.send = lambda rid, rtype, data: None

        try:
            srv = Server()
            srv.state = "ready"
            mock_agent = MagicMock()
            mock_agent.run.return_value = "response"
            srv.directories["dir1"] = {
                "dir_id": "dir1",
                "state": "ready",
                "agent": mock_agent,
                "conversation_history": [],
            }
            result = srv.handle_query(1, {"text": "hello"})
            assert result["type"] == "result"
            assert result["data"]["text"] == "response"
        finally:
            srv_mod.send = original_send

    def test_query_no_ready_directories(self):
        from server import Server
        srv = Server()
        srv.state = "ready"
        result = srv.handle_query(1, {"text": "hello"})
        assert result["type"] == "error"
        assert "No ready directories" in result["data"]["message"]


class TestRemoveDirectory:
    """Test add/remove directory lifecycle."""

    def test_remove_existing_directory(self):
        from server import Server
        srv = Server()
        srv.state = "ready"
        srv.directories["test"] = {
            "dir_id": "test",
            "state": "ready",
            "path": "/tmp/test",
            "conversation_history": [],
        }
        result = srv.handle_remove_directory(1, {"directoryId": "test"})
        assert result["type"] == "result"
        assert result["data"]["status"] == "ok"
        assert "test" not in srv.directories

    def test_remove_unknown_directory_errors(self):
        from server import Server
        srv = Server()
        result = srv.handle_remove_directory(1, {"directoryId": "nope"})
        assert result["type"] == "error"

    def test_remove_last_resets_state(self):
        from server import Server
        srv = Server()
        srv.state = "ready"
        srv.directories["test"] = {
            "dir_id": "test",
            "state": "ready",
            "path": "/tmp/test",
            "conversation_history": [],
        }
        srv.handle_remove_directory(1, {"directoryId": "test"})
        assert srv.state == "not_initialized"


class TestDirectorySummary:
    """Test summary generation."""

    def test_generate_summary_with_index(self):
        from server import Server
        srv = Server()
        mock_model = MagicMock()
        mock_model.generate.return_value = "  A project proposal with architecture docs.  "
        srv.model = mock_model

        mock_searcher = MagicMock()
        mock_searcher.search_and_extract.return_value = "Found: proposal.md, architecture decisions, PRD documents"

        srv.directories["test"] = {
            "dir_id": "test",
            "searcher": mock_searcher,
            "state": "ready",
        }

        summary = srv._generate_summary("test")
        assert summary == "A project proposal with architecture docs."
        mock_searcher.search_and_extract.assert_called_once()
        mock_model.generate.assert_called_once()
        prompt_messages = mock_model.generate.call_args[0][0]
        assert "Found: proposal.md" in prompt_messages[0]["content"]

    def test_generate_summary_without_model(self):
        from server import Server
        srv = Server()
        srv.model = None
        srv.directories["test"] = {"dir_id": "test", "searcher": MagicMock(), "state": "ready"}
        summary = srv._generate_summary("test")
        assert summary == ""

    def test_generate_summary_missing_directory(self):
        from server import Server
        srv = Server()
        srv.model = MagicMock()
        summary = srv._generate_summary("nonexistent")
        assert summary == ""


class TestCollectStats:
    """Test directory stat collection."""

    def test_collect_stats(self, tmp_path):
        from server import Server
        srv = Server()
        # Create test files
        (tmp_path / "doc.pdf").write_bytes(b"fake pdf content")
        (tmp_path / "notes.txt").write_text("hello")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "data.csv").write_text("a,b,c")

        stats = srv._collect_stats(tmp_path)
        assert stats["fileCount"] == 3
        assert stats["types"]["pdf"] == 1
        assert stats["types"]["txt"] == 1
        assert stats["types"]["csv"] == 1
        assert stats["totalSize"] > 0

    def test_collect_stats_enhanced(self, tmp_path):
        from server import Server
        srv = Server()
        (tmp_path / "big.pdf").write_bytes(b"x" * 10000)
        (tmp_path / "small.pdf").write_bytes(b"x" * 500)
        (tmp_path / "notes.md").write_text("hello world")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "deep").mkdir()
        (tmp_path / "sub" / "deep" / "data.csv").write_text("a,b,c")
        (tmp_path / "sub" / "readme.md").write_text("readme")

        stats = srv._collect_stats(tmp_path)
        assert stats["fileCount"] == 5
        assert stats["types"]["pdf"] == 2
        assert stats["types"]["md"] == 2
        assert stats["types"]["csv"] == 1
        assert "sizeByType" in stats
        assert stats["sizeByType"]["pdf"] == 10500
        assert stats["sizeByType"]["md"] > 0
        assert "largestFiles" in stats
        assert len(stats["largestFiles"]) <= 3
        assert stats["largestFiles"][0]["name"] == "big.pdf"
        assert stats["largestFiles"][0]["size"] == 10000
        assert "avgFileSize" in stats
        assert stats["avgFileSize"] == stats["totalSize"] // stats["fileCount"]
        assert "dirs" in stats
        assert stats["dirs"]["count"] == 2
        assert stats["dirs"]["maxDepth"] == 2


class TestMultiDirectoryFlow:
    """Test that dispatch routes all new methods."""

    def test_dispatch_remove_directory(self):
        from server import Server
        srv = Server()
        srv.directories["test"] = {
            "dir_id": "test",
            "state": "ready",
            "path": "/tmp/test",
            "conversation_history": [],
        }
        result = srv.dispatch({"id": 1, "method": "remove_directory", "params": {"directoryId": "test"}})
        assert result["type"] == "result"
        assert result["data"]["status"] == "ok"

    def test_dispatch_reindex_unknown(self):
        from server import Server
        srv = Server()
        result = srv.dispatch({"id": 1, "method": "reindex", "params": {"directoryId": "nope"}})
        assert result["type"] == "error"

    def test_dispatch_unknown_method(self):
        from server import Server
        srv = Server()
        result = srv.dispatch({"id": 1, "method": "nonexistent", "params": {}})
        assert result["type"] == "error"
        assert "Unknown method" in result["data"]["message"]


class TestGetFileGraph:
    """Test file graph handler."""

    def test_get_file_graph_unknown_directory(self):
        from server import Server
        srv = Server()
        srv.state = "ready"
        result = srv.handle_get_file_graph(1, {"directoryId": "nonexistent"})
        assert result["type"] == "error"
        assert "Unknown directory" in result["data"]["message"]

    def test_get_file_graph_not_ready(self):
        from server import Server
        srv = Server()
        srv.state = "ready"
        srv.directories["test"] = {
            "dir_id": "test",
            "state": "indexing",
            "path": "/tmp/test",
        }
        result = srv.handle_get_file_graph(1, {"directoryId": "test"})
        assert result["type"] == "error"
        assert "not ready" in result["data"]["message"]

    def test_get_file_graph_returns_graph_structure(self):
        from server import Server
        srv = Server()
        srv.state = "ready"
        mock_searcher = MagicMock()
        srv.directories["test"] = {
            "dir_id": "test",
            "state": "ready",
            "path": "/tmp/test",
            "searcher": mock_searcher,
        }
        with patch("server.build_file_graph") as mock_build:
            mock_build.return_value = {
                "nodes": [{"id": "a.pdf", "name": "a.pdf", "type": "pdf", "size": 100, "dir": "", "passageCount": 3}],
                "edges": [{"source": "a.pdf", "target": "b.pdf", "type": "similarity", "weight": 0.8}],
            }
            result = srv.handle_get_file_graph(1, {"directoryId": "test"})
            assert result["type"] == "result"
            assert "nodes" in result["data"]
            assert "edges" in result["data"]
            assert len(result["data"]["nodes"]) == 1
            mock_build.assert_called_once()

    def test_get_file_graph_uses_cache(self):
        from server import Server
        srv = Server()
        srv.state = "ready"
        cached_graph = {"nodes": [], "edges": []}
        srv.directories["test"] = {
            "dir_id": "test",
            "state": "ready",
            "path": "/tmp/test",
            "searcher": MagicMock(),
            "file_graph": cached_graph,
        }
        with patch("server.build_file_graph") as mock_build:
            result = srv.handle_get_file_graph(1, {"directoryId": "test"})
            assert result["type"] == "result"
            mock_build.assert_not_called()

    def test_dispatch_routes_get_file_graph(self):
        from server import Server
        srv = Server()
        srv.state = "ready"
        srv.directories["test"] = {
            "dir_id": "test",
            "state": "ready",
            "path": "/tmp/test",
            "searcher": MagicMock(),
        }
        with patch("server.build_file_graph") as mock_build:
            mock_build.return_value = {"nodes": [], "edges": []}
            result = srv.dispatch({"id": 1, "method": "getFileGraph", "params": {"directoryId": "test"}})
            assert result["type"] == "result"
