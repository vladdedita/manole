"""Tests for the NDJSON server protocol layer."""
import json
import io
import sys
from pathlib import Path
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
        initial = srv.debug
        result = srv.handle_toggle_debug(1)
        assert srv.debug is (not initial)
        assert result["data"]["debug"] is (not initial)

    def test_toggle_debug_propagates_to_directories(self):
        from server import Server
        srv = Server()
        srv.debug = False  # start from known state
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
            mock_agent.run.return_value = ("hello world", [])
            srv.directories["test"] = {
                "agent": mock_agent,
                "conversation_history": [],
                "state": "ready",
                "dir_id": "test",
                "path": "/tmp/test",
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
            mock_agent.run.return_value = ("response", [])
            srv.directories["dir1"] = {
                "dir_id": "dir1",
                "state": "ready",
                "agent": mock_agent,
                "conversation_history": [],
                "path": "/tmp/dir1",
            }
            result = srv.handle_query(1, {"text": "hello"})
            assert result["type"] == "result"
            assert result["data"]["text"] == "response"
            assert result["data"]["sources"] == []
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


class TestCheckModels:
    """Acceptance: check_models returns per-model present/missing status."""

    def test_check_models_returns_status_for_each_manifest_model(self, tmp_path):
        """Given a models directory with one model present and others missing,
        check_models returns ready=false with per-model exists status."""
        from server import Server

        srv = Server()
        manifest = {
            "version": 1,
            "models": [
                {"id": "text-model", "filename": "text.gguf", "repo_id": "r/t", "required": True},
                {"id": "vision-model", "filename": "vision.gguf", "repo_id": "r/v", "required": True},
            ],
        }
        # Create only the text model file
        (tmp_path / "text.gguf").write_bytes(b"\x00" * 1024)

        with patch("server.load_manifest", return_value=manifest):
            result = srv.dispatch({
                "id": 1,
                "method": "check_models",
                "params": {"modelsDir": str(tmp_path)},
            })

        assert result["type"] == "result"
        data = result["data"]
        assert data["ready"] is False
        models = {m["id"]: m for m in data["models"]}
        assert models["text-model"]["exists"] is True
        assert models["text-model"]["size_bytes"] == 1024
        assert models["vision-model"]["exists"] is False

    def test_check_models_ready_when_all_present(self, tmp_path):
        """When all required models exist, ready=true."""
        from server import Server

        srv = Server()
        manifest = {
            "version": 1,
            "models": [
                {"id": "text-model", "filename": "text.gguf", "repo_id": "r/t", "required": True},
            ],
        }
        (tmp_path / "text.gguf").write_bytes(b"\x00" * 512)

        with patch("server.load_manifest", return_value=manifest):
            result = srv.dispatch({
                "id": 1,
                "method": "check_models",
                "params": {"modelsDir": str(tmp_path)},
            })

        assert result["type"] == "result"
        assert result["data"]["ready"] is True

    def test_check_models_uses_default_models_dir(self):
        """When modelsDir param is absent, uses get_models_dir()."""
        from server import Server

        srv = Server()
        manifest = {
            "version": 1,
            "models": [
                {"id": "m1", "filename": "m1.gguf", "repo_id": "r/m1", "required": True},
            ],
        }
        fake_dir = Path("/tmp/test_models_default_dir_abc")

        with patch("server.load_manifest", return_value=manifest), \
             patch("server.get_models_dir", return_value=fake_dir):
            result = srv.dispatch({
                "id": 1,
                "method": "check_models",
                "params": {},
            })

        assert result["type"] == "result"
        # Model should be missing since fake dir doesn't exist
        assert result["data"]["ready"] is False


class TestDownloadModels:
    """Acceptance: download_models fetches missing models with progress events."""

    def test_download_models_sends_progress_and_completes(self, tmp_path):
        """Given one missing model, download_models downloads it and sends
        setup_progress events with status downloading/complete, then returns
        all_models_ready."""
        from server import Server
        import server as srv_mod

        sent = []
        original_send = srv_mod.send
        srv_mod.send = lambda rid, rtype, data: sent.append(
            {"id": rid, "type": rtype, "data": data}
        )

        try:
            srv = Server()
            manifest = {
                "version": 1,
                "models": [
                    {"id": "text-model", "filename": "text.gguf",
                     "repo_id": "test/repo", "required": True},
                ],
            }

            def fake_download(repo_id, filename, local_dir, **kwargs):
                # Simulate download by creating the file
                Path(local_dir).mkdir(parents=True, exist_ok=True)
                (Path(local_dir) / filename).write_bytes(b"\x00" * 2048)
                return str(Path(local_dir) / filename)

            with patch("server.load_manifest", return_value=manifest), \
                 patch("server.hf_hub_download", side_effect=fake_download):
                result = srv.dispatch({
                    "id": 2,
                    "method": "download_models",
                    "params": {"modelsDir": str(tmp_path)},
                })

            assert result["type"] == "result"
            assert result["data"]["status"] == "all_models_ready"

            # Verify setup_progress events were sent
            progress_events = [m for m in sent if m["type"] == "setup_progress"]
            assert len(progress_events) >= 1
            # Should have at least a complete event
            complete_events = [e for e in progress_events
                               if e["data"].get("status") == "complete"]
            assert len(complete_events) == 1
            assert complete_events[0]["data"]["model_id"] == "text-model"

        finally:
            srv_mod.send = original_send

    def test_download_models_skips_existing(self, tmp_path):
        """Models already present are not re-downloaded."""
        from server import Server
        import server as srv_mod

        sent = []
        original_send = srv_mod.send
        srv_mod.send = lambda rid, rtype, data: sent.append(
            {"id": rid, "type": rtype, "data": data}
        )

        try:
            srv = Server()
            manifest = {
                "version": 1,
                "models": [
                    {"id": "text-model", "filename": "text.gguf",
                     "repo_id": "test/repo", "required": True},
                ],
            }
            # Pre-create the model file
            (tmp_path / "text.gguf").write_bytes(b"\x00" * 1024)

            with patch("server.load_manifest", return_value=manifest), \
                 patch("server.hf_hub_download") as mock_dl:
                result = srv.dispatch({
                    "id": 2,
                    "method": "download_models",
                    "params": {"modelsDir": str(tmp_path)},
                })

            assert result["type"] == "result"
            assert result["data"]["status"] == "all_models_ready"
            mock_dl.assert_not_called()

        finally:
            srv_mod.send = original_send

    def test_download_models_reports_error_per_model(self, tmp_path):
        """When a download fails, setup_progress error event includes model name."""
        from server import Server
        import server as srv_mod

        sent = []
        original_send = srv_mod.send
        srv_mod.send = lambda rid, rtype, data: sent.append(
            {"id": rid, "type": rtype, "data": data}
        )

        try:
            srv = Server()
            manifest = {
                "version": 1,
                "models": [
                    {"id": "vision-model", "filename": "vision.gguf",
                     "repo_id": "test/repo", "required": True},
                ],
            }

            with patch("server.load_manifest", return_value=manifest), \
                 patch("server.hf_hub_download",
                       side_effect=OSError("Network error")):
                result = srv.dispatch({
                    "id": 2,
                    "method": "download_models",
                    "params": {"modelsDir": str(tmp_path)},
                })

            # Should return error result
            assert result["type"] == "error"
            assert "vision-model" in result["data"]["message"]

            # Should have sent an error progress event
            error_events = [m for m in sent
                            if m["type"] == "setup_progress"
                            and m["data"].get("status") == "error"]
            assert len(error_events) == 1
            assert error_events[0]["data"]["model_id"] == "vision-model"
            assert "Network error" in error_events[0]["data"]["error"]

        finally:
            srv_mod.send = original_send


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


class TestQuerySources:
    """Test that query results include resolved source paths."""

    def test_query_result_includes_sources(self, tmp_path):
        """The result from handle_query includes a sources array with resolved paths."""
        from server import Server
        import server as srv_mod
        original_send = srv_mod.send
        srv_mod.send = lambda rid, rtype, data: None

        try:
            srv = Server()
            srv.state = "ready"

            # Create a real file so path resolution finds it
            (tmp_path / "file.pdf").write_bytes(b"fake")

            mock_agent = MagicMock()
            mock_agent.run.return_value = ("answer text", ["file.pdf"])
            srv.directories["test"] = {
                "agent": mock_agent,
                "conversation_history": [],
                "state": "ready",
                "dir_id": "test",
                "path": str(tmp_path),
            }

            result = srv.handle_query(1, {"text": "test query", "directoryId": "test"})
            assert result["type"] == "result"
            assert result["data"]["text"] == "answer text"
            assert result["data"]["sources"] == [str(tmp_path / "file.pdf")]
        finally:
            srv_mod.send = original_send

    def test_query_resolves_nested_source(self, tmp_path):
        """Sources in subdirectories are found via os.walk."""
        from server import Server
        import server as srv_mod
        original_send = srv_mod.send
        srv_mod.send = lambda rid, rtype, data: None

        try:
            srv = Server()
            srv.state = "ready"

            sub = tmp_path / "sub"
            sub.mkdir()
            (sub / "deep.pdf").write_bytes(b"fake")

            mock_agent = MagicMock()
            mock_agent.run.return_value = ("found it", ["deep.pdf"])
            srv.directories["test"] = {
                "agent": mock_agent,
                "conversation_history": [],
                "state": "ready",
                "dir_id": "test",
                "path": str(tmp_path),
            }

            result = srv.handle_query(1, {"text": "find deep", "directoryId": "test"})
            assert result["data"]["sources"] == [str(sub / "deep.pdf")]
        finally:
            srv_mod.send = original_send

    def test_query_unresolvable_source_fallback(self):
        """Sources that can't be found on disk fall back to the bare name."""
        from server import Server
        import server as srv_mod
        original_send = srv_mod.send
        srv_mod.send = lambda rid, rtype, data: None

        try:
            srv = Server()
            srv.state = "ready"

            mock_agent = MagicMock()
            mock_agent.run.return_value = ("text", ["nonexistent.pdf"])
            srv.directories["test"] = {
                "agent": mock_agent,
                "conversation_history": [],
                "state": "ready",
                "dir_id": "test",
                "path": "/tmp/empty_dir_unlikely_to_exist",
            }

            result = srv.handle_query(1, {"text": "q", "directoryId": "test"})
            assert result["data"]["sources"] == ["nonexistent.pdf"]
        finally:
            srv_mod.send = original_send

    def test_query_all_includes_sources(self, tmp_path):
        """_query_all also includes sources in each directory result."""
        from server import Server
        import server as srv_mod
        original_send = srv_mod.send
        srv_mod.send = lambda rid, rtype, data: None

        try:
            srv = Server()
            srv.state = "ready"

            (tmp_path / "report.pdf").write_bytes(b"data")

            mock_agent = MagicMock()
            mock_agent.run.return_value = ("merged", ["report.pdf"])
            srv.directories["d1"] = {
                "agent": mock_agent,
                "conversation_history": [],
                "state": "ready",
                "dir_id": "d1",
                "path": str(tmp_path),
            }

            result = srv._query_all(1, "search everything")
            assert result["type"] == "result"
            assert len(result["data"]["results"]) == 1
            r = result["data"]["results"][0]
            assert r["text"] == "merged"
            assert r["sources"] == [str(tmp_path / "report.pdf")]
        finally:
            srv_mod.send = original_send


def _init_patches():
    """Return a dict of patch targets for handle_init's local imports."""
    return {
        "chat.build_index": "test_index",
        "chat.find_index_path": "/tmp/test_index",
        "chat.get_index_name": "test_index",
    }


def _make_init_context(tmp_path, mock_searcher_inst, mock_captioner_cls=None,
                       mock_cache_cls=None):
    """Create a stack of patches for handle_init dependencies."""
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch("chat.build_index", return_value="test_index"))
    stack.enter_context(patch("chat.find_index_path", return_value="/tmp/test_index"))
    stack.enter_context(patch("leann.LeannSearcher", return_value=MagicMock()))
    stack.enter_context(patch("file_reader.FileReader", return_value=MagicMock()))
    stack.enter_context(patch("toolbox.ToolBox", return_value=MagicMock()))
    stack.enter_context(patch("searcher.Searcher", return_value=mock_searcher_inst))
    stack.enter_context(patch("tools.ToolRegistry", return_value=MagicMock()))
    stack.enter_context(patch("router.route", return_value=MagicMock()))
    stack.enter_context(patch("rewriter.QueryRewriter", return_value=MagicMock()))
    stack.enter_context(patch("agent.Agent", return_value=MagicMock()))
    if mock_captioner_cls is not None:
        stack.enter_context(patch("image_captioner.ImageCaptioner", mock_captioner_cls))
    if mock_cache_cls is not None:
        stack.enter_context(patch("caption_cache.CaptionCache", mock_cache_cls))
    return stack


class TestEagerVisionLoading:
    """Test that server eagerly loads vision model during init."""

    def test_handle_init_calls_load_vision_after_load(self, tmp_path):
        """Given handle_init is called, load_vision() is called after load()
        during the loading_model phase."""
        from server import Server
        import server as srv_mod

        sent = []
        original_send = srv_mod.send
        srv_mod.send = lambda rid, rtype, data: sent.append((rtype, data))

        try:
            srv = Server()

            mock_model = MagicMock()
            mock_model.generate.return_value = "Summary"

            # Track call order
            call_order = []
            mock_model.load.side_effect = lambda: call_order.append("load")
            mock_model.load_vision.side_effect = lambda: call_order.append("load_vision")

            (tmp_path / "doc.txt").write_text("hello")

            mock_searcher_inst = MagicMock()
            mock_searcher_inst.search_and_extract.return_value = "facts"

            mock_captioner = MagicMock()
            mock_captioner._find_images.return_value = []
            mock_captioner_cls = MagicMock(return_value=mock_captioner)
            mock_cache_cls = MagicMock(return_value=MagicMock())

            with _make_init_context(tmp_path, mock_searcher_inst,
                                    mock_captioner_cls, mock_cache_cls), \
                 patch("models.ModelManager", return_value=mock_model):
                result = srv.handle_init(1, {"dataDir": str(tmp_path)})

            assert result["data"]["status"] == "ready"
            mock_model.load.assert_called_once()
            mock_model.load_vision.assert_called_once()
            assert call_order == ["load", "load_vision"]

        finally:
            srv_mod.send = original_send


class TestForegroundCaptioning:
    """Acceptance: handle_init runs summary + captioning inline (no background thread)."""

    def test_handle_init_sends_status_events_inline_and_no_thread(self, tmp_path):
        """AC1-5: status events in order, server calls captioner.run() directly
        without pre-scanning, no thread, errors don't block ready."""
        from server import Server
        import server as srv_mod

        sent = []
        original_send = srv_mod.send
        srv_mod.send = lambda rid, rtype, data: sent.append((rtype, data))

        try:
            srv = Server()

            # Pre-wire model so handle_init skips model loading
            mock_model = MagicMock()
            mock_model.generate.return_value = "Test summary"
            srv.model = mock_model

            # Create test files including an image
            (tmp_path / "doc.txt").write_text("hello")
            (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

            mock_searcher_inst = MagicMock()
            mock_searcher_inst.search_and_extract.return_value = "some facts"

            mock_captioner = MagicMock()
            mock_captioner_cls = MagicMock(return_value=mock_captioner)

            mock_cache_cls = MagicMock(return_value=MagicMock())

            with _make_init_context(tmp_path, mock_searcher_inst,
                                    mock_captioner_cls, mock_cache_cls):
                result = srv.handle_init(1, {"dataDir": str(tmp_path)})

            # AC1: "summarizing" status sent
            status_events = [(t, d) for t, d in sent if t == "status"]
            status_states = [d["state"] for _, d in status_events]
            assert "summarizing" in status_states, f"Expected 'summarizing' in {status_states}"

            # AC2: server does NOT send "captioning" status (captioner.run() is responsible)
            server_captioning = [s for s in status_states if s == "captioning"]
            assert len(server_captioning) == 0, \
                "Server must not send captioning status — captioner.run() owns that"

            # AC3: server does NOT call _find_images() — no pre-scan
            mock_captioner._find_images.assert_not_called()

            # AC4: no threading.Thread in handle_init
            import inspect
            source = inspect.getsource(srv.handle_init)
            assert "threading.Thread" not in source, "handle_init must not create threads"

            # AC5: result is still ready
            assert result["data"]["status"] == "ready"

            # Verify captioner.run() was called directly
            mock_captioner.run.assert_called_once()

        finally:
            srv_mod.send = original_send

    def test_handle_init_does_not_prescan_images(self, tmp_path):
        """Server does not call _find_images() or count uncached — delegates to captioner.run()."""
        from server import Server
        import server as srv_mod

        sent = []
        original_send = srv_mod.send
        srv_mod.send = lambda rid, rtype, data: sent.append((rtype, data))

        try:
            srv = Server()
            mock_model = MagicMock()
            mock_model.generate.return_value = "Summary"
            srv.model = mock_model

            (tmp_path / "doc.txt").write_text("hello")
            (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

            mock_searcher_inst = MagicMock()
            mock_searcher_inst.search_and_extract.return_value = "facts"

            mock_captioner = MagicMock()
            mock_captioner_cls = MagicMock(return_value=mock_captioner)
            mock_cache_cls = MagicMock(return_value=MagicMock())

            with _make_init_context(tmp_path, mock_searcher_inst,
                                    mock_captioner_cls, mock_cache_cls):
                result = srv.handle_init(1, {"dataDir": str(tmp_path)})

            # Server must NOT call _find_images or send captioning status
            mock_captioner._find_images.assert_not_called()
            status_states = [d["state"] for t, d in sent if t == "status"]
            assert "captioning" not in status_states, \
                "Server must not send captioning status — captioner.run() owns that"

            # But captioner.run() must be called
            mock_captioner.run.assert_called_once()
            assert result["data"]["status"] == "ready"

        finally:
            srv_mod.send = original_send

    def test_handle_init_errors_do_not_block_ready(self, tmp_path):
        """AC5: exceptions in summary or captioning don't prevent ready."""
        from server import Server
        import server as srv_mod

        sent = []
        original_send = srv_mod.send
        srv_mod.send = lambda rid, rtype, data: sent.append((rtype, data))

        try:
            srv = Server()
            mock_model = MagicMock()
            mock_model.generate.side_effect = RuntimeError("model exploded")
            srv.model = mock_model

            (tmp_path / "doc.txt").write_text("hello")

            mock_searcher_inst = MagicMock()
            mock_searcher_inst.search_and_extract.side_effect = RuntimeError("search failed")

            mock_captioner = MagicMock()
            mock_captioner._find_images.side_effect = RuntimeError("captioner broke")
            mock_captioner_cls = MagicMock(return_value=mock_captioner)
            mock_cache_cls = MagicMock(return_value=MagicMock())

            with _make_init_context(tmp_path, mock_searcher_inst,
                                    mock_captioner_cls, mock_cache_cls):
                result = srv.handle_init(1, {"dataDir": str(tmp_path)})

            assert result["data"]["status"] == "ready"
            dir_updates = [d for t, d in sent if t == "directory_update" and d.get("state") == "ready"]
            assert len(dir_updates) > 0, "Must still send ready directory_update"

        finally:
            srv_mod.send = original_send
