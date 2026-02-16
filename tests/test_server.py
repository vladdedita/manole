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


class TestShutdown:
    """Test clean shutdown."""

    def test_shutdown_sets_running_false(self):
        from server import Server
        srv = Server()
        srv.running = True
        result = srv.handle_shutdown(1)
        assert srv.running is False
