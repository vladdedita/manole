"""NDJSON stdio server — thin adapter between Electron and the Python backend.

Reads JSON requests from stdin (one per line), dispatches to existing modules,
writes JSON responses to stdout. The core modules (Agent, Searcher, ModelManager)
are untouched — this is purely a protocol adapter.

Protocol:
    Request:  {"id": int, "method": str, "params": dict}
    Response: {"id": int|null, "type": str, "data": dict}
"""
import json
import sys
import time
from pathlib import Path


def parse_request(line: str) -> dict | None:
    try:
        req = json.loads(line.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(req, dict) or "method" not in req:
        return None
    return req


def format_response(req_id, resp_type: str, data: dict) -> str:
    return json.dumps({"id": req_id, "type": resp_type, "data": data})


def send(req_id, resp_type: str, data: dict):
    """Write a single NDJSON line to stdout."""
    print(format_response(req_id, resp_type, data), flush=True)


class Server:
    """NDJSON protocol server wrapping the existing backend."""

    def __init__(self):
        self.state = "not_initialized"
        self.debug = False
        self.running = True
        self.start_time = time.time()

        # Initialized by handle_init
        self.model = None
        self.agent = None
        self.searcher = None
        self.rewriter = None
        self.conversation_history = []
        self.data_dir = None
        self.index_name = None

    def _log(self, message: str):
        """Send a log message to the UI via stderr."""
        import sys as _sys
        _sys.stderr.write(message + "\n")
        _sys.stderr.flush()

    def handle_ping(self, req_id) -> dict:
        data = {
            "state": self.state,
            "uptime": round(time.time() - self.start_time, 1),
        }
        return {"id": req_id, "type": "result", "data": data}

    def handle_toggle_debug(self, req_id) -> dict:
        self.debug = not self.debug
        if self.agent:
            self.agent.debug = self.debug
        if self.searcher:
            self.searcher.debug = self.debug
        if self.rewriter:
            self.rewriter.debug = self.debug
        return {"id": req_id, "type": "result", "data": {"debug": self.debug}}

    def handle_shutdown(self, req_id) -> dict:
        self.running = False
        return {"id": req_id, "type": "result", "data": {"status": "shutting_down"}}

    def handle_init(self, req_id, params: dict) -> dict:
        """Initialize the backend: load model, build/reuse index, wire agent."""
        from chat import build_index, find_index_path, get_index_name
        from leann import LeannSearcher
        from models import ModelManager
        from searcher import Searcher
        from file_reader import FileReader
        from toolbox import ToolBox
        from tools import ToolRegistry
        from router import route
        from rewriter import QueryRewriter
        from agent import Agent

        data_dir = params.get("dataDir", "./test_data")
        data_dir_path = Path(data_dir).resolve()

        if not data_dir_path.is_dir():
            return {"id": req_id, "type": "error", "data": {"message": f"Not a directory: {data_dir}"}}

        send(None, "status", {"state": "loading_model"})
        self._log("Loading model...")

        # Load model
        self.model = ModelManager()
        self.model.load()
        self._log("Model loaded.")

        send(None, "status", {"state": "indexing"})
        self._log(f"Indexing {data_dir_path}...")

        # Build or reuse index
        reuse = params.get("reuse")
        if reuse:
            self.index_name = reuse
            self._log(f"Reusing index: {reuse}")
        else:
            self.index_name = build_index(data_dir_path)
            self._log(f"Index built: {self.index_name}")

        index_path = find_index_path(self.index_name)

        # Wire components
        leann_searcher = LeannSearcher(index_path, enable_warmup=True)
        file_reader = FileReader()
        toolbox = ToolBox(str(data_dir_path))
        self.searcher = Searcher(
            leann_searcher, self.model,
            file_reader=file_reader, toolbox=toolbox,
            debug=self.debug,
        )
        tool_registry = ToolRegistry(self.searcher, toolbox)

        class RouterWrapper:
            @staticmethod
            def route(query, intent=None):
                return route(query, intent=intent)

        self.rewriter = QueryRewriter(self.model, debug=self.debug)
        self.agent = Agent(
            self.model, tool_registry, RouterWrapper(),
            rewriter=self.rewriter, debug=self.debug,
        )
        self.data_dir = str(data_dir_path)
        self.conversation_history = []
        self.state = "ready"
        self._log("All components wired. Ready.")

        send(None, "status", {"state": "ready"})
        return {
            "id": req_id, "type": "result",
            "data": {"status": "ready", "indexName": self.index_name},
        }

    def handle_query(self, req_id, params: dict) -> dict:
        """Run agent loop with streaming tokens."""
        if self.state != "ready":
            return {"id": req_id, "type": "error", "data": {"message": "Not initialized"}}

        query = params.get("text", "").strip()
        if not query:
            return {"id": req_id, "type": "error", "data": {"message": "Empty query"}}

        self._log(f"Query: {query[:80]}")

        def on_token(text):
            send(req_id, "token", {"text": text})

        response = self.agent.run(
            query,
            history=self.conversation_history,
            on_token=on_token,
        )

        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        return {"id": req_id, "type": "result", "data": {"text": response}}

    def handle_list_indexes(self, req_id) -> dict:
        """List available LEANN indexes."""
        indexes = []
        for base in [Path(".leann/indexes"), Path.home() / ".leann/indexes"]:
            if base.is_dir():
                for d in sorted(base.iterdir()):
                    if d.is_dir():
                        indexes.append(d.name)
        return {"id": req_id, "type": "result", "data": {"indexes": indexes}}

    def dispatch(self, req: dict):
        """Route a parsed request to the appropriate handler."""
        req_id = req.get("id")
        method = req["method"]
        params = req.get("params", {})

        handlers = {
            "ping": lambda: self.handle_ping(req_id),
            "init": lambda: self.handle_init(req_id, params),
            "query": lambda: self.handle_query(req_id, params),
            "toggle_debug": lambda: self.handle_toggle_debug(req_id),
            "list_indexes": lambda: self.handle_list_indexes(req_id),
            "shutdown": lambda: self.handle_shutdown(req_id),
        }

        handler = handlers.get(method)
        if not handler:
            return {"id": req_id, "type": "error", "data": {"message": f"Unknown method: {method}"}}

        try:
            return handler()
        except Exception as e:
            return {"id": req_id, "type": "error", "data": {"message": str(e)}}

    def run(self, input_stream=None):
        """Main loop: read stdin, dispatch, write stdout."""
        stream = input_stream or sys.stdin
        for line in stream:
            line = line.strip()
            if not line:
                continue

            req = parse_request(line)
            if req is None:
                send(None, "error", {"message": "Invalid JSON"})
                continue

            result = self.dispatch(req)
            if result:
                print(format_response(result["id"], result["type"], result["data"]), flush=True)

            if not self.running:
                break


if __name__ == "__main__":
    # Redirect all print output from core modules to stderr so only
    # NDJSON protocol messages go to stdout
    import io
    import os

    # Save real stdout for NDJSON
    _real_stdout = sys.stdout

    # Redirect Python's stdout to stderr for all non-protocol prints
    # (build_index, model loading, debug traces, etc.)
    sys.stdout = sys.stderr

    # Override send() to use the real stdout
    import builtins
    _original_print = builtins.print

    def _ndjson_print(*args, **kwargs):
        """Print to real stdout only when flush=True (our protocol sends)."""
        if kwargs.get("flush"):
            kwargs["file"] = _real_stdout
        _original_print(*args, **kwargs)

    builtins.print = _ndjson_print

    server = Server()
    server.run(sys.stdin)
