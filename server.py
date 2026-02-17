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

from graph import build_file_graph


def make_dir_id(path: str) -> str:
    """Derive a stable directory ID from an absolute path."""
    return Path(path).name.replace(" ", "_").replace("/", "_")


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

        # Shared model (loaded once on first init)
        self.model = None
        self.rewriter = None

        # Per-directory state
        self.directories: dict[str, dict] = {}

    def _log(self, message: str):
        """Send a log message to the UI via stderr."""
        import sys as _sys
        _sys.stderr.write(message + "\n")
        _sys.stderr.flush()

    def _collect_stats(self, data_dir: Path) -> dict:
        """Walk a directory and return file statistics."""
        file_count = 0
        total_size = 0
        types: dict[str, int] = {}
        size_by_type: dict[str, int] = {}
        largest_files: list[dict] = []
        dir_count = 0
        max_depth = 0

        base_depth = len(data_dir.parts)
        for p in data_dir.rglob("*"):
            if p.is_dir():
                dir_count += 1
                depth = len(p.parts) - base_depth
                if depth > max_depth:
                    max_depth = depth
                continue
            if p.is_file():
                file_size = p.stat().st_size
                file_count += 1
                total_size += file_size
                ext = p.suffix.lstrip(".").lower()
                if ext:
                    types[ext] = types.get(ext, 0) + 1
                    size_by_type[ext] = size_by_type.get(ext, 0) + file_size
                largest_files.append({"name": p.name, "size": file_size})

        largest_files.sort(key=lambda f: f["size"], reverse=True)
        largest_files = largest_files[:3]

        return {
            "fileCount": file_count,
            "totalSize": total_size,
            "types": types,
            "sizeByType": size_by_type,
            "largestFiles": largest_files,
            "avgFileSize": total_size // file_count if file_count else 0,
            "dirs": {"count": dir_count, "maxDepth": max_depth},
        }

    def _generate_summary(self, dir_id: str) -> str:
        """Query the index to generate a content-aware summary."""
        if not self.model:
            return ""
        entry = self.directories.get(dir_id)
        if not entry or "searcher" not in entry:
            return ""
        searcher = entry["searcher"]
        facts = searcher.search_and_extract(
            "What are the main topics, purpose, and content of these documents?",
            top_k=5,
        )
        prompt = (
            f"Based on these document excerpts:\n{facts}\n\n"
            "In 2-3 sentences, describe what this collection of documents is about. "
            "Be specific about the project or domain. Be concise."
        )
        messages = [{"role": "user", "content": prompt}]
        result = self.model.generate(messages)
        return (result or "").strip()

    def handle_ping(self, req_id) -> dict:
        data = {
            "state": self.state,
            "uptime": round(time.time() - self.start_time, 1),
        }
        return {"id": req_id, "type": "result", "data": data}

    def handle_toggle_debug(self, req_id) -> dict:
        self.debug = not self.debug
        for entry in self.directories.values():
            if "agent" in entry:
                entry["agent"].debug = self.debug
            if "searcher" in entry:
                entry["searcher"].debug = self.debug
        if self.rewriter:
            self.rewriter.debug = self.debug
        return {"id": req_id, "type": "result", "data": {"debug": self.debug}}

    def handle_shutdown(self, req_id) -> dict:
        self.running = False
        return {"id": req_id, "type": "result", "data": {"status": "shutting_down"}}

    def handle_init(self, req_id, params: dict) -> dict:
        """Initialize a directory: load model (once), build/reuse index, wire agent."""
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

        dir_id = make_dir_id(str(data_dir_path))

        # Load model only once (shared across directories)
        if self.model is None:
            send(None, "status", {"state": "loading_model"})
            self._log("Loading model...")
            self.model = ModelManager()
            self.model.load()
            self._log("Model loaded.")

        send(None, "directory_update", {"directoryId": dir_id, "state": "indexing"})
        self._log(f"Indexing {data_dir_path}...")

        # Build or reuse index
        reuse = params.get("reuse")
        if reuse:
            index_name = reuse
            self._log(f"Reusing index: {reuse}")
        else:
            index_name = build_index(data_dir_path)
            self._log(f"Index built: {index_name}")

        index_path = find_index_path(index_name)

        # Wire components
        leann_searcher = LeannSearcher(index_path, enable_warmup=True)
        file_reader = FileReader()
        toolbox = ToolBox(str(data_dir_path))
        searcher = Searcher(
            leann_searcher, self.model,
            file_reader=file_reader, toolbox=toolbox,
            debug=self.debug,
        )
        tool_registry = ToolRegistry(searcher, toolbox)

        class RouterWrapper:
            @staticmethod
            def route(query, intent=None):
                return route(query, intent=intent)

        self.rewriter = QueryRewriter(self.model, debug=self.debug)
        agent = Agent(
            self.model, tool_registry, RouterWrapper(),
            rewriter=self.rewriter, debug=self.debug,
        )

        # Collect stats
        stats = self._collect_stats(data_dir_path)

        # Store directory entry
        self.directories[dir_id] = {
            "dir_id": dir_id,
            "path": str(data_dir_path),
            "index_name": index_name,
            "searcher": searcher,
            "agent": agent,
            "state": "ready",
            "stats": stats,
            "summary": "",
            "conversation_history": [],
        }

        self.state = "ready"
        self._log("All components wired. Ready.")

        # Send ready immediately so the UI unblocks
        send(None, "directory_update", {
            "directoryId": dir_id, "state": "ready",
            "stats": stats,
        })

        # Generate summary (may be slow) and send as a follow-up update
        try:
            self._log(f"Generating summary for {dir_id}...")
            summary = self._generate_summary(dir_id)
            self._log(f"Summary result: {repr(summary[:100]) if summary else '(empty)'}")
            self.directories[dir_id]["summary"] = summary
            if summary:
                send(None, "directory_update", {
                    "directoryId": dir_id, "state": "ready",
                    "stats": stats, "summary": summary,
                })
                self._log(f"Sent directory_update with summary for {dir_id}")
        except Exception as exc:
            self._log(f"Summary generation failed: {exc}")
        return {
            "id": req_id, "type": "result",
            "data": {"status": "ready", "directoryId": dir_id, "indexName": index_name},
        }

    def handle_query(self, req_id, params: dict) -> dict:
        """Run agent loop with streaming tokens."""
        if self.state != "ready":
            return {"id": req_id, "type": "error", "data": {"message": "Not initialized"}}

        query = params.get("text", "").strip()
        if not query:
            return {"id": req_id, "type": "error", "data": {"message": "Empty query"}}

        search_all = params.get("searchAll", False)
        if search_all:
            return self._query_all(req_id, query)

        # Resolve directory
        dir_id = params.get("directoryId")
        if dir_id is None:
            # Fall back to first ready directory
            ready = [e for e in self.directories.values() if e.get("state") == "ready"]
            if not ready:
                return {"id": req_id, "type": "error", "data": {"message": "No ready directories"}}
            entry = ready[0]
        else:
            entry = self.directories.get(dir_id)
            if entry is None:
                return {"id": req_id, "type": "error", "data": {"message": f"Unknown directory: {dir_id}"}}
            if entry.get("state") != "ready":
                return {"id": req_id, "type": "error", "data": {"message": f"Directory not ready: {dir_id}"}}

        self._log(f"Query: {query[:80]}")

        step_count = [0]

        def on_token(text):
            send(req_id, "token", {"text": text})

        def on_step(step: int, tool: str, params: dict):
            """Notify UI that a new agent step is starting."""
            step_count[0] = step
            send(req_id, "agent_step", {"step": step, "tool": tool, "params": params})

        agent = entry["agent"]
        conversation_history = entry["conversation_history"]

        response = agent.run(
            query,
            history=conversation_history,
            on_token=on_token,
            on_step=on_step,
        )

        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": response})
        if len(conversation_history) > 10:
            entry["conversation_history"] = conversation_history[-10:]

        return {"id": req_id, "type": "result", "data": {"text": response}}

    def _query_all(self, req_id, query: str) -> dict:
        """Run query against all ready directories and merge results."""
        results = []
        for entry in self.directories.values():
            if entry.get("state") != "ready":
                continue
            agent = entry["agent"]
            conversation_history = entry["conversation_history"]

            def on_token(text):
                send(req_id, "token", {"text": text})

            response = agent.run(query, history=conversation_history, on_token=on_token)

            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": response})
            if len(conversation_history) > 10:
                entry["conversation_history"] = conversation_history[-10:]

            results.append({"directoryId": entry["dir_id"], "text": response})

        return {"id": req_id, "type": "result", "data": {"results": results}}

    def handle_remove_directory(self, req_id, params: dict) -> dict:
        """Remove a directory from the server."""
        dir_id = params.get("directoryId")
        if not dir_id or dir_id not in self.directories:
            return {"id": req_id, "type": "error", "data": {"message": f"Unknown directory: {dir_id}"}}
        del self.directories[dir_id]
        # If no directories left, reset state
        if not self.directories:
            self.state = "not_initialized"
        return {"id": req_id, "type": "result", "data": {"status": "ok"}}

    def handle_reindex(self, req_id, params: dict) -> dict:
        """Re-index a previously added directory."""
        dir_id = params.get("directoryId")
        if not dir_id or dir_id not in self.directories:
            return {"id": req_id, "type": "error", "data": {"message": f"Unknown directory: {dir_id}"}}
        stored_path = self.directories[dir_id]["path"]
        return self.handle_init(req_id, {"dataDir": stored_path})

    def handle_list_indexes(self, req_id) -> dict:
        """List available LEANN indexes."""
        indexes = []
        for base in [Path(".leann/indexes"), Path.home() / ".leann/indexes"]:
            if base.is_dir():
                for d in sorted(base.iterdir()):
                    if d.is_dir():
                        indexes.append(d.name)
        return {"id": req_id, "type": "result", "data": {"indexes": indexes}}

    def handle_get_file_graph(self, req_id, params: dict) -> dict:
        """Compute and return the file relationship graph."""
        dir_id = params.get("directoryId")
        if not dir_id or dir_id not in self.directories:
            return {"id": req_id, "type": "error", "data": {"message": f"Unknown directory: {dir_id}"}}

        entry = self.directories[dir_id]
        if entry.get("state") != "ready":
            return {"id": req_id, "type": "error", "data": {"message": f"Directory not ready: {dir_id}"}}

        # Return cached graph if available
        if "file_graph" in entry:
            return {"id": req_id, "type": "result", "data": entry["file_graph"]}

        searcher = entry.get("searcher")
        if not searcher:
            return {"id": req_id, "type": "error", "data": {"message": "No searcher available"}}

        self._log(f"Computing file graph for {dir_id}...")
        graph = build_file_graph(searcher.leann, entry["path"])
        entry["file_graph"] = graph
        self._log(f"File graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")

        return {"id": req_id, "type": "result", "data": graph}

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
            "remove_directory": lambda: self.handle_remove_directory(req_id, params),
            "reindex": lambda: self.handle_reindex(req_id, params),
            "getFileGraph": lambda: self.handle_get_file_graph(req_id, params),
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
        self._log("Server ready, waiting for commands...")
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
    import os
    import builtins

    # Save real stdout fd for NDJSON protocol messages
    _real_stdout_fd = os.dup(1)
    _real_stdout = os.fdopen(_real_stdout_fd, "w")

    # Redirect fd 1 to stderr so C libraries (llama.cpp, ggml)
    # that write directly to fd 1 go to stderr instead
    os.dup2(2, 1)

    # Also redirect Python-level stdout to stderr
    sys.stdout = sys.stderr

    # Override print so protocol sends (flush=True) go to the real stdout
    _original_print = builtins.print

    def _ndjson_print(*args, **kwargs):
        """Print to real stdout only when flush=True (our protocol sends)."""
        if kwargs.get("flush"):
            kwargs["file"] = _real_stdout
        _original_print(*args, **kwargs)

    builtins.print = _ndjson_print

    server = Server()
    server.run(sys.stdin)
