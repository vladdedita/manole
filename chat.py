"""
Interactive chat mode: index a directory and chat about its contents.

Uses `leann build` CLI for indexing (handles PDFs, chunking, etc.)
and an agent loop for answering queries.

Usage:
    uv run python chat.py /path/to/directory
    uv run python chat.py                      # defaults to ./test_data
    uv run python chat.py --reuse myindex      # skip indexing, reuse existing index
"""

import subprocess
import sys
import time
from pathlib import Path


def get_index_name(data_dir: Path) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_-]", "_", data_dir.name)


def build_index(data_dir: Path, force: bool = False, pipeline: str = "leann") -> str:
    index_name = get_index_name(data_dir)

    if pipeline == "kreuzberg":
        from indexer import KreuzbergIndexer

        print(f"Indexing: {data_dir} (kreuzberg pipeline)")
        t0 = time.time()
        indexer = KreuzbergIndexer()
        indexer.build(data_dir, index_name, force=force)
        print(f"\nIndex built in {time.time() - t0:.1f}s")
        return index_name

    # Use the leann binary from the venv
    leann_bin = Path(sys.executable).parent / "leann"
    cmd = [
        str(leann_bin),
        "build", index_name,
        "--docs", str(data_dir),
        "--no-compact",
    ]
    if force:
        cmd.append("--force")

    print(f"Indexing: {data_dir}")
    print(f"Running: {' '.join(cmd)}\n")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\nleann build failed (exit code {result.returncode})")
        sys.exit(1)

    print(f"\nIndex built in {time.time() - t0:.1f}s")
    return index_name


def find_index_path(index_name: str) -> str:
    """Find the .leann index file created by the CLI."""
    # leann stores indexes in ./.leann/indexes/<name>/ (local) or ~/.leann/indexes/<name>/
    candidates = [
        Path(".leann") / "indexes" / index_name,
        Path.home() / ".leann" / "indexes" / index_name,
        Path("./indexes") / index_name,
    ]

    for base in candidates:
        # The index is a set of files like documents.leann.meta.json
        # LeannSearcher expects the path without the .meta.json suffix
        for meta_file in base.rglob("*.leann.meta.json"):
            return str(meta_file).removesuffix(".meta.json")

    # Try leann list to find it
    leann_bin = str(Path(sys.executable).parent / "leann")
    result = subprocess.run(
        [leann_bin, "list"],
        capture_output=True, text=True,
    )
    print(f"Could not locate index '{index_name}'. Available indexes:")
    print(result.stdout)
    sys.exit(1)


def chat_loop(index_name: str, data_dir: str):
    from leann import LeannSearcher
    from models import ModelManager
    from searcher import Searcher
    from file_reader import FileReader
    from toolbox import ToolBox
    from tools import ToolRegistry
    from router import route
    from rewriter import QueryRewriter
    from agent import Agent

    print("\nLoading LFM2.5-1.2B-Instruct...")
    t0 = time.time()

    index_path = find_index_path(index_name)
    print(f"Using index: {index_path}")

    # Load model
    model = ModelManager()
    model.load()

    # Create search and tools
    leann_searcher = LeannSearcher(index_path, enable_warmup=True)
    file_reader = FileReader()
    toolbox = ToolBox(data_dir, debug=True)
    searcher = Searcher(leann_searcher, model, file_reader=file_reader, toolbox=toolbox, debug=True)
    tool_registry = ToolRegistry(searcher, toolbox, debug=True)

    # Create router (module-level function wrapped for Agent interface)
    class RouterWrapper:
        @staticmethod
        def route(query, intent=None):
            return route(query, intent=intent, debug=agent.debug)

    # Create rewriter and agent
    rewriter = QueryRewriter(model, debug=True)
    agent = Agent(model, tool_registry, RouterWrapper(), rewriter=rewriter, debug=True)

    print(f"Ready in {time.time() - t0:.1f}s")
    print("=" * 50)
    print("Ask anything about your files. Type 'quit' to exit.")
    print("Type 'debug' to toggle trace.")
    print("=" * 50)

    conversation_history = []

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if query.lower() == "debug":
            agent.debug = not agent.debug
            searcher.debug = agent.debug
            rewriter.debug = agent.debug
            tool_registry.debug = agent.debug
            toolbox.debug = agent.debug
            print(f"Trace: {'ON' if agent.debug else 'OFF'}")
            continue

        t0 = time.time()
        response = agent.run(query, history=conversation_history)
        elapsed = time.time() - t0
        print(f"\n{response}")
        print(f"\n({elapsed:.1f}s)")

        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": response})
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]


def main():
    args = sys.argv[1:]

    if "--reuse" in args:
        idx = args.index("--reuse")
        if idx + 1 >= len(args):
            print("Error: --reuse requires an index name")
            sys.exit(1)
        index_name = args[idx + 1]
        data_dir = args[idx + 2] if idx + 2 < len(args) else "./test_data"
        chat_loop(index_name, str(Path(data_dir).resolve()))
        return

    pipeline = "leann"
    if "--pipeline" in args:
        idx = args.index("--pipeline")
        if idx + 1 >= len(args):
            print("Error: --pipeline requires a value (leann|kreuzberg)")
            sys.exit(1)
        pipeline = args[idx + 1]
        args = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]

    force = "--force" in args
    args = [a for a in args if a != "--force"]

    if args:
        data_dir = Path(args[0]).resolve()
    else:
        data_dir = Path("./test_data").resolve()

    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a directory")
        sys.exit(1)

    index_name = build_index(data_dir, force=force, pipeline=pipeline)
    chat_loop(index_name, str(data_dir))


if __name__ == "__main__":
    main()
