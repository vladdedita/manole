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
    return data_dir.name.replace(" ", "_").replace("/", "_")


def build_index(data_dir: Path, force: bool = False) -> str:
    index_name = get_index_name(data_dir)

    # Use the leann binary from the venv
    leann_bin = Path(sys.executable).parent / "leann"
    cmd = [
        str(leann_bin),
        "build", index_name,
        "--docs", str(data_dir),
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
    from toolbox import ToolBox
    from tools import ToolRegistry
    from router import route
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
    searcher = Searcher(leann_searcher, model, debug=True)
    toolbox = ToolBox(data_dir)
    tool_registry = ToolRegistry(searcher, toolbox)

    # Create router (module-level function wrapped for Agent interface)
    class RouterWrapper:
        @staticmethod
        def route(query):
            return route(query)

    # Create agent
    agent = Agent(model, tool_registry, RouterWrapper(), debug=True)

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

    force = "--force" in args
    args = [a for a in args if a != "--force"]

    if args:
        data_dir = Path(args[0]).resolve()
    else:
        data_dir = Path("./test_data").resolve()

    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a directory")
        sys.exit(1)

    index_name = build_index(data_dir, force=force)
    chat_loop(index_name, str(data_dir))


if __name__ == "__main__":
    main()
