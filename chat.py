"""
Interactive chat mode: index a directory and chat about its contents.

Uses `leann build` CLI for indexing (handles PDFs, chunking, etc.)
and LeannChat for the interactive RAG loop.

Usage:
    uv run python chat.py /path/to/directory
    uv run python chat.py                      # defaults to ./test_data
    uv run python chat.py --reuse myindex      # skip indexing, reuse existing index
"""

import subprocess
import sys
import time
from pathlib import Path
from leann import LeannChat
import json
import re


def parse_json(text: str) -> dict | None:
    """Parse JSON from LLM output with fallback regex extraction."""
    # Try direct parse
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find a JSON object in the text
    match = re.search(r'\{[^{}]+\}', text)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: extract "relevant" field via regex
    rel_match = re.search(r'"relevant"\s*:\s*(true|false)', text, re.IGNORECASE)
    if rel_match:
        relevant = rel_match.group(1).lower() == "true"
        facts_match = re.search(r'"facts"\s*:\s*\[([^\]]*)\]', text)
        facts = []
        if facts_match:
            facts = [f.strip().strip('"') for f in facts_match.group(1).split(",") if f.strip()]
        return {"relevant": relevant, "facts": facts}

    return None


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


TOP_K = 5
DEBUG_SOURCES = True

PLANNER_PROMPT = (
    "Analyze this user question and extract search parameters.\n"
    "Question: {query}\n\n"
    "Output a JSON object with:\n"
    '- "keywords": list of 2-4 search terms\n'
    '- "file_filter": file extension to filter by (e.g. "pdf", "txt") or null\n'
    '- "source_hint": filename substring to filter by, or null\n\n'
    "JSON:\n"
)

MAP_PROMPT = (
    "Read this text and answer: does it contain information relevant to the question?\n\n"
    "Question: {query}\n"
    "Text: {chunk_text}\n\n"
    "Output a JSON object with:\n"
    '- "relevant": true or false\n'
    '- "facts": list of specific facts found (dates, names, numbers, filenames). '
    "Empty list if not relevant.\n\n"
    "JSON:\n"
)

REDUCE_PROMPT = (
    "Here are facts extracted from the user's files:\n"
    "{facts_list}\n\n"
    "Question: {query}\n\n"
    "Using ONLY these facts, write a concise answer. "
    'If the facts don\'t answer the question, say "No relevant information found."\n\n'
    "Answer:\n"
)


def chat_loop(index_name: str):
    print("\nLoading LFM2.5-1.2B-Instruct...")
    t0 = time.time()

    index_path = find_index_path(index_name)
    print(f"Using index: {index_path}")

    chat = LeannChat(
        index_path,
        llm_config={
            "type": "hf",
            "model": "LiquidAI/LFM2.5-1.2B-Instruct",
        },
    )
    print(f"Ready in {time.time() - t0:.1f}s")
    print("=" * 50)
    print("Ask anything about your files. Type 'quit' to exit.")
    print("Type 'debug' to toggle source display.")
    print("=" * 50)

    show_sources = DEBUG_SOURCES

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
            show_sources = not show_sources
            print(f"Source display: {'ON' if show_sources else 'OFF'}")
            continue

        t0 = time.time()

        # Search for relevant chunks
        results = chat.searcher.search(query, top_k=TOP_K)
        search_time = time.time() - t0

        if show_sources:
            print(f"\n--- SOURCES ({len(results)} chunks, {search_time:.2f}s) ---")
            for i, r in enumerate(results):
                source = r.metadata.get("source", "?")
                print(f"  [{i+1}] score={r.score:.3f}  {source}")
                print(f"       {r.text[:150].replace(chr(10), ' ')}...")
            print("---")

        # Build prompt with strict RAG instructions
        context = "\n\n".join([r.text for r in results])
        prompt = RAG_PROMPT.format(context=context, question=query)

        # Ask the LLM with low temperature for factual grounding
        response = chat.llm.ask(prompt, temperature=0.1)
        elapsed = time.time() - t0

        print(f"\n{response}")
        print(f"\n({elapsed:.1f}s)")


def main():
    # Parse args
    args = sys.argv[1:]

    # --reuse mode: skip indexing, use existing index
    if "--reuse" in args:
        idx = args.index("--reuse")
        if idx + 1 >= len(args):
            print("Error: --reuse requires an index name")
            sys.exit(1)
        index_name = args[idx + 1]
        chat_loop(index_name)
        return

    # --force flag
    force = "--force" in args
    args = [a for a in args if a != "--force"]

    # Directory to index
    if args:
        data_dir = Path(args[0]).resolve()
    else:
        data_dir = Path("./test_data").resolve()

    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a directory")
        sys.exit(1)

    index_name = build_index(data_dir, force=force)
    chat_loop(index_name)


if __name__ == "__main__":
    main()
