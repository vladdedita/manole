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


def confidence_score(answer: str, facts: list[str]) -> float:
    """Compute token overlap between answer and source facts. Returns 0.0-1.0."""
    if not facts:
        return 0.0
    answer_tokens = set(answer.lower().split())
    if not answer_tokens:
        return 0.0
    fact_tokens = set()
    for fact in facts:
        fact_tokens.update(fact.lower().split())
    if not fact_tokens:
        return 0.0
    overlap = answer_tokens & fact_tokens
    return len(overlap) / len(answer_tokens)


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


class AgenticRAG:
    """Multi-stage agentic RAG pipeline for small language models."""

    def __init__(self, searcher, llm, top_k=5, debug=True):
        self.searcher = searcher
        self.llm = llm
        self.top_k = top_k
        self.debug = debug

    def _log(self, stage: str, msg: str):
        if self.debug:
            print(f"  [{stage}] {msg}")

    def _plan(self, query: str) -> dict:
        """Stage 1: Extract search parameters from query."""
        self._log("PLAN", f"Analyzing query: {query}")
        response = self.llm.ask(PLANNER_PROMPT.format(query=query), temperature=0.0)
        parsed = parse_json(response)

        default = {"keywords": [], "file_filter": None, "source_hint": None}
        if parsed is None:
            self._log("PLAN", "Failed to parse planner output, using defaults")
            return default

        result = {
            "keywords": parsed.get("keywords", []),
            "file_filter": parsed.get("file_filter"),
            "source_hint": parsed.get("source_hint"),
        }
        self._log("PLAN", f"Result: {result}")
        return result

    def _search(self, query: str, plan: dict) -> list:
        """Stage 2: Search with optional metadata filters."""
        metadata_filters = None
        source_hint = plan.get("source_hint")
        file_filter = plan.get("file_filter")

        if source_hint or file_filter:
            metadata_filters = {}
            if source_hint:
                metadata_filters["source"] = {"contains": source_hint}
            elif file_filter:
                metadata_filters["source"] = {"contains": f".{file_filter}"}

        self._log("SEARCH", f"Filters: {metadata_filters}")
        results = self.searcher.search(
            query, top_k=self.top_k, metadata_filters=metadata_filters
        )
        self._log("SEARCH", f"Found {len(results)} chunks")
        return results



    def _map_chunk(self, query: str, chunk) -> dict:
        """Stage 3: Extract facts from a single chunk."""
        prompt = MAP_PROMPT.format(query=query, chunk_text=chunk.text[:500])
        response = self.llm.ask(prompt, temperature=0.0)
        parsed = parse_json(response)

        source = chunk.metadata.get("source", chunk.id)

        if parsed is None:
            self._log("MAP", f"  {source}: parse failed, treating as relevant")
            return {"relevant": True, "facts": [chunk.text[:200]], "source": source}

        result = {
            "relevant": parsed.get("relevant", True),
            "facts": parsed.get("facts", []),
            "source": source,
        }
        self._log("MAP", f"  {source}: relevant={result['relevant']}, facts={len(result['facts'])}")
        return result

    def _filter(self, mapped: list[dict]) -> list[dict]:
        """Stage 4: Drop irrelevant chunks."""
        relevant = [m for m in mapped if m["relevant"]]
        self._log("FILTER", f"Kept {len(relevant)}/{len(mapped)} chunks")
        return relevant

    def _reduce(self, query: str, relevant: list[dict]) -> str:
        """Stage 5: Synthesize answer from extracted facts."""
        if not relevant:
            return "No relevant information found."

        facts_list = ""
        for item in relevant:
            source = item.get("source", "?")
            facts_list += f"\nFrom {source}:\n"
            for fact in item["facts"]:
                facts_list += f"  - {fact}\n"

        prompt = REDUCE_PROMPT.format(facts_list=facts_list, query=query)
        self._log("REDUCE", f"Synthesizing from {len(relevant)} sources")
        answer = self.llm.ask(prompt, temperature=0.1)
        return answer.strip()

    def _confidence_check(self, answer: str, relevant: list[dict]) -> str:
        """Stage 6: Python-based confidence check via token overlap. No LLM call."""
        all_facts = []
        for item in relevant:
            all_facts.extend(item.get("facts", []))

        score = confidence_score(answer, all_facts)
        self._log("CHECK", f"Confidence: {score:.2f}")

        if score < 0.2:
            self._log("CHECK", "Low confidence — answer may not be grounded in sources")
            return f"{answer}\n\n(Low confidence) Answer may not reflect source documents."

        return answer

    def ask(self, query: str) -> str:
        """Run the full agentic RAG pipeline."""
        t0 = time.time()

        # Stage 1: Plan
        plan = self._plan(query)

        # Stage 2: Search
        results = self._search(query, plan)
        if not results:
            # Retry without filters
            self._log("SEARCH", "No results with filters, retrying without")
            results = self.searcher.search(query, top_k=self.top_k)

        if not results:
            return "No relevant information found."

        # Stage 3: Map
        self._log("MAP", f"Processing {len(results)} chunks...")
        mapped = [self._map_chunk(query, chunk) for chunk in results]

        # Stage 4: Filter
        relevant = self._filter(mapped)

        # Stage 5: Reduce
        answer = self._reduce(query, relevant)

        # Stage 6: Confidence check (Python token overlap, no LLM call)
        answer = self._confidence_check(answer, relevant)

        elapsed = time.time() - t0
        self._log("DONE", f"Pipeline completed in {elapsed:.1f}s")
        return answer
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

    rag = AgenticRAG(chat.searcher, chat.llm, top_k=TOP_K, debug=DEBUG_SOURCES)

    print(f"Ready in {time.time() - t0:.1f}s")
    print("=" * 50)
    print("Ask anything about your files. Type 'quit' to exit.")
    print("Type 'debug' to toggle pipeline trace.")
    print("=" * 50)

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
            rag.debug = not rag.debug
            print(f"Pipeline trace: {'ON' if rag.debug else 'OFF'}")
            continue

        t0 = time.time()
        response = rag.ask(query)
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
