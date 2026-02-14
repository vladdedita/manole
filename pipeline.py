"""AgenticRAG: dual-model agentic RAG pipeline orchestrator."""
import time

from planner import Planner
from searcher import Searcher
from mapper import Mapper
from reducer import Reducer
from toolbox import ToolBox


class AgenticRAG:
    """Orchestrates the full Plan → Search → Map → Filter → Reduce pipeline."""

    def __init__(
        self,
        models,
        leann_searcher,
        data_dir: str,
        top_k: int = 5,
        debug: bool = False,
        history_window: int = 3,
    ):
        self.planner = Planner(models, debug=debug)
        self.searcher = Searcher(leann_searcher)
        self.mapper = Mapper(models, debug=debug)
        self.reducer = Reducer(models, debug=debug)
        self.toolbox = ToolBox(data_dir)
        self.top_k = top_k
        self.debug = debug
        self.history: list[tuple[str, str]] = []
        self.history_window = history_window

    def _build_context(self) -> str:
        if not self.history:
            return ""
        lines = ["Recent conversation:"]
        for q, a in self.history:
            lines.append(f"  User: {q}")
            lines.append(f"  Assistant: {a[:200]}")
        return "\n".join(lines)

    def _log(self, msg: str):
        if self.debug:
            print(f"  [PIPELINE] {msg}")

    def ask(self, query: str) -> str:
        """Run the full agentic RAG pipeline."""
        t0 = time.time()
        context = self._build_context()

        # Stage 1: Plan
        plan = self.planner.plan(query, context=context)
        self._log(f"Plan: tool={plan['tool']}")

        # Filesystem path: ToolBox → format → return
        if plan["tool"] == "filesystem":
            result = self.toolbox.execute(plan)
            answer = self.reducer.format_filesystem_answer(query, result)
            self._log(f"Filesystem answer in {time.time() - t0:.1f}s")
            self.history.append((query, answer))
            self.history = self.history[-self.history_window:]
            return answer

        # Hybrid path: ToolBox narrows file scope → search within those
        if plan["tool"] == "hybrid":
            file_paths = self.toolbox.get_matching_files(plan)
            self._log(f"Hybrid: {len(file_paths)} files from ToolBox")
            chunks = self.searcher.search(plan, top_k=self.top_k, file_filter_paths=file_paths)
        else:
            # Semantic search path
            chunks = self.searcher.search(plan, top_k=self.top_k)

        # Fallback: retry without filters
        if not chunks:
            self._log("No results with filters, retrying unfiltered")
            chunks = self.searcher.search_unfiltered(plan, top_k=self.top_k)

        if not chunks:
            answer = "No relevant information found in your files."
            self.history.append((query, answer))
            self.history = self.history[-self.history_window:]
            return answer

        # Stage 3: Map
        mapped = self.mapper.extract_facts(query, chunks)

        # Stage 4: Filter
        relevant = [m for m in mapped if m["relevant"]]
        self._log(f"Filter: {len(relevant)}/{len(mapped)} chunks relevant")

        if not relevant:
            answer = "No relevant information found in your files."
            self.history.append((query, answer))
            self.history = self.history[-self.history_window:]
            return answer

        # Stage 5: Reduce
        answer = self.reducer.synthesize(query, relevant, context=context)

        # Stage 6: Confidence check
        answer = self.reducer.confidence_check(answer, relevant)

        self._log(f"Pipeline completed in {time.time() - t0:.1f}s")
        self.history.append((query, answer))
        self.history = self.history[-self.history_window:]
        return answer
