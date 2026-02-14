"""AgenticRAG: dual-model agentic RAG pipeline orchestrator."""
import time

from rewriter import QueryRewriter
from planner import Planner
from searcher import Searcher
from mapper import Mapper
from reducer import Reducer
from toolbox import ToolBox


class AgenticRAG:
    """Orchestrates the full Rewrite → Plan → Search → Map → Filter → Reduce pipeline."""

    def __init__(
        self,
        models,
        leann_searcher,
        data_dir: str,
        top_k: int = 10,
        debug: bool = False,
        history_window: int = 3,
    ):
        self.rewriter = QueryRewriter(models, debug=debug)
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

        # Stage 0: Rewrite
        rewrite = self.rewriter.rewrite(query, context=context)
        resolved_query = rewrite["resolved_query"]
        search_query = rewrite["search_query"]
        self._log(f"Rewrite: intent={rewrite['intent']}, search_query={search_query}")

        # Stage 1: Plan (uses resolved query, not raw)
        plan = self.planner.plan(resolved_query, context=context)
        self._log(f"Plan: tool={plan['tool']}")

        # Filesystem path: ToolBox → format → return
        if plan["tool"] == "filesystem":
            result = self.toolbox.execute(plan)
            answer = self.reducer.format_filesystem_answer(resolved_query, result)
            self._log(f"Filesystem answer in {time.time() - t0:.1f}s")
            self.history.append((query, answer))
            self.history = self.history[-self.history_window:]
            return answer

        # Semantic search path (uses rewriter's search_query)
        chunks = self.searcher.search(plan, top_k=self.top_k, search_query=search_query)

        # Fallback: retry without filters
        if not chunks:
            self._log("No results with filters, retrying unfiltered")
            chunks = self.searcher.search_unfiltered(plan, top_k=self.top_k, search_query=search_query)

        if not chunks:
            answer = "No relevant information found in your files."
            self.history.append((query, answer))
            self.history = self.history[-self.history_window:]
            return answer

        # Pre-filter: drop chunks with low vector similarity scores
        if len(chunks) > 1:
            top_score = chunks[0].score
            threshold = top_score * 0.8
            before = len(chunks)
            chunks = [c for c in chunks if c.score >= threshold]
            if self.debug and len(chunks) < before:
                self._log(f"Score filter: {len(chunks)}/{before} chunks above {threshold:.2f} (top={top_score:.2f})")

        # Stage 3: Map (uses resolved query for relevance check)
        mapped = self.mapper.extract_facts(resolved_query, chunks)

        # Stage 4: Filter
        relevant = [m for m in mapped if m["relevant"]]
        self._log(f"Filter: {len(relevant)}/{len(mapped)} chunks relevant")

        if not relevant:
            answer = "No relevant information found in your files."
            self.history.append((query, answer))
            self.history = self.history[-self.history_window:]
            return answer

        # Stage 5: Reduce
        answer = self.reducer.synthesize(resolved_query, relevant, context=context)

        # Stage 6: Confidence check
        answer = self.reducer.confidence_check(answer, relevant)

        self._log(f"Pipeline completed in {time.time() - t0:.1f}s")
        self.history.append((query, answer))
        self.history = self.history[-self.history_window:]
        return answer
