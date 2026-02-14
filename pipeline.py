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
    ):
        self.planner = Planner(models, debug=debug)
        self.searcher = Searcher(leann_searcher)
        self.mapper = Mapper(models, debug=debug)
        self.reducer = Reducer(models, debug=debug)
        self.toolbox = ToolBox(data_dir)
        self.top_k = top_k
        self.debug = debug

    def _log(self, msg: str):
        if self.debug:
            print(f"  [PIPELINE] {msg}")

    def ask(self, query: str) -> str:
        """Run the full agentic RAG pipeline."""
        t0 = time.time()

        # Stage 1: Plan
        plan = self.planner.plan(query)
        self._log(f"Plan: tool={plan['tool']}")

        # Filesystem path: ToolBox → format → return
        if plan["tool"] == "filesystem":
            result = self.toolbox.execute(plan)
            answer = self.reducer.format_filesystem_answer(query, result)
            self._log(f"Filesystem answer in {time.time() - t0:.1f}s")
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
            return "No relevant information found in your files."

        # Stage 3: Map
        mapped = self.mapper.extract_facts(query, chunks)

        # Stage 4: Filter
        relevant = [m for m in mapped if m["relevant"]]
        self._log(f"Filter: {len(relevant)}/{len(mapped)} chunks relevant")

        if not relevant:
            return "No relevant information found in your files."

        # Stage 5: Reduce
        answer = self.reducer.synthesize(query, relevant)

        # Stage 6: Confidence check
        answer = self.reducer.confidence_check(answer, relevant)

        self._log(f"Pipeline completed in {time.time() - t0:.1f}s")
        return answer
