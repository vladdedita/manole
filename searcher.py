"""Stage 2: Searcher — vector search with metadata filters via LeannSearcher."""


class Searcher:
    """Wraps LeannSearcher with plan-based metadata filter construction."""

    def __init__(self, leann_searcher):
        self.leann = leann_searcher

    def _build_filters(self, plan: dict) -> dict | None:
        source_hint = plan.get("source_hint")
        file_filter = plan.get("file_filter")
        if source_hint:
            return {"source": {"contains": source_hint}}
        if file_filter:
            return {"source": {"contains": f".{file_filter}"}}
        return None

    def search(self, plan: dict, top_k: int = 5, file_filter_paths: list[str] | None = None) -> list:
        query = " ".join(plan.get("keywords", []))
        if not query:
            query = "document"
        metadata_filters = self._build_filters(plan)
        results = self.leann.search(query, top_k=top_k, metadata_filters=metadata_filters)
        if file_filter_paths:
            path_basenames = {p.rsplit("/", 1)[-1].lower() for p in file_filter_paths}
            results = [
                r for r in results
                if r.metadata.get("source", "").rsplit("/", 1)[-1].lower() in path_basenames
            ]
        return results

    def search_unfiltered(self, plan: dict, top_k: int = 5) -> list:
        query = " ".join(plan.get("keywords", []))
        if not query:
            query = "document"
        return self.leann.search(query, top_k=top_k)
