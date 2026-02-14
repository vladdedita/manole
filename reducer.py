"""Stage 5: Reducer — answer synthesis and confidence checking."""

REDUCE_PROMPT = (
    "Here are facts extracted from the user's files:\n"
    "{facts_list}\n\n"
    "Question: {query}\n\n"
    "Using ONLY these facts, write a concise answer. "
    'If the facts don\'t answer the question, say "No relevant information found."\n\n'
    "Answer:\n"
)

FILESYSTEM_PROMPT = (
    "The user asked: {query}\n\n"
    "Here is the result from their filesystem:\n"
    "{result}\n\n"
    "Write a brief, natural answer based on this data.\n\n"
    "Answer:\n"
)


def confidence_score(answer: str, facts: list[str]) -> float:
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


class Reducer:
    """Synthesizes answers from extracted facts using 1.2B-RAG model."""

    def __init__(self, models, debug: bool = False):
        self.models = models
        self.debug = debug

    def synthesize(self, query: str, relevant: list[dict]) -> str:
        if not relevant:
            return "No relevant information found."
        facts_list = ""
        for item in relevant:
            source = item.get("source", "?")
            facts_list += f"\nFrom {source}:\n"
            for fact in item["facts"]:
                facts_list += f"  - {fact}\n"
        prompt = REDUCE_PROMPT.format(facts_list=facts_list, query=query)
        if self.debug:
            print(f"  [REDUCE] Synthesizing from {len(relevant)} sources")
        answer = self.models.synthesize(prompt)
        return answer.strip()

    def confidence_check(self, answer: str, relevant: list[dict]) -> str:
        all_facts = []
        for item in relevant:
            all_facts.extend(item.get("facts", []))
        score = confidence_score(answer, all_facts)
        if self.debug:
            print(f"  [CHECK] Confidence: {score:.2f}")
        if score < 0.2:
            if self.debug:
                print("  [CHECK] Low confidence — answer may not be grounded")
            return f"{answer}\n\n(Low confidence) Answer may not reflect source documents."
        return answer

    def format_filesystem_answer(self, query: str, result: str) -> str:
        prompt = FILESYSTEM_PROMPT.format(query=query, result=result)
        answer = self.models.synthesize(prompt)
        return answer.strip()
