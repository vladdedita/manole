"""Stage 5: Reducer — answer synthesis and confidence checking."""

REDUCE_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using ONLY the provided facts. "
    "State the answer directly. Use every relevant fact in your answer. "
    "Be concise."
)

FILESYSTEM_SYSTEM = (
    "You are a helpful assistant. Answer the user's question based on the filesystem data provided. "
    "Be brief and natural."
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

    def synthesize(self, query: str, relevant: list[dict], context: str = "") -> str:
        if not relevant:
            return "No relevant information found."
        facts_list = ""
        for item in relevant:
            source = item.get("source", "?")
            facts_list += f"\nFrom {source}:\n"
            for fact in item["facts"]:
                facts_list += f"  - {fact}\n"
        # Question first, then facts — model sees what to answer before the data
        user_msg = f"Question: {query}\n"
        if context:
            user_msg += f"{context}\n"
        user_msg += f"\nFacts:\n{facts_list}"
        if self.debug:
            print(f"  [REDUCE] Synthesizing from {len(relevant)} sources")
            print(f"  [REDUCE] User msg:\n{user_msg}")
        answer = self.models.synthesize(REDUCE_SYSTEM, user_msg)
        if self.debug:
            print(f"  [REDUCE] Raw answer: {answer}")
        answer = answer.strip()
        # Small models produce hedging preambles like "X is not explicitly stated. However, ..."
        # Strip the preamble and keep the actual answer after "However,"
        for hedge in ["However, ", "That said, ", "Nevertheless, "]:
            idx = answer.find(hedge)
            if idx > 0 and idx < len(answer) // 2:
                answer = answer[idx + len(hedge):]
                # Capitalize first letter
                answer = answer[0].upper() + answer[1:] if answer else answer
                if self.debug:
                    print(f"  [REDUCE] Stripped hedging preamble at '{hedge.strip()}'")
                break
        # Small models sometimes append "No relevant information found" after a valid answer.
        # Strip the trailing fallback if real content precedes it.
        fallback = "No relevant information found."
        if answer.endswith(fallback) and len(answer) > len(fallback) + 5:
            answer = answer[: -len(fallback)].strip()
            if self.debug:
                print("  [REDUCE] Stripped trailing fallback phrase")
        # Strip trailing "Answer: ..." summaries that restate in vague terms
        ans_idx = answer.rfind("\nAnswer:")
        if ans_idx > 0:
            answer = answer[:ans_idx].strip()
            if self.debug:
                print("  [REDUCE] Stripped trailing 'Answer:' summary")
        return answer

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
        user_msg = f"The user asked: {query}\n\nFilesystem result:\n{result}"
        answer = self.models.synthesize(FILESYSTEM_SYSTEM, user_msg)
        return answer.strip()
