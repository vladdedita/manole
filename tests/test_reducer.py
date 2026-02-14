"""Tests for Reducer — answer synthesis and confidence checking."""
from reducer import Reducer, REDUCE_PROMPT, confidence_score


class FakeModelManager:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts = []

    def synthesize(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else ""


def test_reduce_prompt_contains_placeholders():
    assert "{facts_list}" in REDUCE_PROMPT
    assert "{query}" in REDUCE_PROMPT


def test_synthesize_from_facts():
    models = FakeModelManager(["Found 2 invoices: #123 for $50 and #456 for $75."])
    reducer = Reducer(models)
    relevant = [
        {"relevant": True, "facts": ["Invoice #123", "Amount: $50"], "source": "a.pdf"},
        {"relevant": True, "facts": ["Invoice #456", "Amount: $75"], "source": "b.pdf"},
    ]
    answer = reducer.synthesize("find invoices", relevant)
    assert "2 invoices" in answer


def test_synthesize_no_relevant_returns_message():
    models = FakeModelManager([])
    reducer = Reducer(models)
    answer = reducer.synthesize("find invoices", [])
    assert "No relevant" in answer


def test_synthesize_prompt_includes_sources():
    models = FakeModelManager(["answer"])
    reducer = Reducer(models)
    relevant = [
        {"relevant": True, "facts": ["fact1"], "source": "report.pdf"},
    ]
    reducer.synthesize("query", relevant)
    assert "report.pdf" in models.prompts[0]


def test_confidence_score_high_overlap():
    facts = ["Invoice #123", "Amount: $50", "Date: Dec 4"]
    answer = "Invoice #123 for $50 dated Dec 4"
    score = confidence_score(answer, facts)
    assert score >= 0.5


def test_confidence_score_low_overlap():
    facts = ["Invoice #123", "Amount: $50"]
    answer = "The weather is nice today and I like cats"
    score = confidence_score(answer, facts)
    assert score < 0.3


def test_confidence_score_empty_facts():
    assert confidence_score("some answer", []) == 0.0


def test_confidence_score_empty_answer():
    assert confidence_score("", ["fact1"]) == 0.0


def test_confidence_check_passes_high():
    models = FakeModelManager([])
    reducer = Reducer(models)
    relevant = [{"facts": ["Invoice #123", "Amount: $50"]}]
    answer = "Invoice #123 for $50"
    result = reducer.confidence_check(answer, relevant)
    assert "(low confidence)" not in result.lower()


def test_confidence_check_flags_low():
    models = FakeModelManager([])
    reducer = Reducer(models)
    relevant = [{"facts": ["Invoice #123", "Amount: $50"]}]
    answer = "The sky is blue and birds fly south"
    result = reducer.confidence_check(answer, relevant)
    assert "(low confidence)" in result.lower()


def test_format_filesystem_answer():
    models = FakeModelManager(["You have 5 PDF files in your collection."])
    reducer = Reducer(models)
    result = reducer.format_filesystem_answer("how many PDFs?", "Found 5 .pdf files.")
    assert "5" in result
