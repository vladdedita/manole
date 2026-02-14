"""
Prototype: LeannChat RAG feature exploration.

Uses the metadata-enriched index from prototype.py and tests the
RAG (retrieval + generation) pipeline with a local HF model.
"""

import time
from pathlib import Path
from leann import LeannChat


METADATA_INDEX = str(Path("./indexes/with_metadata.leann"))

TEST_QUERIES = [
    "Where is the documentation about Kubernetes?",
    "Find files with financial data",
    "What are the key topics in machine learning research?",
    "Where did I save the AI agent project ideas?",
    "Summarize notes about the Manole project",
    "Which documents mention client names or email addresses?",
]


def test_leann_chat():
    print("=" * 60)
    print("LeannChat RAG Test")
    print("=" * 60)

    model_name = "LiquidAI/LFM2.5-1.2B-Instruct"
    print(f"\nInitializing LeannChat with {model_name} (local HF model)...")
    print("(First run will download the model)\n")

    t0 = time.time()
    chat = LeannChat(
        METADATA_INDEX,
        llm_config={
            "type": "hf",
            "model": model_name,
        },
    )
    print(f"Chat initialized in {time.time() - t0:.2f}s\n")

    for query in TEST_QUERIES:
        print("-" * 60)
        print(f"Q: {query}")
        print("-" * 60)

        t0 = time.time()
        response = chat.ask(query, top_k=3)
        elapsed = time.time() - t0

        print(f"A: {response}")
        print(f"({elapsed:.2f}s)\n")


if __name__ == "__main__":
    test_leann_chat()
