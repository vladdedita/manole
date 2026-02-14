# Manole - Project Overview

## Purpose
Offline AI-powered file system search and RAG (Retrieval Augmented Generation) application. Indexes local directories and enables semantic search + conversational Q&A over personal files using local LLM models.

## Tech Stack
- Python 3.13
- uv (package manager)
- leann (vector indexing, search, RAG via LeannBuilder/LeannSearcher/LeannChat)
- LiquidAI LFM2.5-1.2B-Instruct (local HuggingFace LLM)
- HNSW backend for vector search
- pytest for testing

## Key Files
- `chat.py` — Main interactive chat application with agentic RAG pipeline (AgenticRAG class)
- `prototype.py` — Early LEANN exploration script (content-only vs metadata-enriched indexing)
- `preprocess.py` — LFM2.5-VL vision model preprocessing for images and scanned PDFs
- `main.py` — Placeholder entry point
- `tests/test_agentic_rag.py` — Tests for the agentic RAG pipeline
- `pyproject.toml` — Project config and dependencies

## Architecture
The AgenticRAG class in chat.py implements a multi-stage pipeline:
1. Planner (LLM) — extracts search keywords and metadata filters from query
2. Searcher — LEANN vector search with optional metadata filters
3. Map (LLM per chunk) — extracts structured facts from each retrieved chunk
4. Filter — drops irrelevant chunks (Python logic)
5. Reduce (LLM) — synthesizes final answer from extracted facts
6. Confidence Check — Python token overlap check (no LLM call)
