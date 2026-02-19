"""Tests for file graph computation."""
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


def make_passage(passage_id, text, file_path, file_name=None):
    """Helper to create a passage dict matching leann JSONL format."""
    meta = {"file_path": file_path}
    if file_name:
        meta["file_name"] = file_name
    return {"id": passage_id, "text": text, "metadata": meta}


class TestBuildNodes:
    """Test node construction from passages."""

    def test_groups_passages_by_file(self):
        from graph import build_nodes
        passages = [
            make_passage("0", "hello world", "/data/doc.pdf"),
            make_passage("1", "second chunk", "/data/doc.pdf"),
            make_passage("2", "other file", "/data/notes.md"),
        ]
        nodes = build_nodes(passages, base_dir="/data")
        assert len(nodes) == 2
        by_id = {n["id"]: n for n in nodes}
        assert by_id["doc.pdf"]["passageCount"] == 2
        assert by_id["notes.md"]["passageCount"] == 1

    def test_extracts_file_metadata(self):
        from graph import build_nodes
        passages = [
            make_passage("0", "content", "/data/sub/report.pdf"),
        ]
        nodes = build_nodes(passages, base_dir="/data")
        assert len(nodes) == 1
        node = nodes[0]
        assert node["id"] == "sub/report.pdf"
        assert node["name"] == "report.pdf"
        assert node["type"] == "pdf"
        assert node["dir"] == "sub"

    def test_handles_missing_metadata(self):
        from graph import build_nodes
        passages = [{"id": "0", "text": "hello", "metadata": {}}]
        nodes = build_nodes(passages, base_dir="/data")
        assert len(nodes) == 0


class TestSimilarityEdges:
    """Test content similarity edge computation."""

    def test_returns_edges_above_threshold(self):
        from graph import compute_similarity_edges
        embeddings = {
            "a.pdf": np.array([1.0, 0.0, 0.0]),
            "b.pdf": np.array([0.9, 0.1, 0.0]),
            "c.pdf": np.array([0.0, 0.0, 1.0]),
        }
        edges = compute_similarity_edges(embeddings, top_k=2, threshold=0.5)
        sources_targets = {(e["source"], e["target"]) for e in edges}
        assert ("a.pdf", "b.pdf") in sources_targets or ("b.pdf", "a.pdf") in sources_targets
        assert all(e["type"] == "similarity" for e in edges)
        assert all(0 <= e["weight"] <= 1 for e in edges)

    def test_respects_threshold(self):
        from graph import compute_similarity_edges
        embeddings = {
            "a.pdf": np.array([1.0, 0.0]),
            "b.pdf": np.array([0.0, 1.0]),
        }
        edges = compute_similarity_edges(embeddings, top_k=5, threshold=0.5)
        assert len(edges) == 0

    def test_no_self_edges(self):
        from graph import compute_similarity_edges
        embeddings = {
            "a.pdf": np.array([1.0, 0.0]),
            "b.pdf": np.array([0.9, 0.1]),
        }
        edges = compute_similarity_edges(embeddings, top_k=5, threshold=0.0)
        for e in edges:
            assert e["source"] != e["target"]


class TestExtractEntities:
    """Test entity extraction from text."""

    def test_extracts_emails(self):
        from graph import extract_entities
        entities = extract_entities("Contact us at billing@anthropic.com or support@openai.com")
        assert "email" in entities
        assert "billing@anthropic.com" in entities["email"]
        assert "support@openai.com" in entities["email"]

    def test_extracts_tax_ids(self):
        from graph import extract_entities
        entities = extract_entities("CUI 48862329 registered in Romania. VAT RO48862329.")
        assert "tax_id" in entities
        assert "48862329" in entities["tax_id"]

    def test_extracts_iban(self):
        from graph import extract_entities
        entities = extract_entities("Pay to IBAN: RO49AAAA1B31007593840000")
        assert "iban" in entities
        assert "RO49AAAA1B31007593840000" in entities["iban"]

    def test_extracts_company_names(self):
        from graph import extract_entities
        entities = extract_entities("Invoice from Acme International Corp for services")
        assert "company" in entities
        assert "ACME INTERNATIONAL" in entities["company"]

    def test_empty_text_returns_empty(self):
        from graph import extract_entities
        entities = extract_entities("")
        assert entities == {}

    def test_normalizes_emails_lowercase(self):
        from graph import extract_entities
        entities = extract_entities("Email: Admin@Company.COM")
        assert "admin@company.com" in entities["email"]

    def test_normalizes_company_names_uppercase(self):
        from graph import extract_entities
        entities = extract_entities("Billed by Cloud Services LLC")
        assert "CLOUD SERVICES" in entities["company"]


class TestReferenceEdges:
    """Test explicit reference detection."""

    def test_detects_filename_mentions(self):
        from graph import compute_reference_edges
        passages_by_file = {
            "readme.md": ["See report.pdf for details"],
            "report.pdf": ["This is the report"],
        }
        file_ids = {"readme.md", "report.pdf"}
        edges = compute_reference_edges(passages_by_file, file_ids)
        assert len(edges) >= 1
        ref_edges = [e for e in edges if "mentions" in e.get("label", "")]
        assert len(ref_edges) >= 1
        assert ref_edges[0]["type"] == "reference"

    def test_no_self_references(self):
        from graph import compute_reference_edges
        passages_by_file = {
            "readme.md": ["This readme.md explains things"],
        }
        file_ids = {"readme.md"}
        edges = compute_reference_edges(passages_by_file, file_ids)
        assert len(edges) == 0

    def test_detects_shared_email(self):
        from graph import compute_reference_edges
        passages_by_file = {
            "invoice1.pdf": ["Billed to client@example.com"],
            "invoice2.pdf": ["Payment from client@example.com"],
            "unrelated.pdf": ["No entities here"],
        }
        file_ids = set(passages_by_file.keys())
        edges = compute_reference_edges(passages_by_file, file_ids)
        pairs = {(e["source"], e["target"]) for e in edges}
        assert ("invoice1.pdf", "invoice2.pdf") in pairs
        # unrelated should not connect
        for e in edges:
            assert "unrelated.pdf" not in (e["source"], e["target"])

    def test_detects_shared_tax_id(self):
        from graph import compute_reference_edges
        passages_by_file = {
            "receipt.pdf": ["CUI 48862329 company details"],
            "statement.pdf": ["Tax ID CUI 48862329 on record"],
        }
        file_ids = set(passages_by_file.keys())
        edges = compute_reference_edges(passages_by_file, file_ids)
        assert len(edges) >= 1
        assert edges[0]["type"] == "reference"
        assert "tax_id" in edges[0]["label"]

    def test_detects_shared_company_name(self):
        from graph import compute_reference_edges
        passages_by_file = {
            "inv_a.pdf": ["Invoice from Cloud Services LLC"],
            "inv_b.pdf": ["Payment to Cloud Services LLC"],
        }
        file_ids = set(passages_by_file.keys())
        edges = compute_reference_edges(passages_by_file, file_ids)
        assert len(edges) >= 1
        assert "company" in edges[0]["label"]

    def test_deduplicates_edges_keeps_highest_weight(self):
        from graph import compute_reference_edges, ENTITY_WEIGHTS
        passages_by_file = {
            "a.pdf": ["Contact: info@acme.com CUI 12345678 Acme Corp SRL"],
            "b.pdf": ["From: info@acme.com CUI 12345678 Acme Corp SRL"],
        }
        file_ids = set(passages_by_file.keys())
        edges = compute_reference_edges(passages_by_file, file_ids)
        # Should produce exactly one edge per file pair
        assert len(edges) == 1
        # Email has highest weight (1.0), should win
        assert edges[0]["weight"] == ENTITY_WEIGHTS["email"]

    def test_multiple_file_pairs_from_shared_entity(self):
        from graph import compute_reference_edges
        passages_by_file = {
            "a.pdf": ["CUI 99887766"],
            "b.pdf": ["CUI 99887766"],
            "c.pdf": ["CUI 99887766"],
        }
        file_ids = set(passages_by_file.keys())
        edges = compute_reference_edges(passages_by_file, file_ids)
        # 3 files sharing one entity -> 3 edges (a-b, a-c, b-c)
        assert len(edges) == 3


class TestStructureEdges:
    """Test directory hierarchy edges."""

    def test_parent_child_edges(self):
        from graph import compute_structure_edges
        file_ids = ["docs/a.pdf", "docs/b.pdf", "src/c.py"]
        edges, dir_nodes = compute_structure_edges(file_ids)
        assert len(edges) > 0
        assert all(e["type"] == "structure" for e in edges)
        # Should create directory nodes for "docs" and "src"
        dir_ids = {n["id"] for n in dir_nodes}
        assert "docs" in dir_ids
        assert "src" in dir_ids
        assert all(n["type"] == "dir" for n in dir_nodes)

    def test_root_files_get_virtual_root(self):
        from graph import compute_structure_edges
        file_ids = ["a.pdf", "b.pdf"]
        edges, dir_nodes = compute_structure_edges(file_ids)
        # Root-level files should get a virtual root node connecting them
        assert len(edges) == 2
        assert all(e["source"] == "." for e in edges)
        dir_ids = {n["id"] for n in dir_nodes}
        assert "." in dir_ids
