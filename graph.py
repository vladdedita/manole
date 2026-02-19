"""File graph computation from leann index data.

Reads passage data and embeddings from a leann index, groups by source file,
and computes three types of edges: content similarity, explicit references,
and directory structure.
"""
import json
import re
from pathlib import Path, PurePosixPath

import numpy as np


def build_nodes(passages: list[dict], base_dir: str) -> list[dict]:
    """Group passages by file_path and build node metadata."""
    base = Path(base_dir)
    by_file: dict[str, list[dict]] = {}

    for p in passages:
        meta = p.get("metadata", {})
        file_path = meta.get("file_path")
        if not file_path:
            continue
        try:
            rel = str(PurePosixPath(Path(file_path).relative_to(base)))
        except ValueError:
            rel = PurePosixPath(file_path).name
        by_file.setdefault(rel, []).append(p)

    nodes = []
    for rel_path, file_passages in by_file.items():
        pp = PurePosixPath(rel_path)
        nodes.append({
            "id": rel_path,
            "name": pp.name,
            "type": pp.suffix.lstrip(".").lower() if pp.suffix else "",
            "size": 0,
            "dir": str(pp.parent) if str(pp.parent) != "." else "",
            "passageCount": len(file_passages),
        })
    return nodes


def compute_similarity_edges(
    embeddings: dict[str, np.ndarray],
    top_k: int = 5,
    threshold: float = 0.6,
) -> list[dict]:
    """Compute content similarity edges between files using cosine similarity."""
    file_ids = list(embeddings.keys())
    n = len(file_ids)
    if n < 2:
        return []

    matrix = np.stack([embeddings[fid] for fid in file_ids])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = matrix / norms
    sim_matrix = normalized @ normalized.T

    edges = []
    seen = set()

    for i in range(n):
        scores = sim_matrix[i].copy()
        scores[i] = -1
        top_indices = np.argsort(scores)[::-1][:top_k]

        for j in top_indices:
            score = float(scores[j])
            if score < threshold:
                continue
            pair = tuple(sorted((file_ids[i], file_ids[j])))
            if pair in seen:
                continue
            seen.add(pair)
            edges.append({
                "source": file_ids[i],
                "target": file_ids[j],
                "type": "similarity",
                "weight": round(score, 3),
            })

    return edges


_EMAIL_RE = re.compile(r"\b[\w.-]+@[\w.-]+\.\w{2,}\b")
_TAX_ID_RE = re.compile(
    r"\b(?:CIF|VAT|TIN|CUI|RO|IE)\s*[\d]{6,12}\b", re.IGNORECASE
)
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b")
_COMPANY_SUFFIX_RE = re.compile(
    r"\b([A-Z][\w.-]+(?:\s+[A-Z][\w.-]+)+)"
    r"\s+(?:SRL|LLC|Ltd|GmbH|Inc|SA|SCA|PLC|Corp|Co)\b"
)

ENTITY_WEIGHTS = {
    "email": 1.0,
    "tax_id": 0.9,
    "iban": 0.8,
    "company": 0.7,
    "filename": 1.0,
}


def extract_entities(text: str) -> dict[str, set[str]]:
    """Extract structured entities from text using regex patterns.

    Returns dict mapping entity type to set of normalized entity values.
    """
    entities: dict[str, set[str]] = {
        "email": set(),
        "tax_id": set(),
        "iban": set(),
        "company": set(),
    }

    for m in _EMAIL_RE.finditer(text):
        entities["email"].add(m.group().lower())

    for m in _TAX_ID_RE.finditer(text):
        digits = re.sub(r"[^\d]", "", m.group())
        entities["tax_id"].add(digits)

    for m in _IBAN_RE.finditer(text):
        entities["iban"].add(m.group().upper())

    for m in _COMPANY_SUFFIX_RE.finditer(text):
        name = m.group(1).strip()
        if len(name) >= 4:
            entities["company"].add(name.upper())

    # Remove empty categories
    return {k: v for k, v in entities.items() if v}


def compute_reference_edges(
    passages_by_file: dict[str, list[str]],
    file_ids: set[str],
) -> list[dict]:
    """Detect references between files via shared entities and filename mentions."""
    # Phase 1: Extract entities per file
    entities_by_file: dict[str, dict[str, set[str]]] = {}
    for fid, texts in passages_by_file.items():
        combined = " ".join(texts)
        entities_by_file[fid] = extract_entities(combined)

    # Phase 2: Build inverted index — entity -> set of file IDs
    inverted: dict[tuple[str, str], set[str]] = {}
    for fid, ent_map in entities_by_file.items():
        for ent_type, values in ent_map.items():
            for val in values:
                key = (ent_type, val)
                inverted.setdefault(key, set()).add(fid)

    # Phase 3: Create edges from shared entities
    best_edges: dict[tuple[str, str], dict] = {}

    for (ent_type, ent_val), fids in inverted.items():
        if len(fids) < 2:
            continue
        weight = ENTITY_WEIGHTS.get(ent_type, 0.5)
        fid_list = sorted(fids)
        for i, src in enumerate(fid_list):
            for tgt in fid_list[i + 1:]:
                pair = (src, tgt)
                existing = best_edges.get(pair)
                if existing is None or weight > existing["weight"]:
                    best_edges[pair] = {
                        "source": src,
                        "target": tgt,
                        "type": "reference",
                        "weight": weight,
                        "label": f"shared {ent_type}: {ent_val}",
                    }

    # Phase 4: Filename-mention detection (original logic, as additional signal)
    name_to_ids: dict[str, set[str]] = {}
    for fid in file_ids:
        name = PurePosixPath(fid).name
        name_to_ids.setdefault(name, set()).add(fid)

    for source_id, texts in passages_by_file.items():
        combined = " ".join(texts)
        for name, target_ids in name_to_ids.items():
            if len(name) < 4:
                continue
            if re.search(re.escape(name), combined, re.IGNORECASE):
                for target_id in target_ids:
                    if target_id == source_id:
                        continue
                    pair = tuple(sorted((source_id, target_id)))
                    existing = best_edges.get(pair)
                    if existing is None or ENTITY_WEIGHTS["filename"] > existing["weight"]:
                        best_edges[pair] = {
                            "source": source_id,
                            "target": target_id,
                            "type": "reference",
                            "weight": ENTITY_WEIGHTS["filename"],
                            "label": f"mentions {name}",
                        }

    return list(best_edges.values())


def compute_structure_edges(file_ids: list[str]) -> tuple[list[dict], list[dict]]:
    """Compute directory hierarchy edges and directory nodes.

    Returns:
        Tuple of (edges, directory_nodes). Directory nodes are synthetic nodes
        representing folders, needed so the graph can display the tree structure.
    """
    # Collect all directory paths
    dirs = set()
    for fid in file_ids:
        parts = PurePosixPath(fid).parts
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))

    # Create synthetic directory nodes
    dir_nodes = []
    for d in sorted(dirs):
        pp = PurePosixPath(d)
        dir_nodes.append({
            "id": d,
            "name": pp.name + "/",
            "type": "dir",
            "size": 0,
            "dir": str(pp.parent) if str(pp.parent) != "." else "",
            "passageCount": 0,
        })

    edges = []
    seen = set()

    # File → parent directory edges
    for fid in file_ids:
        pp = PurePosixPath(fid)
        parent = str(pp.parent) if str(pp.parent) != "." else ""
        if parent and parent in dirs:
            pair = (parent, fid)
            if pair not in seen:
                seen.add(pair)
                edges.append({
                    "source": parent,
                    "target": fid,
                    "type": "structure",
                    "weight": 1.0,
                    "label": "contains",
                })

    # Directory → parent directory edges
    for d in dirs:
        pp = PurePosixPath(d)
        parent = str(pp.parent) if str(pp.parent) != "." else ""
        if parent and parent in dirs:
            pair = (parent, d)
            if pair not in seen:
                seen.add(pair)
                edges.append({
                    "source": parent,
                    "target": d,
                    "type": "structure",
                    "weight": 1.0,
                    "label": "contains",
                })

    # If no subdirectories exist (all files at root), connect all files to a
    # virtual root so the structure tab still shows something
    if not edges and len(file_ids) > 1:
        dir_nodes.append({
            "id": ".",
            "name": "/",
            "type": "dir",
            "size": 0,
            "dir": "",
            "passageCount": 0,
        })
        for fid in file_ids:
            edges.append({
                "source": ".",
                "target": fid,
                "type": "structure",
                "weight": 1.0,
                "label": "contains",
            })

    return edges, dir_nodes


def load_passages_from_index(index_path: str) -> list[dict]:
    """Load all passages from a leann JSONL file."""
    jsonl_candidates = [
        Path(f"{index_path}.passages.jsonl"),
    ]
    meta_path = Path(f"{index_path}.meta.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        for source in meta.get("passage_sources", []):
            rel = source.get("path_relative") or source.get("path", "")
            if rel:
                jsonl_candidates.insert(0, meta_path.parent / rel)

    for jsonl_path in jsonl_candidates:
        if jsonl_path.exists():
            passages = []
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        passages.append(json.loads(line))
            return passages

    return []


def compute_file_embeddings(
    passages: list[dict],
    leann_searcher,
    base_dir: str,
) -> dict[str, np.ndarray]:
    """Compute file-level embeddings by averaging passage embeddings."""
    base = Path(base_dir)

    texts_by_file: dict[str, list[str]] = {}
    for p in passages:
        meta = p.get("metadata", {})
        file_path = meta.get("file_path")
        if not file_path:
            continue
        try:
            rel = str(PurePosixPath(Path(file_path).relative_to(base)))
        except ValueError:
            rel = PurePosixPath(file_path).name
        texts_by_file.setdefault(rel, []).append(p.get("text", ""))

    file_embeddings = {}
    backend = leann_searcher.backend_impl

    for file_id, texts in texts_by_file.items():
        sample_texts = texts[:5]
        combined = " ".join(sample_texts)[:2000]

        try:
            embedding = backend.compute_query_embedding(
                combined, use_server_if_available=True
            )
            if embedding is not None and len(embedding) > 0:
                file_embeddings[file_id] = embedding.flatten()
        except Exception:
            continue

    return file_embeddings


def build_file_graph(
    leann_searcher,
    base_dir: str,
    top_k: int = 5,
    threshold: float = 0.6,
) -> dict:
    """Build the complete file graph from a leann index."""
    index_path = leann_searcher.meta_path_str.removesuffix(".meta.json")
    passages = load_passages_from_index(index_path)

    if not passages:
        return {"nodes": [], "edges": []}

    nodes = build_nodes(passages, base_dir)
    node_ids = {n["id"] for n in nodes}

    base = Path(base_dir)
    for node in nodes:
        try:
            stat = (base / node["id"]).stat()
            node["size"] = stat.st_size
        except (OSError, ValueError):
            pass

    passages_by_file: dict[str, list[str]] = {}
    for p in passages:
        meta = p.get("metadata", {})
        file_path = meta.get("file_path")
        if not file_path:
            continue
        try:
            rel = str(PurePosixPath(Path(file_path).relative_to(base)))
        except ValueError:
            rel = PurePosixPath(file_path).name
        passages_by_file.setdefault(rel, []).append(p.get("text", ""))

    edges = []

    try:
        file_embeddings = compute_file_embeddings(passages, leann_searcher, base_dir)
        if len(file_embeddings) >= 2:
            edges.extend(compute_similarity_edges(file_embeddings, top_k, threshold))
    except Exception:
        pass

    edges.extend(compute_reference_edges(passages_by_file, node_ids))

    structure_edges, dir_nodes = compute_structure_edges(list(node_ids))
    edges.extend(structure_edges)
    nodes.extend(dir_nodes)

    return {"nodes": nodes, "edges": edges}
