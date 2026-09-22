"""Unit-Tests für Stufe 2 (Communities, Berichte, globaler Suchpfad, Router) — ohne LLM/Qdrant."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from qdrant_client import models

from app.graph.builder import build_graph
from app.graph.communities import CommunityIndex, Partition, detect_communities, prune_graph
from app.graph.community_reports import (
    build_report_messages,
    embedding_text,
    parse_report,
    report_to_text,
)
from app.graph.global_search import (
    build_global_points,
    chunk_texts,
    map_hint,
    preselect_communities,
)
from app.graph.router import heuristic_mode

BOOK = "b1"


def _two_cliques() -> list[dict]:
    """Zwei dichte Gruppen (A0..A3, B0..B3), verbunden über EINE schwache Kante; jede
    Entity in >= 2 Chunks, dazu ein seltener Knoten (Rare) mit nur einer Erwähnung."""
    rows = []
    a = [f"A{i}" for i in range(4)]
    b = [f"B{i}" for i in range(4)]
    idx = 0

    def row(ents, rels):
        nonlocal idx
        r = {"key": f"{BOOK}:{idx}", "book_id": BOOK,
             "entities": [{"name": e, "type": "PERSON", "desc": f"about {e}"} for e in ents],
             "relations": [{"source": s, "target": t, "rel": "knows"} for s, t in rels]}
        idx += 1
        return r

    for _ in range(3):  # Clique A dreimal, Clique B dreimal
        rows.append(row(a, [(a[i], a[j]) for i in range(4) for j in range(i + 1, 4)]))
        rows.append(row(b, [(b[i], b[j]) for i in range(4) for j in range(i + 1, 4)]))
    rows.append(row([a[0], b[0], "Rare"], [(a[0], b[0])]))  # Brücke + seltener Knoten
    return rows


def _kg():
    kg, _ = build_graph(_two_cliques(), "Test", n_chunks=7)
    return kg


def test_prune_removes_rare_and_isolated_nodes():
    kg = _kg()
    pruned = prune_graph(kg.graph, min_mentions=2)
    assert "PERSON:rare" not in pruned
    assert pruned.number_of_nodes() == 8


def test_detect_communities_is_deterministic_and_splits_cliques():
    kg = _kg()
    p1 = detect_communities(kg, seed=42, min_size=3)
    p2 = detect_communities(kg, seed=42, min_size=3)
    assert p1.to_json() == p2.to_json()
    assert len(p1.communities) == 2
    sizes = sorted(c.size for c in p1.communities)
    assert sizes == [4, 4]
    assert p1.n_dropped_nodes == 1  # Rare
    assert p1.modularity > 0.3
    for c in p1.communities:
        assert c.chunk_keys and len(c.chunk_keys) <= 3
        assert c.entities[0] in c.entities  # nach Grad sortiert, nicht leer
    assert [c.id for c in p1.communities] == ["c000", "c001"]


def test_partition_roundtrip(tmp_path):
    p = detect_communities(_kg(), seed=42)
    p.save(tmp_path)
    assert Partition.load(tmp_path).to_json() == p.to_json()


def test_report_prompt_and_parse_and_text():
    kg = _kg()
    part = detect_communities(kg, seed=42)
    chunks = {f"{BOOK}:{i}": {"chapter": "CHAPTER I", "text": "x" * 2000} for i in range(7)}
    messages = build_report_messages(kg, part.communities[0], chunks, title="Test Novel")
    assert messages[0]["role"] == "system" and "Test Novel" in messages[0]["content"]
    assert "ENTITIES:" in messages[1]["content"] and "RELATIONSHIPS:" in messages[1]["content"]
    assert len(messages[1]["content"]) <= 18000

    raw = ('```json\n{"title":"Group A","summary":"They know each other.",'
           '"findings":[{"summary":"f1","explanation":"A0 and A1"},"plain finding"]}\n```')
    rep = parse_report(raw)
    assert rep["title"] == "Group A" and len(rep["findings"]) == 2
    text = report_to_text(rep)
    assert text.startswith("Group A") and "- f1 — A0 and A1" in text
    assert "plain finding" in embedding_text(rep)
    with pytest.raises(ValueError):
        parse_report('{"title": "no summary"}')


class _FakeVectors:
    async def get_by_indices(self, book_id, idxs, with_vectors=False):
        return [models.Record(id=f"id-{i}", payload={"book_id": book_id, "chunk_index": i,
                                                      "text": f"chunk {i}", "chapter": "I"})
                for i in idxs]


def _index() -> CommunityIndex:
    kg = _kg()
    part = detect_communities(kg, seed=42)
    reports = {
        c.id: {"community_id": c.id, "title": f"T{c.id}", "summary": "s", "findings": [],
               "text": f"report {c.id}", "book_title": "Test Novel", "entities": []}
        for c in part.communities
    }
    ix = CommunityIndex(group_id="Test", partition=part, reports=reports,
                        ids=[c.id for c in part.communities],
                        vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    return ix


def test_global_points_order_dedup_and_texts():
    ix = _index()
    sel = preselect_communities(ix, np.array([0.9, 0.1], dtype=np.float32), m=2)
    assert [c for c, _ in sel] == ["c000", "c001"]
    pts = asyncio.run(build_global_points(ix, sel, vectors=_FakeVectors(), chunks_per_community=2))
    kinds = [(p.payload or {}).get("kind", "chunk") for p in pts]
    assert kinds[0] == "community" and kinds[1] == "chunk"
    assert kinds.count("community") == 2
    assert pts[0].payload["chapter"].startswith("Zusammenfassung: Tc000")
    assert pts[0].payload["char_start"] is None and pts[0].score >= pts[1].score
    keys = [(p.payload["book_id"], p.payload["chunk_index"]) for p in pts if "chunk_index" in p.payload]
    assert len(keys) == len(set(keys))  # Chunks community-übergreifend dedupliziert
    assert all(t.startswith("chunk") for t in chunk_texts(pts))


def test_map_results_reorder_filter_and_hint():
    ix = _index()
    sel = [("c000", 0.9), ("c001", 0.8)]
    mres = {"c000": {"points": [], "score": 10, "similarity": 0.9},
            "c001": {"points": [{"description": "key point", "score": 80}], "score": 80,
                     "similarity": 0.8}}
    pts = asyncio.run(build_global_points(ix, sel, vectors=_FakeVectors(), map_results=mres))
    comm = [p.payload["community_id"] for p in pts if p.payload.get("kind") == "community"]
    assert comm == ["c001"]  # c000 unter MAP_MIN_SCORE -> raus
    hint = map_hint(mres, pts)
    assert hint.startswith("Vorab extrahierte Kernaussagen:") and "[1] (80/100) key point" in hint


def test_router_heuristic():
    assert heuristic_mode("Wie entwickelt sich Minas Rolle im Verlauf des Romans?") == "global"
    assert heuristic_mode("Welche Rolle spielen Tagebücher als Erzählmittel?") == "global"
    assert heuristic_mode("Wann kommt Jonathan in Bistritz an?") == "local"
    assert heuristic_mode("Was isst Jonathan zum Frühstück?") is None
