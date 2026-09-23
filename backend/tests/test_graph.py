"""Unit-Tests für app.graph — reine Funktionen, kein Ollama/Qdrant nötig.

Aufruf (aus backend/): uv run pytest tests -q
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from qdrant_client import models

from app.graph.builder import build_graph, entity_key, merge_aliases, normalize_name
from app.graph.extractor import parse_extraction
from app.graph.retriever import (
    chunk_graph_scores,
    chunk_key,
    fuse,
    hop_activations,
    link_entities,
    seeds_from_dense,
)
from app.rag.retriever import dedup_overlaps

BOOK = "b1"


# --- Extraktion / Parser -------------------------------------------------------

def test_parse_handles_fences_prose_and_caps():
    raw = (
        "Sure, here is the graph:\n```json\n"
        '{"entities":[' + ",".join(
            f'{{"name":"E{i}","type":"character","desc":"d{i}"}}' for i in range(12)
        ) + '],"relations":[{"source":"E0","target":"E1","rel":"knows","desc":"x"}]}'
        "\n```\nThanks!"
    )
    ex = parse_extraction(raw)
    assert len(ex.entities) == 8  # MAX_ENTITIES
    assert ex.entities[0]["type"] == "PERSON"  # Alias 'character' -> PERSON
    assert ex.relations == [{"source": "E0", "target": "E1", "rel": "knows"}]
    assert ex.truncated is False


def test_parse_repairs_truncated_json():
    raw = ('{"entities":[{"name":"Lucy","type":"PERSON","desc":"a"},'
           '{"name":"Whitby","type":"PLACE","desc":"b"}],'
           '"relations":[{"source":"Lucy","target":"Whitby","rel":"visits"},{"source":"Lu')
    ex = parse_extraction(raw)
    assert ex.truncated is True
    assert [e["name"] for e in ex.entities] == ["Lucy", "Whitby"]
    assert len(ex.relations) == 1


def test_parse_drops_relations_with_unknown_endpoints_and_self_loops():
    raw = ('{"entities":[{"name":"A","type":"PERSON"},{"name":"B","type":"PLACE"}],'
           '"relations":[{"source":"A","target":"Ghost","rel":"x"},'
           '{"source":"a","target":"A","rel":"self"},{"source":"A","target":"b","rel":"in"}]}')
    ex = parse_extraction(raw)
    assert ex.relations == [{"source": "A", "target": "b", "rel": "in"}]


def test_parse_garbage_raises():
    with pytest.raises(ValueError):
        parse_extraction("I could not find any entities.")


# --- Normalisierung / Merge ----------------------------------------------------

def test_normalize_name_strips_titles_and_accents():
    assert normalize_name("Count Dracula") == "dracula"
    assert normalize_name("Dr. John Seward") == "john seward"
    assert normalize_name("Van Helsing") == "van helsing"  # 'van' ist kein Titel
    assert normalize_name("Bistriţa") == "bistrita"
    assert normalize_name("Landlord's wife") == "landlord wife"  # kein Prefix-Merge mit "landlord"
    assert entity_key("The Count", "PERSON") == "PERSON:"  # nur Titel -> leer (wird verworfen)


def test_merge_rule_unique_prefix_suffix_only_for_persons():
    keys = [
        "PERSON:lucy", "PERSON:lucy westenra",
        "PERSON:harker", "PERSON:jonathan harker", "PERSON:mina harker",
        "PERSON:van helsing", "PERSON:abraham van helsing",
        "PLACE:london", "PLACE:london bridge",
    ]
    mapping, ambiguous = merge_aliases(keys)
    assert mapping == {
        "PERSON:lucy": "PERSON:lucy westenra",
        "PERSON:van helsing": "PERSON:abraham van helsing",
    }
    assert ambiguous == [("PERSON:harker", ["PERSON:jonathan harker", "PERSON:mina harker"])]
    assert "PLACE:london" not in mapping  # Orte werden nie gemergt


# --- Graph-Build ---------------------------------------------------------------

def _rows() -> list[dict]:
    def ent(name, typ="PERSON", desc=""):
        return {"name": name, "type": typ, "desc": desc}

    def rel(s, t, r="knows"):
        return {"source": s, "target": t, "rel": r}

    return [
        {"key": f"{BOOK}:0", "book_id": BOOK, "entities": [ent("Count Dracula"), ent("Lucy Westenra")],
         "relations": [rel("Count Dracula", "Lucy Westenra", "bites")]},
        {"key": f"{BOOK}:1", "book_id": BOOK, "entities": [ent("Dracula"), ent("Lucy")],
         "relations": [rel("Dracula", "Lucy")]},
        {"key": f"{BOOK}:2", "book_id": BOOK, "entities": [ent("Dracula"), ent("Whitby", "PLACE")],
         "relations": [rel("Dracula", "Whitby", "arrives")]},
        {"key": f"{BOOK}:3", "book_id": BOOK, "entities": [ent("Dracula"), ent("wolves", "OBJECT")],
         "relations": [rel("Dracula", "wolves", "commands")], "error": None},
    ]


def test_build_graph_merges_aliases_and_computes_idf_and_weights():
    kg, report = build_graph(_rows(), "Horror", n_chunks=4)
    ids = set(kg.graph.nodes)
    assert ids == {"PERSON:dracula", "PERSON:lucy westenra", "PLACE:whitby"}  # 'wolves' weg
    assert report.n_common_nouns_dropped == 1
    assert report.merges == [("PERSON:lucy", "PERSON:lucy westenra")]
    assert kg.mentions["PERSON:dracula"] == [f"{BOOK}:0", f"{BOOK}:1", f"{BOOK}:2", f"{BOOK}:3"]
    assert kg.idf("PERSON:dracula") == 0.0  # Hub: in allen Chunks -> ln(4/4)
    assert kg.idf("PLACE:whitby") == pytest.approx(np.log(4), abs=1e-3)
    assert kg.graph["PERSON:dracula"]["PERSON:lucy westenra"]["weight"] == 2  # zwei Chunks
    assert kg.name("PERSON:lucy westenra") == "Lucy Westenra"
    assert kg.chunk_entities[f"{BOOK}:1"] == ["PERSON:dracula", "PERSON:lucy westenra"]


def test_graph_scores_prefer_rare_entities_and_hop_decays():
    kg, _ = build_graph(_rows(), "Horror", n_chunks=4)
    act = hop_activations(kg, [("PLACE:whitby", 0.9)], neighbours=3, decay=0.5)
    assert act == {"PLACE:whitby": 0.9, "PERSON:dracula": 0.45}
    scores = chunk_graph_scores(kg, act)
    # Dracula hat idf 0 -> nur Whitby trägt bei -> nur Chunk 2 bekommt Signal.
    assert set(scores) == {f"{BOOK}:2"}
    assert scores[f"{BOOK}:2"] == pytest.approx(0.9 * np.log(4), abs=1e-3)


def test_link_entities_respects_threshold_and_top_m():
    kg, _ = build_graph(_rows(), "Horror", n_chunks=4)
    kg.entity_ids = sorted(kg.graph.nodes)
    kg.entity_vectors = np.eye(3, dtype=np.float32)  # jede Entity eine eigene Achse
    q = np.array([0.2, 0.9, 0.5], dtype=np.float32)
    linked = link_entities(kg, q, top_m=2, min_sim=0.45)
    assert [e for e, _ in linked] == [kg.entity_ids[1], kg.entity_ids[2]]


# --- Fusion --------------------------------------------------------------------

def _point(idx: int, score: float) -> models.ScoredPoint:
    return models.ScoredPoint(
        id=f"id-{idx}", version=0, score=score,
        payload={"book_id": BOOK, "chunk_index": idx, "char_start": idx * 100,
                 "char_end": idx * 100 + 90, "text": f"chunk {idx}"},
    )


class _FakeVectors:
    """Liefert Records mit 2-dim-Vektoren; Chunk i hat Vektor (cos_i, sqrt(1-cos_i^2))."""

    def __init__(self, cos: dict[int, float]) -> None:
        self.cos = cos

    async def get_by_indices(self, book_id, idxs, with_vectors=False):
        out = []
        for i in idxs:
            if i in self.cos:
                c = self.cos[i]
                p = _point(i, c)
                out.append(models.Record(id=p.id, payload=p.payload,
                                         vector=[c, float(np.sqrt(1 - c * c))]))
        return out


def test_fuse_alpha_zero_is_identical_to_dense_and_alpha_pulls_graph_chunk_in():
    kg, _ = build_graph(_rows(), "Horror", n_chunks=4)
    dense = [_point(0, 0.80), _point(1, 0.78), _point(3, 0.70)]  # Chunk 2 fehlt im dense
    qvec = np.array([1.0, 0.0], dtype=np.float32)
    vectors = _FakeVectors({2: 0.69})  # knapp unter dem letzten dense-Treffer
    act = {"PLACE:whitby": 1.0}
    scores = chunk_graph_scores(kg, act)  # nur Chunk 2

    base = dedup_overlaps(dense, 2)
    fused0 = asyncio.run(fuse(dense, scores, act, kg=kg, qvec=qvec, vectors=vectors,
                              top_k=2, pool=20, alpha=0.0))
    assert [chunk_key(p.payload) for p in fused0] == [chunk_key(p.payload) for p in base]
    assert [p.score for p in fused0] == [p.score for p in base]

    fused = asyncio.run(fuse(dense, scores, act, kg=kg, qvec=qvec, vectors=vectors,
                             top_k=2, pool=20, alpha=0.10))
    keys = [chunk_key(p.payload) for p in fused]
    assert keys == [f"{BOOK}:0", f"{BOOK}:2"]  # 0.69 + 0.10 = 0.79 > 0.78
    assert fused[1].payload["graph_entities"] == ["Whitby"]


def test_seeds_from_dense_weights_by_rank_and_caps():
    kg, _ = build_graph(_rows(), "Horror", n_chunks=4)
    seeds = seeds_from_dense(kg, [_point(2, 0.9), _point(0, 0.8)], seed_hits=3, max_seeds=1)
    assert list(seeds) == ["PLACE:whitby"]  # Dracula (idf 0) fällt beim Kürzen raus
