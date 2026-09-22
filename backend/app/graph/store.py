"""Persistenz des Entity-Graphen: graph.json (Struktur) + entity_vectors.npz (Embeddings).

Ein Graph pro Gruppe unter `<graph_dir>/<group>/`. Der Graph ist klein (≤ einige
tausend Knoten) und liegt zur Laufzeit komplett im Speicher — deshalb Dateien +
networkx statt Postgres/Qdrant (KISS, Rebuild = Datei überschreiben).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
import numpy as np

GRAPH_FILE = "graph.json"
VECTORS_FILE = "entity_vectors.npz"
EXTRACT_FILE = "extract.jsonl"


@dataclass
class KnowledgeGraph:
    group_id: str
    book_ids: list[str]
    n_chunks: int
    graph: nx.Graph  # Knoten: name/type/aliases/desc/n_mentions/idf; Kanten: weight/rels
    mentions: dict[str, list[str]]  # entity_id -> chunk_keys ("book_id:chunk_index")
    chunk_entities: dict[str, list[str]]  # chunk_key -> entity_ids (Umkehrindex)
    entity_ids: list[str] = field(default_factory=list)  # Zeilenreihenfolge der Vektoren
    entity_vectors: np.ndarray | None = None  # (n, 1024) float32, L2-normalisiert
    model: str = ""
    built_at: str = ""

    # --- Zugriff -------------------------------------------------------------
    def name(self, entity_id: str) -> str:
        return self.graph.nodes[entity_id]["name"]

    def idf(self, entity_id: str) -> float:
        return float(self.graph.nodes[entity_id]["idf"])

    def entity_text(self, entity_id: str) -> str:
        """Text, der für das Entity-Embedding verwendet wird (Linking DE-Frage -> EN-Entity)."""
        node = self.graph.nodes[entity_id]
        desc = node.get("desc") or ""
        return f"{node['name']} ({node['type']}): {desc}" if desc else f"{node['name']} ({node['type']})"

    # --- Persistenz ----------------------------------------------------------
    def to_json(self) -> dict:
        nodes = [
            {"id": nid, **{k: v for k, v in data.items()}}
            for nid, data in sorted(self.graph.nodes(data=True))
        ]
        edges = [
            {"source": u, "target": v, **data}
            for u, v, data in sorted(self.graph.edges(data=True))
        ]
        return {
            "group_id": self.group_id,
            "book_ids": self.book_ids,
            "n_chunks": self.n_chunks,
            "model": self.model,
            "built_at": self.built_at or datetime.now(UTC).isoformat(timespec="seconds"),
            "nodes": nodes,
            "edges": edges,
            "mentions": self.mentions,
        }

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / GRAPH_FILE).write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        if self.entity_vectors is not None and self.entity_ids:
            np.savez_compressed(
                directory / VECTORS_FILE,
                ids=np.array(self.entity_ids, dtype=object),
                vectors=self.entity_vectors.astype(np.float32),
            )
        elif (directory / VECTORS_FILE).exists():
            # Kein Vektor-Set mitgegeben (--no-embed): alte Vektoren passen nicht mehr zu den
            # Knoten -> löschen, sonst würde das Linking auf falsche Entities zeigen.
            (directory / VECTORS_FILE).unlink()

    @classmethod
    def load(cls, directory: Path) -> KnowledgeGraph:
        data = json.loads((directory / GRAPH_FILE).read_text(encoding="utf-8"))
        graph = nx.Graph()
        for node in data["nodes"]:  # sortierte Einfügereihenfolge (Louvain-Determinismus)
            nid = node.pop("id")
            graph.add_node(nid, **node)
        for edge in data["edges"]:
            graph.add_edge(edge.pop("source"), edge.pop("target"), **edge)
        mentions = {k: list(v) for k, v in data["mentions"].items()}
        chunk_entities: dict[str, list[str]] = {}
        for eid, keys in mentions.items():
            for key in keys:
                chunk_entities.setdefault(key, []).append(eid)
        kg = cls(
            group_id=data["group_id"], book_ids=data["book_ids"], n_chunks=data["n_chunks"],
            graph=graph, mentions=mentions, chunk_entities=chunk_entities,
            model=data.get("model", ""), built_at=data.get("built_at", ""),
        )
        vec_path = directory / VECTORS_FILE
        if vec_path.exists():
            with np.load(vec_path, allow_pickle=True) as npz:
                kg.entity_ids = [str(x) for x in npz["ids"]]
                kg.entity_vectors = npz["vectors"].astype(np.float32)
        return kg


def merge_graphs(graphs: list[KnowledgeGraph]) -> KnowledgeGraph:
    """Mehrere Gruppen-Graphen (Sammelgruppe) zu einem zusammenlegen.

    Entity-IDs werden mit der Gruppe präfixiert, damit „Dracula" aus zwei Gruppen nicht
    kollidiert; Chunk-Keys sind ohnehin global eindeutig (book_id:chunk_index).
    """
    if len(graphs) == 1:
        return graphs[0]
    graph = nx.Graph()
    mentions: dict[str, list[str]] = {}
    chunk_entities: dict[str, list[str]] = {}
    ids: list[str] = []
    vecs: list[np.ndarray] = []
    for kg in graphs:
        prefix = f"{kg.group_id}/"
        relabeled = nx.relabel_nodes(kg.graph, {n: prefix + n for n in kg.graph.nodes})
        graph = nx.compose(graph, relabeled)
        for eid, keys in kg.mentions.items():
            mentions[prefix + eid] = list(keys)
            for key in keys:
                chunk_entities.setdefault(key, []).append(prefix + eid)
        if kg.entity_vectors is not None:
            ids.extend(prefix + e for e in kg.entity_ids)
            vecs.append(kg.entity_vectors)
    return KnowledgeGraph(
        group_id="+".join(g.group_id for g in graphs),
        book_ids=sorted({b for g in graphs for b in g.book_ids}),
        n_chunks=sum(g.n_chunks for g in graphs),
        graph=graph, mentions=mentions, chunk_entities=chunk_entities,
        entity_ids=ids, entity_vectors=np.vstack(vecs) if vecs else None,
    )
