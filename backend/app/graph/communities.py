"""Community-Erkennung (Louvain) auf dem Entity-Graphen + Community-Index (Stufe 2, Exp 9).

Warum Louvain statt Leiden: `networkx.community.louvain_partitions` ist eingebaut, rein
Python und mit `seed` + sortierter Knotenreihenfolge (Builder) deterministisch — kein
C-Build (igraph/leidenalg). Leidens Garantie zusammenhängender Communities wird nachgebaut:
jede Louvain-Community wird in ihre Zusammenhangskomponenten zerlegt. Kleine Reste
(< min_size) werden nicht zusammengefasst, sondern im Partition-JSON als `rest` gelistet.

Artefakte unter <graph_dir>/<group>/:
  partition.json           Communities (Entities nach Grad, repräsentative Chunk-Keys)
  communities.jsonl        LLM-Berichte je Community (Checkpoint, community_reports.py)
  community_vectors.npz    bge-m3-Embeddings der Berichte (Vorauswahl bei globalen Fragen)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np

from app.graph.store import KnowledgeGraph

PARTITION_FILE = "partition.json"
REPORTS_FILE = "communities.jsonl"
COMMUNITY_VECTORS_FILE = "community_vectors.npz"


@dataclass
class Community:
    id: str
    level: int
    size: int
    entities: list[str]  # entity_ids, nach gewichtetem Grad absteigend
    internal_edges: int
    weight_sum: int
    chunk_keys: list[str]  # repräsentative Chunks (deterministisch, ohne LLM)


@dataclass
class Partition:
    group_id: str
    algorithm: str
    seed: int
    resolution: float
    min_size: int
    min_mentions: int
    level: int  # Index in den Louvain-Stufen (-1 = finale Stufe, beste Modularität)
    n_levels: int
    communities_per_level: list[int]
    n_nodes: int  # nach Pruning
    n_edges: int
    n_dropped_nodes: int
    modularity: float
    rest: list[str] = field(default_factory=list)
    communities: list[Community] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / PARTITION_FILE).write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> Partition:
        data = json.loads((directory / PARTITION_FILE).read_text(encoding="utf-8"))
        data["communities"] = [Community(**c) for c in data["communities"]]
        return cls(**data)


def prune_graph(graph: nx.Graph, *, min_mentions: int = 2) -> nx.Graph:
    """Extraktionsrauschen entfernen: seltene Entities (< min_mentions) und Isolierte."""
    pruned = graph.copy()
    rare = [n for n, d in pruned.nodes(data=True) if d.get("n_mentions", 0) < min_mentions]
    pruned.remove_nodes_from(rare)
    pruned.remove_nodes_from([n for n in list(pruned.nodes) if pruned.degree(n) == 0])
    return pruned


def representative_chunks(
    kg: KnowledgeGraph, entities: list[str], graph: nx.Graph, *, top_n: int = 3
) -> list[str]:
    """Chunks, die möglichst viele (und zentrale) Community-Entities erwähnen.

    Score je Chunk = Σ gewichteter Grad der erwähnten Community-Entities; Tie-Break über den
    Chunk-Key (deterministisch). Kein LLM nötig.
    """
    degree = dict(graph.degree(weight="weight"))
    score: dict[str, float] = {}
    for eid in entities:
        for key in kg.mentions.get(eid, ()):
            score[key] = score.get(key, 0.0) + float(degree.get(eid, 0))
    ranked = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ranked[:top_n]]


def detect_communities(
    kg: KnowledgeGraph,
    *,
    resolution: float = 1.0,
    seed: int = 42,
    min_size: int = 3,
    min_mentions: int = 2,
    level: int = -1,
    top_chunks: int = 3,
) -> Partition:
    graph = prune_graph(kg.graph, min_mentions=min_mentions)
    n_dropped = kg.graph.number_of_nodes() - graph.number_of_nodes()
    levels = list(nx.community.louvain_partitions(
        graph, weight="weight", resolution=resolution, seed=seed
    )) if graph.number_of_nodes() else []
    chosen = levels[level] if levels else []
    modularity = (
        nx.community.modularity(graph, chosen, weight="weight") if chosen else 0.0
    )

    communities: list[Community] = []
    rest: list[str] = []
    degree = dict(graph.degree(weight="weight"))
    # Sortierte Verarbeitung -> stabile IDs bei gleichem Seed.
    for members in sorted(chosen, key=lambda c: sorted(c)[0]):
        for comp in sorted(nx.connected_components(graph.subgraph(members)),
                           key=lambda c: sorted(c)[0]):
            if len(comp) < min_size:
                rest.extend(sorted(comp))
                continue
            sub = graph.subgraph(comp)
            entities = sorted(comp, key=lambda e: (-degree.get(e, 0), e))
            communities.append(Community(
                id="",  # wird unten nach Größe vergeben
                level=0,
                size=len(comp),
                entities=entities,
                internal_edges=sub.number_of_edges(),
                weight_sum=int(sum(d.get("weight", 1) for _, _, d in sub.edges(data=True))),
                chunk_keys=representative_chunks(kg, entities, graph, top_n=top_chunks),
            ))
    communities.sort(key=lambda c: (-c.size, c.entities[0]))
    for i, c in enumerate(communities):
        c.id = f"c{i:03d}"

    return Partition(
        group_id=kg.group_id, algorithm="louvain", seed=seed, resolution=resolution,
        min_size=min_size, min_mentions=min_mentions, level=level, n_levels=len(levels),
        communities_per_level=[len(p) for p in levels],
        n_nodes=graph.number_of_nodes(), n_edges=graph.number_of_edges(),
        n_dropped_nodes=n_dropped, modularity=round(float(modularity), 4),
        rest=sorted(rest), communities=communities,
    )


# --- Community-Index (Partition + Berichte + Vektoren) für die Laufzeit -------------

@dataclass
class CommunityIndex:
    group_id: str
    partition: Partition
    reports: dict[str, dict]  # community_id -> Bericht (title/summary/findings/text/...)
    ids: list[str] = field(default_factory=list)  # community_ids in Vektor-Reihenfolge
    vectors: np.ndarray | None = None
    book_title: str = ""

    def community(self, cid: str) -> Community:
        return next(c for c in self.partition.communities if c.id == cid)

    @classmethod
    def load(cls, directory: Path) -> CommunityIndex | None:
        if not (directory / PARTITION_FILE).exists() or not (directory / REPORTS_FILE).exists():
            return None
        partition = Partition.load(directory)
        reports = load_reports(directory / REPORTS_FILE)
        idx = cls(group_id=partition.group_id, partition=partition, reports=reports)
        vec_path = directory / COMMUNITY_VECTORS_FILE
        if vec_path.exists():
            with np.load(vec_path, allow_pickle=True) as npz:
                idx.ids = [str(x) for x in npz["ids"]]
                idx.vectors = npz["vectors"].astype(np.float32)
        idx.book_title = next((r.get("book_title", "") for r in reports.values()), "")
        return idx

    @classmethod
    def merge(cls, indexes: list[CommunityIndex]) -> CommunityIndex:
        """Mehrere Gruppen (Sammelgruppe): IDs mit Gruppe präfixieren, Vektoren stapeln."""
        if len(indexes) == 1:
            return indexes[0]
        reports: dict[str, dict] = {}
        ids: list[str] = []
        vecs: list[np.ndarray] = []
        communities: list[Community] = []
        for ix in indexes:
            prefix = f"{ix.group_id}/"
            for cid, rep in ix.reports.items():
                reports[prefix + cid] = rep
            for c in ix.partition.communities:
                communities.append(Community(**{**asdict(c), "id": prefix + c.id}))
            if ix.vectors is not None:
                ids.extend(prefix + i for i in ix.ids)
                vecs.append(ix.vectors)
        base = indexes[0].partition
        partition = Partition(**{**asdict(base), "communities": [], "rest": []})
        partition.communities = communities
        partition.group_id = "+".join(ix.group_id for ix in indexes)
        return cls(group_id=partition.group_id, partition=partition, reports=reports,
                   ids=ids, vectors=np.vstack(vecs) if vecs else None)


def load_reports(path: Path) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                reports[row["community_id"]] = row
    return reports


def save_community_vectors(directory: Path, ids: list[str], vectors: np.ndarray) -> None:
    np.savez_compressed(
        directory / COMMUNITY_VECTORS_FILE,
        ids=np.array(ids, dtype=object), vectors=vectors.astype(np.float32),
    )
