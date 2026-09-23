"""Graph-gestütztes Retrieval (Exp 8): dense-erhaltende Fusion mit einem Graph-Signal.

Grundidee: Dense (bge-m3, contextual) bleibt der Träger. Der Graph liefert pro Chunk
ein zusätzliches Signal g(c) und darf Kandidaten nur um höchstens `alpha` anheben:

    s(c) = cos(q, c) + alpha * g(c) / max g

Bei alpha = 0 ist das Ergebnis exakt die Baseline (Sanity-Check im Eval). Bewusst KEIN
RRF: In Exp 2/6 verdrängte Rang-Fusion dense-only-Treffer und verschlechterte netto.

Zwei Varianten:
  link   — Frage -> Entities (Cosine Frage-Embedding vs. Entity-Embeddings, d. h. die
           deutsche Frage findet englische Entities) -> 1-Hop-Nachbarn (gedämpft) ->
           Chunks, in denen diese Entities erwähnt werden: g(c) = Σ a(e) * idf(e).
  expand — dense zuerst; Seeds = Entities der Top-3-Treffer (Pseudo-Relevance-Feedback);
           g(c) nur für Chunks, die >= 2 Seeds gemeinsam erwähnen (kein Hop).
  graph_only — Diagnose: Ranking rein nach g (wie `sparse` solo in Exp 6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qdrant_client import models

from app.clients.ollama_client import OllamaClient
from app.clients.qdrant_client import VectorStore
from app.graph.store import KnowledgeGraph
from app.rag.retriever import DEDUP_EXTRA, build_group_filter, dedup_overlaps, embed_question

MODES = ("link", "expand", "graph_only")


@dataclass(frozen=True)
class GraphParams:
    alpha: float = 0.03  # Gewicht des Graph-Signals (≈ Cosine-Abstand Rang 1<->5 bei bge-m3)
    top_m: int = 5  # verlinkte Entities je Frage
    min_sim: float = 0.45  # Mindest-Cosine fürs Linking
    hop_neighbours: int = 3  # Nachbarn je verlinkter Entity (nach Kantengewicht)
    hop_decay: float = 0.5  # Aktivierung der Nachbarn relativ zur Entity
    pool: int = 20  # so viele Graph-Kandidaten kommen zusätzlich in die Fusion
    expand_dense_k: int = 16  # dense-Treffer, aus denen `expand` seine Seeds zieht
    expand_seed_hits: int = 3  # nur die besten N Treffer liefern Seeds
    expand_max_seeds: int = 8
    expand_min_shared: int = 2  # Ko-Erwähnung: Chunk muss >= 2 Seeds enthalten


def chunk_key(payload: dict | None) -> str:
    payload = payload or {}
    return f"{payload.get('book_id')}:{payload.get('chunk_index')}"


def link_entities(
    kg: KnowledgeGraph, qvec: np.ndarray, *, top_m: int, min_sim: float
) -> list[tuple[str, float]]:
    """Top-m Entities nach Cosine zur Frage (>= min_sim)."""
    if kg.entity_vectors is None or not kg.entity_ids:
        return []
    sims = kg.entity_vectors @ qvec
    order = np.argsort(-sims)[:top_m]
    return [(kg.entity_ids[i], float(sims[i])) for i in order if sims[i] >= min_sim]


def hop_activations(
    kg: KnowledgeGraph, linked: list[tuple[str, float]], *, neighbours: int, decay: float
) -> dict[str, float]:
    """Verlinkte Entities + je Entity die stärksten Nachbarn (gedämpft)."""
    act = {eid: sim for eid, sim in linked}
    for eid, sim in linked:
        nbrs = sorted(
            kg.graph[eid].items(), key=lambda kv: kv[1].get("weight", 1), reverse=True
        )[:neighbours]
        for nb, _ in nbrs:
            if nb not in dict(linked):
                act[nb] = max(act.get(nb, 0.0), decay * sim)
    return act


def chunk_graph_scores(
    kg: KnowledgeGraph, activations: dict[str, float], *, min_shared: int = 1
) -> dict[str, float]:
    """g(c) = Σ_{e erwähnt in c} a(e) * idf(e); optional nur Chunks mit >= min_shared Entities."""
    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    for eid, a in activations.items():
        idf = kg.idf(eid)
        if a * idf <= 0.0:  # Hubs (idf 0) tragen nichts bei -> keine Null-Kandidaten erzeugen
            continue
        for key in kg.mentions.get(eid, ()):
            scores[key] = scores.get(key, 0.0) + a * idf
            counts[key] = counts.get(key, 0) + 1
    if min_shared > 1:
        scores = {k: v for k, v in scores.items() if counts[k] >= min_shared}
    return scores


def seeds_from_dense(
    kg: KnowledgeGraph, points: list[models.ScoredPoint], *, seed_hits: int, max_seeds: int
) -> dict[str, float]:
    """Entities der besten dense-Treffer, gewichtet mit 1/Rang, auf max_seeds nach a*idf gekürzt."""
    act: dict[str, float] = {}
    for rank, p in enumerate(points[:seed_hits], start=1):
        for eid in kg.chunk_entities.get(chunk_key(p.payload), ()):
            act[eid] = act.get(eid, 0.0) + 1.0 / rank
    top = sorted(act.items(), key=lambda kv: kv[1] * kg.idf(kv[0]), reverse=True)[:max_seeds]
    return dict(top)


async def fetch_chunks(
    vectors: VectorStore, keys: list[str], *, with_vectors: bool
) -> dict[str, models.Record]:
    """Chunks per deterministischer ID holen, gruppiert je Buch."""
    by_book: dict[str, list[int]] = {}
    for key in keys:
        book_id, _, idx = key.rpartition(":")
        by_book.setdefault(book_id, []).append(int(idx))
    found: dict[str, models.Record] = {}
    for book_id, idxs in by_book.items():
        for rec in await vectors.get_by_indices(book_id, idxs, with_vectors=with_vectors):
            found[chunk_key(rec.payload)] = rec
    return found


def _annotate(kg: KnowledgeGraph, payload: dict, key: str, g: float, act: dict[str, float]) -> dict:
    names = [kg.name(e) for e in kg.chunk_entities.get(key, ()) if e in act][:5]
    return {**payload, "graph_score": round(g, 4), "graph_entities": names}


async def fuse(
    dense: list[models.ScoredPoint],
    graph_scores: dict[str, float],
    activations: dict[str, float],
    *,
    kg: KnowledgeGraph,
    qvec: np.ndarray,
    vectors: VectorStore,
    top_k: int,
    pool: int,
    alpha: float,
) -> list[models.ScoredPoint]:
    gmax = max(graph_scores.values(), default=0.0)
    dense_keys = {chunk_key(p.payload) for p in dense}
    fused: list[models.ScoredPoint] = []
    for p in dense:  # dense-Reihenfolge zuerst -> stabile Sortierung erhält sie bei alpha=0
        key = chunk_key(p.payload)
        g = graph_scores.get(key, 0.0) / gmax if gmax else 0.0
        fused.append(models.ScoredPoint(
            id=p.id, version=p.version, score=p.score + alpha * g,
            payload=_annotate(kg, p.payload or {}, key, g, activations),
        ))
    # Graph-only-Kandidaten können nur mit alpha > 0 überhaupt aufholen (ihr Cosine liegt
    # unter dem letzten dense-Treffer) — deshalb bei alpha=0 gar nicht erst holen.
    if alpha > 0 and gmax:
        ranked = sorted(graph_scores.items(), key=lambda kv: kv[1], reverse=True)
        extra_keys = [k for k, _ in ranked[:pool] if k not in dense_keys]
        for key, rec in (await fetch_chunks(vectors, extra_keys, with_vectors=True)).items():
            vec = np.asarray(rec.vector, dtype=np.float32)
            cos = float(vec @ qvec)
            g = graph_scores[key] / gmax
            fused.append(models.ScoredPoint(
                id=rec.id, version=0, score=cos + alpha * g,
                payload=_annotate(kg, rec.payload or {}, key, g, activations),
            ))
    fused.sort(key=lambda p: p.score, reverse=True)
    return dedup_overlaps(fused, top_k)


async def graph_retrieve(
    question: str,
    *,
    kg: KnowledgeGraph,
    mode: str,
    ollama: OllamaClient,
    vectors: VectorStore,
    embed_model: str,
    top_k: int,
    group_id: str | None = None,
    group_ids: list[str] | None = None,
    params: GraphParams = GraphParams(),
    qvec: list[float] | None = None,
    diagnostics: dict | None = None,
) -> list[models.ScoredPoint]:
    """Signatur wie `app.rag.retriever.retrieve` plus Graph; liefert ScoredPoints (top_k)."""
    if mode not in MODES:
        raise ValueError(f"unbekannter Graph-Modus {mode!r}")
    qvec = qvec or await embed_question(question, ollama=ollama, embed_model=embed_model)
    q = np.asarray(qvec, dtype=np.float32)
    flt = build_group_filter(group_id, group_ids)
    n_dense = top_k + DEDUP_EXTRA

    if mode == "expand":
        dense = await vectors.search(qvec, max(params.expand_dense_k, n_dense), flt)
        seeds = dedup_overlaps(dense, params.expand_dense_k)
        activations = seeds_from_dense(
            kg, seeds, seed_hits=params.expand_seed_hits, max_seeds=params.expand_max_seeds
        )
        scores = chunk_graph_scores(kg, activations, min_shared=params.expand_min_shared)
        dense = dense[:n_dense]
    else:
        linked = link_entities(kg, q, top_m=params.top_m, min_sim=params.min_sim)
        activations = hop_activations(
            kg, linked, neighbours=params.hop_neighbours, decay=params.hop_decay
        )
        scores = chunk_graph_scores(kg, activations)
        if diagnostics is not None:
            diagnostics["linked"] = [(kg.name(e), round(s, 3)) for e, s in linked]
        if mode == "graph_only":
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n_dense]
            gmax = ranked[0][1] if ranked else 0.0
            found = await fetch_chunks(vectors, [k for k, _ in ranked], with_vectors=False)
            points = [
                models.ScoredPoint(
                    id=found[k].id, version=0, score=g / gmax,
                    payload=_annotate(kg, found[k].payload or {}, k, g / gmax, activations),
                )
                for k, g in ranked if k in found
            ]
            return dedup_overlaps(points, top_k)
        dense = await vectors.search(qvec, n_dense, flt)

    if diagnostics is not None:
        diagnostics["n_activated"] = len(activations)
        diagnostics["n_graph_candidates"] = len(scores)
    return await fuse(
        dense, scores, activations, kg=kg, qvec=q, vectors=vectors,
        top_k=top_k, pool=params.pool, alpha=params.alpha,
    )
