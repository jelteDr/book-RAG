"""Globaler Suchpfad (Stufe 2, Exp 9): Community-Berichte als Quellen für thematische Fragen.

Ablauf: Frage einbetten -> Top-m Community-Berichte (Cosine gegen community_vectors) ->
optional Map-Step (je Bericht bis 5 relevante Punkte mit 0-100-Score) -> Quellen bauen:
je Community der Bericht als Quelle [i] plus 2 repräsentative Originalpassagen [i+1], [i+2],
damit Zitate im Buchtext verankert bleiben. Die Quellen sind ScoredPoint-förmig, deshalb
funktionieren Prompt-Bau, Zitat-Extraktion und Frontend-Chips unverändert
(`chapter` = "Zusammenfassung: <Titel>", char_start/end = None).
"""

from __future__ import annotations

import json
import uuid

import numpy as np
from qdrant_client import models

from app.clients.ollama_client import OllamaClient
from app.clients.qdrant_client import VectorStore
from app.graph.communities import CommunityIndex
from app.graph.retriever import fetch_chunks
from app.rag.retriever import embed_question

_ID_NAMESPACE = uuid.UUID("636f6d6d-756e-4974-8000-000000000000")

GLOBAL_SYSTEM_EXTRA = (
    "\n7. Die Quellen sind teils ZUSAMMENFASSUNGEN von Themenbereichen des Romans, teils "
    "Originalpassagen. Synthetisiere über mehrere Quellen hinweg, beschreibe Entwicklungen im "
    "Verlauf des Romans und zitiere die Zusammenfassung [n] und — wo vorhanden — die "
    "passende Originalpassage [m]. Antworte in 4-8 Sätzen."
)

MAP_PROMPT = (
    "Question: {question}\n\nReport:\n{report}\n\n"
    "Extract up to 5 points from the report that help answer the question. Give each a "
    "relevance score from 0 to 100. Return compact JSON only: "
    '{{"points":[{{"description":"...","score":85}}]}} — an empty list if nothing is relevant.'
)
MAP_MIN_SCORE = 20
MAP_MAX_TOKENS = 350


def preselect_communities(
    index: CommunityIndex, qvec: np.ndarray, *, m: int
) -> list[tuple[str, float]]:
    """Top-m Communities nach Cosine Frage <-> Berichts-Embedding."""
    if index.vectors is None or not index.ids:
        return []
    sims = index.vectors @ qvec
    order = np.argsort(-sims)[:m]
    return [(index.ids[i], float(sims[i])) for i in order]


def _parse_map(raw: str) -> list[dict]:
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    out = []
    for p in data.get("points") or []:
        if isinstance(p, dict) and p.get("description"):
            try:
                score = max(0, min(100, int(float(p.get("score", 0)))))
            except (TypeError, ValueError):
                score = 0
            out.append({"description": " ".join(str(p["description"]).split())[:300], "score": score})
    return out[:5]


async def map_step(
    question: str, selected: list[tuple[str, float]], index: CommunityIndex,
    *, ollama: OllamaClient, model: str,
) -> dict[str, dict]:
    """Je Community ein LLM-Aufruf; Ergebnis {cid: {points, score}} (score = max Punkt)."""
    results: dict[str, dict] = {}
    for cid, sim in selected:
        report = index.reports[cid]
        prompt = MAP_PROMPT.format(question=question, report=report["text"])
        try:
            raw = await ollama.complete(
                [{"role": "user", "content": prompt}], model, temperature=0.0,
                max_tokens=MAP_MAX_TOKENS, response_format={"type": "json_object"},
            )
            points = _parse_map(raw)
        except Exception:
            points = []
        score = max((p["score"] for p in points), default=int(round(sim * 100)) if not points else 0)
        results[cid] = {"points": points, "score": score, "similarity": sim}
    return results


def community_point(index: CommunityIndex, cid: str, score: float) -> models.ScoredPoint:
    report = index.reports[cid]
    community = index.community(cid)
    book_id = (community.chunk_keys[0].rpartition(":")[0] if community.chunk_keys else None)
    return models.ScoredPoint(
        id=str(uuid.uuid5(_ID_NAMESPACE, f"{index.group_id}:{cid}")), version=0, score=score,
        payload={
            "kind": "community", "community_id": cid, "text": report["text"],
            "book_title": report.get("book_title") or index.book_title,
            "book_id": book_id, "chapter": f"Zusammenfassung: {report.get('title', cid)}",
            "char_start": None, "char_end": None, "chunk_keys": community.chunk_keys,
            "entities": report.get("entities", []),
        },
    )


async def build_global_points(
    index: CommunityIndex,
    selected: list[tuple[str, float]],
    *,
    vectors: VectorStore,
    chunks_per_community: int = 2,
    map_results: dict[str, dict] | None = None,
) -> list[models.ScoredPoint]:
    """[1] Bericht A, [2] Chunk A1, [3] Chunk A2, [4] Bericht B, … (Chunks dedupliziert)."""
    ordered = list(selected)
    if map_results:
        ordered = [(cid, map_results[cid]["score"] / 100) for cid, _ in selected
                   if map_results.get(cid, {}).get("score", 0) >= MAP_MIN_SCORE]
        ordered.sort(key=lambda kv: kv[1], reverse=True)
    wanted: list[str] = []
    for cid, _ in ordered:
        wanted += index.community(cid).chunk_keys[:chunks_per_community]
    found = await fetch_chunks(vectors, list(dict.fromkeys(wanted)), with_vectors=False)

    points: list[models.ScoredPoint] = []
    seen: set[str] = set()
    for cid, score in ordered:
        points.append(community_point(index, cid, score))
        for key in index.community(cid).chunk_keys[:chunks_per_community]:
            rec = found.get(key)
            if rec is None or key in seen:
                continue
            seen.add(key)
            points.append(models.ScoredPoint(
                id=rec.id, version=0, score=round(score - 0.001, 4), payload=rec.payload or {},
            ))
    return points


def map_hint(map_results: dict[str, dict], points: list[models.ScoredPoint]) -> str:
    """Vorab extrahierte Kernaussagen als Hinweisblock (Quellen bleiben identisch)."""
    marker = {(p.payload or {}).get("community_id"): i for i, p in enumerate(points, start=1)
              if (p.payload or {}).get("kind") == "community"}
    lines = []
    for cid, res in map_results.items():
        if cid not in marker:
            continue
        for p in res["points"]:
            lines.append(f"[{marker[cid]}] ({p['score']}/100) {p['description']}")
    return "Vorab extrahierte Kernaussagen:\n" + "\n".join(lines) if lines else ""


async def global_retrieve(
    question: str,
    *,
    index: CommunityIndex,
    ollama: OllamaClient,
    vectors: VectorStore,
    embed_model: str,
    model: str,
    m: int = 6,
    use_map: bool = False,
    chunks_per_community: int = 2,
    qvec: list[float] | None = None,
) -> tuple[list[models.ScoredPoint], dict]:
    """Liefert (Quellen-Points, Meta) — Meta: n_llm_calls, community_ids, map_scores, hint."""
    qvec = qvec or await embed_question(question, ollama=ollama, embed_model=embed_model)
    selected = preselect_communities(index, np.asarray(qvec, dtype=np.float32), m=m)
    meta: dict = {"mode": "global", "n_llm_calls": 0, "community_ids": [c for c, _ in selected],
                  "similarities": [round(s, 3) for _, s in selected], "hint": ""}
    map_results = None
    if use_map and selected:
        map_results = await map_step(question, selected, index, ollama=ollama, model=model)
        meta["n_llm_calls"] = len(selected)
        meta["map_scores"] = {cid: r["score"] for cid, r in map_results.items()}
    points = await build_global_points(
        index, selected, vectors=vectors, chunks_per_community=chunks_per_community,
        map_results=map_results,
    )
    if map_results:
        meta["hint"] = map_hint(map_results, points)
    return points, meta


def is_community_source(payload: dict | None) -> bool:
    return bool((payload or {}).get("kind") == "community")


def chunk_texts(points: list[models.ScoredPoint]) -> list[str]:
    """Nur Buchtext-Quellen (für faith_chunks im Eval)."""
    return [(p.payload or {}).get("text", "") for p in points if not is_community_source(p.payload)]

