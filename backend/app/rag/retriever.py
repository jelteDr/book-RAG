"""Retrieval: Frage einbetten und in Qdrant nach ähnlichen Chunks suchen.

Durch den Chunk-Overlap (~200 Zeichen) können Top-k-Treffer dieselbe Passage
doppelt enthalten. Die Suche holt deshalb einen kleinen Puffer und dedupliziert
überlappende Chunks (gleiches Buch, überschneidender Zeichenbereich) — so trägt
jeder Prompt-Platz unterschiedliche Information.
"""

from __future__ import annotations

from qdrant_client import models

from app.clients.ollama_client import OllamaClient
from app.clients.qdrant_client import VectorStore
from app.util import l2_normalize

# Wie viele zusätzliche Kandidaten geholt werden, damit nach dem Dedup
# trotzdem top_k verschiedene Passagen übrig bleiben.
DEDUP_EXTRA = 8


def _overlaps(a: dict, b: dict) -> bool:
    """True, wenn zwei Payloads dasselbe Buch und einen überlappenden Zeichenbereich haben."""
    if a.get("book_id") != b.get("book_id"):
        return False
    if None in (a.get("char_start"), a.get("char_end"), b.get("char_start"), b.get("char_end")):
        return False
    return a["char_start"] < b["char_end"] and b["char_start"] < a["char_end"]


def dedup_overlaps(
    points: list[models.ScoredPoint], limit: int
) -> list[models.ScoredPoint]:
    """Behält je überlappender Passage nur den bestplatzierten Treffer (max. `limit`)."""
    kept: list[models.ScoredPoint] = []
    for point in points:
        payload = point.payload or {}
        if any(_overlaps(payload, k.payload or {}) for k in kept):
            continue
        kept.append(point)
        if len(kept) == limit:
            break
    return kept


async def retrieve(
    question: str,
    *,
    ollama: OllamaClient,
    vectors: VectorStore,
    embed_model: str,
    top_k: int,
    group_id: str | None = None,
    group_ids: list[str] | None = None,
) -> list[models.ScoredPoint]:
    """`group_ids` (z. B. aus einer Sammelgruppe expandiert) hat Vorrang vor `group_id`."""
    query_emb = (await ollama.embed([question], embed_model))[0]
    query_vec = l2_normalize(query_emb)

    query_filter = None
    if group_ids:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="group_id", match=models.MatchAny(any=group_ids))]
        )
    elif group_id:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="group_id", match=models.MatchValue(value=group_id))]
        )

    # Puffer holen und überlappende Passagen deduplizieren (siehe Modul-Docstring).
    points = await vectors.search(query_vec, top_k + DEDUP_EXTRA, query_filter)
    return dedup_overlaps(points, top_k)
