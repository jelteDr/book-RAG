"""Retrieval: Frage einbetten und in Qdrant nach ähnlichen Chunks suchen."""

from __future__ import annotations

from qdrant_client import models

from app.clients.ollama_client import OllamaClient
from app.clients.qdrant_client import VectorStore
from app.util import l2_normalize


async def retrieve(
    question: str,
    *,
    ollama: OllamaClient,
    vectors: VectorStore,
    embed_model: str,
    top_k: int,
    group_id: str | None = None,
) -> list[models.ScoredPoint]:
    query_emb = (await ollama.embed([question], embed_model))[0]
    query_vec = l2_normalize(query_emb)

    query_filter = None
    if group_id:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="group_id", match=models.MatchValue(value=group_id))]
        )

    return await vectors.search(query_vec, top_k, query_filter)
