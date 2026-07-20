"""Dünne Hülle um den Qdrant-Client.

Collection nutzt **Cosine**-Distanz (bge-m3-Embeddings werden L2-normalisiert).
Die Payload trägt den anzeigbaren Chunk-Text (Source-of-Truth) plus Metadaten
für Filter (book_id, group_id, chapter, char_start/end).
"""

import uuid
from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient, models

# bge-m3 liefert Dense-Vektoren der Dimension 1024.
VECTOR_SIZE = 1024

# Fester Namespace für deterministische Punkt-IDs (book_id + chunk_index).
_ID_NAMESPACE = uuid.UUID("6b6f6f6b-2d72-4167-8000-000000000000")


class VectorStore:
    def __init__(self, url: str, collection: str) -> None:
        self._client = AsyncQdrantClient(url=url)
        self._collection = collection

    async def aclose(self) -> None:
        await self._client.close()

    async def ensure_collection(self) -> None:
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE, distance=models.Distance.COSINE
                ),
            )
            # Indizierte Payload-Felder für schnelles Gruppen-/Buch-Filtering.
            for field in ("book_id", "group_id"):
                await self._client.create_payload_index(
                    self._collection, field, models.PayloadSchemaType.KEYWORD
                )

    async def upsert_chunks(self, points: Sequence[dict]) -> int:
        """Upsert von Chunks. Jeder dict: {vector, payload} mit payload.book_id/chunk_index."""
        structs = [
            models.PointStruct(
                id=str(
                    uuid.uuid5(
                        _ID_NAMESPACE,
                        f"{p['payload']['book_id']}:{p['payload']['chunk_index']}",
                    )
                ),
                vector=p["vector"],
                payload=p["payload"],
            )
            for p in points
        ]
        await self._client.upsert(collection_name=self._collection, points=structs)
        return len(structs)

    async def delete_by_book(self, book_id: str) -> None:
        """Alle Chunks eines Buchs löschen (Re-Ingestion / Copyright-Cleanup)."""
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="book_id", match=models.MatchValue(value=book_id)
                        )
                    ]
                )
            ),
        )

    async def search(
        self, vector: list[float], top_k: int, query_filter: models.Filter | None = None
    ) -> list[models.ScoredPoint]:
        result = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return result.points

    async def get_by_indices(
        self, book_id: str, chunk_indices: Sequence[int]
    ) -> list[models.Record]:
        """Chunks eines Buchs direkt über ihre deterministischen IDs holen.

        Fehlende Indizes (z. B. vor dem Buchanfang / nach dem Ende) fallen
        einfach aus dem Ergebnis — kein Fehler.
        """
        ids = [
            str(uuid.uuid5(_ID_NAMESPACE, f"{book_id}:{i}")) for i in chunk_indices
        ]
        return await self._client.retrieve(
            collection_name=self._collection, ids=ids, with_payload=True
        )

    async def scroll_all(
        self, group_id: str | None = None, batch: int = 256
    ) -> list[models.Record]:
        """Alle Chunks (optional gefiltert) auslesen — z. B. für einen BM25-Index."""
        flt = None
        if group_id:
            flt = models.Filter(
                must=[models.FieldCondition(key="group_id", match=models.MatchValue(value=group_id))]
            )
        results: list[models.Record] = []
        offset = None
        while True:
            points, offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=flt,
                limit=batch,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            results.extend(points)
            if offset is None:
                break
        return results

    async def health(self) -> bool:
        """True, wenn Qdrant erreichbar ist."""
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False
