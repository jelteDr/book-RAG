"""Dünne Hülle um den Qdrant-Client.

Collection nutzt **Cosine**-Distanz (bge-m3-Embeddings werden L2-normalisiert).
Die Payload trägt den anzeigbaren Chunk-Text (Source-of-Truth) plus Metadaten
für Filter (book_id, group_id, chapter, char_start/end).
"""

from qdrant_client import AsyncQdrantClient, models

# bge-m3 liefert Dense-Vektoren der Dimension 1024.
VECTOR_SIZE = 1024


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

    async def health(self) -> bool:
        """True, wenn Qdrant erreichbar ist."""
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False
