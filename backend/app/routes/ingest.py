"""Ingestion-Endpunkt (Platzhalter).

Wird in M1 implementiert: Upload -> Encoding-Sniffing -> Metadaten -> Cleaning
(Dry-Run/Commit-Gate) -> Chunking -> Embeddings -> Qdrant-Upsert.
"""

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["ingest"])


@router.post("/ingest")
async def ingest() -> dict:
    raise HTTPException(status_code=501, detail="Ingestion wird in M1 implementiert.")
