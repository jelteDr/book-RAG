"""Orchestriert die Ingestion: read -> parse -> clean -> chunk -> embed -> upsert.

Dry-Run liefert nur den Cleaning-Report (nichts landet im Index). Erst mit
commit=True wird gechunkt, eingebettet und in Qdrant upgesertet (idempotent:
vorhandene Chunks des Buchs werden zuvor gelöscht).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.clients.ollama_client import OllamaClient
from app.clients.qdrant_client import VectorStore
from app.ingestion.chunker import chunk_book
from app.ingestion.cleaner import clean
from app.ingestion.parser import decode_bytes, parse_gutenberg, read_text
from app.util import l2_normalize


@dataclass
class IngestResult:
    book_id: str
    group_id: str
    title: str | None
    author: str | None
    language: str | None
    n_chunks: int
    committed: bool
    report: dict


async def _embed_batched(
    ollama: OllamaClient, texts: list[str], model: str, batch_size: int = 32
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        vectors.extend(await ollama.embed(texts[i : i + batch_size], model))
    return vectors


async def ingest_file(path: str, **kwargs) -> IngestResult:
    """Ingestet eine Datei vom Pfad (CLI)."""
    return await ingest_text(read_text(path), **kwargs)


async def ingest_bytes(raw: bytes, **kwargs) -> IngestResult:
    """Ingestet hochgeladene Rohbytes (Upload-Endpoint)."""
    return await ingest_text(decode_bytes(raw), **kwargs)


async def ingest_text(
    raw_text: str,
    *,
    book_id: str,
    group_id: str,
    ollama: OllamaClient,
    vectors: VectorStore,
    embed_model: str,
    dry_run: bool = True,
    title: str | None = None,
    author: str | None = None,
) -> IngestResult:
    parsed = parse_gutenberg(raw_text)
    cleaned, report = clean(parsed.body)

    res_title = title or parsed.title
    res_author = author or parsed.author

    if dry_run:
        return IngestResult(
            book_id, group_id, res_title, res_author, parsed.language, 0, False, report.as_dict()
        )

    chunks = chunk_book(cleaned)
    embeddings = await _embed_batched(ollama, [c.text for c in chunks], embed_model)

    points = [
        {
            "vector": l2_normalize(emb),
            "payload": {
                "text": c.text,
                "book_id": book_id,
                "group_id": group_id,
                "book_title": res_title,
                "author": res_author,
                "chapter": c.chapter,
                "chunk_index": c.chunk_index,
                "char_start": c.char_start,
                "char_end": c.char_end,
            },
        }
        for c, emb in zip(chunks, embeddings, strict=True)
    ]

    await vectors.ensure_collection()
    await vectors.delete_by_book(book_id)  # idempotente Re-Ingestion
    n = await vectors.upsert_chunks(points)

    return IngestResult(
        book_id, group_id, res_title, res_author, parsed.language, n, True, report.as_dict()
    )
