"""Orchestriert die Ingestion: read -> parse -> clean -> chunk -> embed -> upsert.

Dry-Run liefert nur den Cleaning-Report (nichts landet im Index). Erst mit
commit=True wird gechunkt, eingebettet und in Qdrant upgesertet (idempotent:
vorhandene Chunks des Buchs werden zuvor gelöscht).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.clients.ollama_client import OllamaClient
from app.clients.qdrant_client import VectorStore
from app.ingestion.chunker import chunk_book
from app.ingestion.cleaner import clean
from app.ingestion.contextualizer import embedding_text, generate_contexts
from app.ingestion.parser import decode_bytes, parse_gutenberg, read_text
from app.util import l2_normalize
from app.validation import validate_document


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
    ollama: OllamaClient, texts: list[str], model: str,
    batch_size: int = 64, concurrency: int = 4,
) -> list[list[float]]:
    """Embeddet in Batches, mehrere Batches parallel (deutlich schneller als sequentiell)."""
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    sem = asyncio.Semaphore(concurrency)

    async def embed_one(batch: list[str]) -> list[list[float]]:
        async with sem:
            return await ollama.embed(batch, model)

    # gather bewahrt die Reihenfolge der Batches -> Chunk-Reihenfolge bleibt korrekt.
    results = await asyncio.gather(*(embed_one(b) for b in batches))
    return [vec for batch_vecs in results for vec in batch_vecs]


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
    contextual: bool = False,
    context_model: str | None = None,
) -> IngestResult:
    parsed = parse_gutenberg(raw_text)
    cleaned, report = clean(parsed.body)

    # Validierung VOR dem Chunking: Länge + gültige Zeichen.
    report_dict = report.as_dict()
    errors = validate_document(cleaned)
    report_dict["validation_errors"] = errors

    res_title = title or parsed.title
    res_author = author or parsed.author

    # Dry-Run ODER ungültig -> nicht chunken/embedden.
    if dry_run or errors:
        return IngestResult(
            book_id, group_id, res_title, res_author, parsed.language, 0, False, report_dict
        )

    chunks = chunk_book(cleaned)

    # Contextual Ingestion (opt-in, Exp 5): LLM-Kontext geht NUR ins Embedding,
    # der anzeigbare Chunk-Text bleibt unverändert. Kostet ~4-6 s pro Chunk.
    contexts: list[str] = [""] * len(chunks)
    if contextual and context_model:
        contexts = await generate_contexts(
            ollama, context_model, title=res_title, chunks=chunks
        )

    embed_texts = [embedding_text(ctx, c.text) for ctx, c in zip(contexts, chunks, strict=True)]
    embeddings = await _embed_batched(ollama, embed_texts, embed_model)

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
                **({"context": ctx} if ctx else {}),
            },
        }
        for c, emb, ctx in zip(chunks, embeddings, contexts, strict=True)
    ]

    await vectors.ensure_collection()
    await vectors.delete_by_book(book_id)  # idempotente Re-Ingestion
    n = await vectors.upsert_chunks(points)

    return IngestResult(
        book_id, group_id, res_title, res_author, parsed.language, n, True, report_dict
    )
