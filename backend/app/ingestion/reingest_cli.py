"""Ein vorhandenes Buch neu einlesen (z. B. nach einem Cleaner-Fix) — ohne Ausfallzeit.

Ablauf je Buch: Datei wie beim Upload dekodieren/bereinigen -> chunken -> (optional) LLM-Kontext je
Chunk -> embedden -> unter einem NEUEN book_key in Qdrant schreiben -> erst danach die alten Punkte
löschen und die DB-Zeile (book_key, n_chunks, file_hash) umstellen. Bis zum Tausch bleibt der alte
Index aktiv; ein Abbruch lässt ihn unberührt.

Die Kontext-Generierung (~4-6 s/Chunk) ist resumierbar: jeder Kontext landet sofort in
results/reingest_<book-db-id>.jsonl (gitignored, Buchtext-Derivat). Ein Neustart überspringt
fertige Chunks (Schlüssel = Hash des Chunk-Texts).

Aufruf (aus backend/, DATABASE_URL auf die lokale Postgres-Instanz):
  DATABASE_URL=postgresql+asyncpg://bookrag:bookrag@localhost:5433/bookrag \\
  uv run python -m app.ingestion.reingest_cli --book 5 --file "../data/GoT/1 - A Game of Thrones.txt" \\
      --contextual [--commit]
Ohne --commit: Vorschau (Cleaning-Report, Chunk-Zahl, Kapitel), nichts wird geschrieben.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path

from app.clients.ollama_client import OllamaClient
from app.clients.qdrant_client import VectorStore
from app.config import settings
from app.db.models import Book, Group
from app.db.session import session_factory
from app.ingestion.chunker import chunk_book
from app.ingestion.cleaner import clean
from app.ingestion.contextualizer import embedding_text, generate_context
from app.ingestion.parser import decode_bytes, parse_gutenberg
from app.ingestion.pipeline import _embed_batched
from app.util import l2_normalize

RESULTS = Path(__file__).resolve().parents[3] / "results"


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:20]


async def run(book_db_id: int, path: Path, contextual: bool, commit: bool) -> None:
    raw = path.read_bytes()
    parsed = parse_gutenberg(decode_bytes(raw))
    cleaned, report = clean(parsed.body)
    chunks = chunk_book(cleaned)
    chapters = [c.chapter for c in chunks if c.chapter]
    async with session_factory() as session:
        book = await session.get(Book, book_db_id)
        if book is None:
            raise SystemExit(f"Buch {book_db_id} nicht gefunden")
        group = await session.get(Group, book.group_id)
    print(f"{book.title} (DB {book_db_id}, Gruppe {group.slug}): {len(chunks)} Chunks "
          f"(bisher {book.n_chunks}), Absätze per Zeilenlänge: {report.paragraphs_by_line_length}, "
          f"Chunks mit Kapitel: {len(chapters)}/{len(chunks)}")
    print("  erste Kapitel:", list(dict.fromkeys(chapters))[:8])
    if not commit:
        print("  Vorschau — mit --commit neu einlesen.")
        return

    ollama = OllamaClient(settings.ollama_base_url, timeout=300)
    vectors = VectorStore(settings.qdrant_url, settings.qdrant_collection)
    new_key = uuid.uuid4().hex
    try:
        contexts: dict[str, str] = {}
        if contextual:
            ckpt = RESULTS / f"reingest_{book_db_id}.jsonl"
            if ckpt.exists():
                for line in ckpt.read_text().splitlines():
                    row = json.loads(line)
                    contexts[row["key"]] = row["context"]
            todo = [c for c in chunks if _key(c.text) not in contexts]
            print(f"  Kontext: {len(chunks) - len(todo)} aus Checkpoint, {len(todo)} offen")
            t0 = time.perf_counter()
            with ckpt.open("a") as f:
                for i, c in enumerate(todo, start=1):
                    ctx = await generate_context(ollama, settings.chat_model, title=book.title,
                                                 chapter=c.chapter, text=c.text)
                    contexts[_key(c.text)] = ctx
                    f.write(json.dumps({"key": _key(c.text), "context": ctx}, ensure_ascii=False) + "\n")
                    f.flush()
                    if i % 25 == 0 or i == len(todo):
                        rate = (time.perf_counter() - t0) / i
                        print(f"  Kontext {i}/{len(todo)} — {rate:.1f} s/Chunk, "
                              f"Rest ~{rate * (len(todo) - i) / 60:.0f} min", flush=True)

        ctx_list = [contexts.get(_key(c.text), "") for c in chunks]
        embeddings = await _embed_batched(
            ollama, [embedding_text(x, c.text) for x, c in zip(ctx_list, chunks, strict=True)],
            settings.embed_model,
        )
        points = [
            {
                "vector": l2_normalize(emb),
                "payload": {
                    "text": c.text, "book_id": new_key, "group_id": group.slug,
                    "book_title": book.title, "author": book.author, "chapter": c.chapter,
                    "chunk_index": c.chunk_index, "char_start": c.char_start,
                    "char_end": c.char_end, "book_chars": len(cleaned),
                    **({"context": x} if x else {}),
                },
            }
            for c, emb, x in zip(chunks, embeddings, ctx_list, strict=True)
        ]
        await vectors.ensure_collection()
        try:
            for i in range(0, len(points), 256):
                await vectors.upsert_chunks(points[i : i + 256])
        except Exception:
            await vectors.delete_by_book(new_key)  # halbfertigen Neuaufbau verwerfen
            raise

        # Tausch: alte Punkte weg, DB auf den neuen Schlüssel.
        old_key = book.book_key
        async with session_factory() as session:
            book = await session.get(Book, book_db_id)
            book.book_key, book.n_chunks = new_key, len(points)
            book.file_hash = hashlib.sha256(raw).hexdigest()
            session.add(book)
            await session.commit()
        await vectors.delete_by_book(old_key)
        print(f"  fertig: {len(points)} Chunks unter neuem Schlüssel, alte Punkte ({old_key[:8]}…) gelöscht")
    finally:
        await ollama.aclose()
        await vectors.aclose()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", type=int, required=True, help="Buch-ID in der DB (Dokumente-Ansicht)")
    ap.add_argument("--file", required=True)
    ap.add_argument("--contextual", action="store_true", help="LLM-Kontext je Chunk (Exp 5)")
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args.book, Path(args.file).resolve(), args.contextual, args.commit))


if __name__ == "__main__":
    main()
