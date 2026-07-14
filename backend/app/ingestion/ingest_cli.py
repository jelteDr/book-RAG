"""CLI zum Ingesten eines Buchs.

Beispiele:
  # Dry-Run (nur Cleaning-Report, nichts wird indexiert):
  uv run python -m app.ingestion.ingest_cli ../data/Dracula/dracula.txt \
      --book-id dracula --group horror-classics

  # Commit (chunken, einbetten, in Qdrant upserten):
  uv run python -m app.ingestion.ingest_cli ../data/Dracula/dracula.txt \
      --book-id dracula --group horror-classics --commit
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict

from app.clients.ollama_client import OllamaClient
from app.clients.qdrant_client import VectorStore
from app.config import settings
from app.ingestion.pipeline import ingest_file


async def _run(args: argparse.Namespace) -> None:
    ollama = OllamaClient(args.ollama_url)
    vectors = VectorStore(args.qdrant_url, settings.qdrant_collection)
    try:
        result = await ingest_file(
            args.path,
            book_id=args.book_id,
            group_id=args.group_id,
            ollama=ollama,
            vectors=vectors,
            embed_model=args.embed_model,
            dry_run=not args.commit,
            title=args.title,
            author=args.author,
        )
    finally:
        await ollama.aclose()
        await vectors.aclose()

    data = asdict(result)
    report = data.pop("report")
    print("=== Ergebnis ===")
    for k, v in data.items():
        print(f"  {k}: {v}")
    print("=== Cleaning-Report ===")
    for k, v in report.items():
        if k in ("before_sample", "after_sample"):
            print(f"  {k}: {v[:160]}…")
        else:
            print(f"  {k}: {v}")
    if not result.committed:
        print("\nDry-Run — nichts indexiert. Mit --commit tatsächlich ingesten.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Buch ingesten (Dry-Run oder Commit).")
    ap.add_argument("path", help="Pfad zur .txt-Datei")
    ap.add_argument("--book-id", required=True)
    ap.add_argument("--group", required=True, dest="group_id", help="Gruppen-ID (Franchise/Genre)")
    ap.add_argument("--title")
    ap.add_argument("--author")
    ap.add_argument("--commit", action="store_true", help="Ohne dieses Flag: Dry-Run")
    ap.add_argument("--ollama-url", default=settings.ollama_base_url)
    ap.add_argument("--qdrant-url", default=settings.qdrant_url)
    ap.add_argument("--embed-model", default=settings.embed_model)
    asyncio.run(_run(ap.parse_args()))


if __name__ == "__main__":
    main()
