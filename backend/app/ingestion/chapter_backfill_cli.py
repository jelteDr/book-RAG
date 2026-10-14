"""Kapitel nachträglich in vorhandene Chunks eintragen — ohne Neu-Chunking, ohne Re-Embedding.

Bücher ohne „Chapter/Kapitel“-Überschriften (Game of Thrones: POV-Namen, Harry Potter:
Kapiteltitel in Großbuchstaben) wurden mit `chapter=None` ingestet; im Chat stand dann statt
einer Kapitelangabe nur die Chunk-Nummer/der Score. Der erweiterte Chunker erkennt solche
Überschriften. Hier wird der bereinigte Text aus der Originaldatei exakt wie beim Upload
rekonstruiert (decode -> parse -> clean), jede Überschrift-Position bestimmt und jedem
gespeicherten Chunk die letzte Überschrift vor seinem `char_start` zugeordnet.

Sicherheitsnetz: geschrieben wird nur, wenn JEDER gespeicherte Chunk-Text exakt an seinen
Offsets im rekonstruierten Text steht (sonst passt die Datei nicht zum Upload -> Abbruch).
Ohne --commit nur Vorschau.

Aufruf (aus backend/):
  uv run python -m app.ingestion.chapter_backfill_cli --group GoT \\
      --book-id 0446f94f... --file "../data/GoT/1 - A Game of Thrones.txt" [--commit]
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
from collections import defaultdict
from pathlib import Path

from app.clients.qdrant_client import VectorStore
from app.config import settings
from app.ingestion.chunker import chapter_starts
from app.ingestion.cleaner import clean
from app.ingestion.parser import decode_bytes, parse_gutenberg


def cleaned_text(path: Path) -> str:
    parsed = parse_gutenberg(decode_bytes(path.read_bytes()))
    return clean(parsed.body)[0]


async def run(group: str, book_id: str, path: Path, commit: bool) -> None:
    text = cleaned_text(path)
    starts = chapter_starts(text)
    offsets = [o for o, _ in starts]
    vectors = VectorStore(settings.qdrant_url, settings.qdrant_collection)
    try:
        records = [r for r in await vectors.scroll_all(group_id=group)
                   if (r.payload or {}).get("book_id") == book_id]
        if not records:
            raise SystemExit(f"Keine Chunks für book_id={book_id} in Gruppe {group}")
        mismatch = [r for r in records
                    if text[r.payload["char_start"]:r.payload["char_end"]] != r.payload["text"]]
        print(f"{path.name}: {len(records)} Chunks, {len(starts)} Überschriften, "
              f"Text-Abgleich {len(records) - len(mismatch)}/{len(records)}")
        if mismatch:
            raise SystemExit("Abbruch: Datei passt nicht exakt zum gespeicherten Upload.")

        by_chapter: dict[str | None, list] = defaultdict(list)
        for r in records:
            i = bisect.bisect_right(offsets, r.payload["char_start"]) - 1
            by_chapter[starts[i][1] if i >= 0 else None].append(r.id)
        print("  erste Überschriften:", [t for _, t in starts[:6]])
        print(f"  Chunks ohne Kapitel (vor der ersten Überschrift): {len(by_chapter.get(None, []))}")
        if not commit:
            print("  Vorschau — mit --commit schreiben.")
            return
        # Bücher ohne erkannte Überschriften behalten ihr Kapitel (None) — nur book_chars kommt
        # dazu, damit der Chat ersatzweise die Position im Buch zeigen kann.
        if starts:
            for chapter, ids in by_chapter.items():
                await vectors.set_payload(ids, {"chapter": chapter})
        await vectors.set_payload([r.id for r in records], {"book_chars": len(text)})
        print(f"  geschrieben: book_chars={len(text)}"
              + (f", {len(by_chapter)} Kapitel" if starts else ", keine Kapitel (Position statt Kapitel)"))
    finally:
        await vectors.aclose()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", required=True)
    ap.add_argument("--book-id", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args.group, args.book_id, Path(args.file).resolve(), args.commit))


if __name__ == "__main__":
    main()
