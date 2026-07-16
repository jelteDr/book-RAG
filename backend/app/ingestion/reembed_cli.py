"""Re-Embed bestehender Bücher mit Contextual Ingestion — OHNE Originaldatei.

Die Chunk-Texte liegen bereits in der Qdrant-Payload; dieses CLI generiert pro
Chunk den LLM-Kontext (bzw. lädt ihn aus einer Checkpoint-Datei), embeddet
Kontext+Text neu und upsertet **in-place** (deterministische Punkt-IDs aus
book_id+chunk_index — die alten Punkte werden überschrieben).

Resumierbar: Kontexte werden nach jedem Chunk in die Checkpoint-Datei geschrieben.

Beispiele (aus dem backend-Verzeichnis):
  # Dracula mit vorhandenen Kontexten aus Exp 5 (schnell, nur Embeddings):
  uv run python -m app.ingestion.reembed_cli --group Horror \
      --contexts ../results/contextual_contexts.jsonl

  # Ganze Gruppe frisch kontextualisieren (teuer: ~4-6 s/Chunk!):
  uv run python -m app.ingestion.reembed_cli --group GoT
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.clients.ollama_client import OllamaClient
from app.clients.qdrant_client import VectorStore
from app.config import settings
from app.ingestion.contextualizer import embedding_text, generate_context
from app.util import l2_normalize


def _key(payload: dict) -> str:
    return f"{payload['book_id']}:{payload['chunk_index']}"


def _load_contexts(path: Path) -> dict[str, str]:
    contexts: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                contexts[row["key"]] = row["context"]
    return contexts


async def run(group_id: str | None, book_id: str | None, ctx_path: Path, batch: int) -> None:
    ollama = OllamaClient(settings.ollama_base_url)
    vectors = VectorStore(settings.qdrant_url, settings.qdrant_collection)
    try:
        records = await vectors.scroll_all(group_id)
        if book_id:
            records = [r for r in records if r.payload["book_id"] == book_id]
        records.sort(key=lambda r: (r.payload["book_id"], r.payload["chunk_index"]))
        if not records:
            print("Keine Chunks gefunden — --group/--book-id prüfen.")
            return
        print(f"{len(records)} Chunks zu re-embedden")

        contexts = _load_contexts(ctx_path)
        todo = [r for r in records if _key(r.payload) not in contexts]
        print(f"Kontexte: {len(records) - len(todo)} aus Checkpoint, {len(todo)} zu generieren")
        ctx_path.parent.mkdir(parents=True, exist_ok=True)
        with ctx_path.open("a", encoding="utf-8") as f:
            for i, rec in enumerate(todo, start=1):
                pl = rec.payload
                ctx = await generate_context(
                    ollama, settings.chat_model,
                    title=pl.get("book_title"), chapter=pl.get("chapter"), text=pl["text"],
                )
                contexts[_key(pl)] = ctx
                f.write(json.dumps({"key": _key(pl), "context": ctx}, ensure_ascii=False) + "\n")
                f.flush()
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} Kontexte generiert")

        done = 0
        for start in range(0, len(records), batch):
            part = records[start : start + batch]
            texts = [embedding_text(contexts.get(_key(r.payload), ""), r.payload["text"]) for r in part]
            embeddings = await ollama.embed(texts, settings.embed_model)
            points = []
            for rec, emb in zip(part, embeddings, strict=True):
                payload = dict(rec.payload)
                ctx = contexts.get(_key(rec.payload), "")
                if ctx:
                    payload["context"] = ctx
                points.append({"vector": l2_normalize(emb), "payload": payload})
            await vectors.upsert_chunks(points)
            done += len(part)
            if done % (batch * 4) == 0 or done == len(records):
                print(f"  {done}/{len(records)} re-embedded + upserted")
        print("Fertig — Punkte wurden in-place überschrieben (gleiche IDs).")
    finally:
        await ollama.aclose()
        await vectors.aclose()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", dest="group_id", help="alle Bücher dieser Gruppe")
    ap.add_argument("--book-id", help="nur dieses Buch (Qdrant-book_id)")
    ap.add_argument("--contexts", default="../results/contextual_contexts.jsonl",
                    help="Checkpoint-Datei (vorhandene Kontexte werden wiederverwendet)")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    if not args.group_id and not args.book_id:
        ap.error("--group oder --book-id angeben")
    asyncio.run(run(args.group_id, args.book_id, Path(args.contexts).resolve(), args.batch))


if __name__ == "__main__":
    main()
