"""Graph-Extraktion: je Chunk Entities + Relationen per LLM (resumierbarer Checkpoint).

Die Chunk-Texte kommen aus der Qdrant-Payload (kein Originaltext nötig). Pro Chunk
eine Zeile in `<graph_dir>/<group>/extract.jsonl` — nach jedem Chunk geflusht, ein
Abbruch kostet also höchstens einen Chunk. Fehlerzeilen (kein JSON, Ollama-Fehler)
werden ebenfalls geschrieben, damit ein Resume nicht endlos an denselben Chunks
hängt; `--retry-errors` macht genau diese Zeilen neu.

Kosten: ein LLM-Aufruf je Chunk, realistisch 10-20 s mit qwen2.5:7b -> Dracula
(581 Chunks) ≈ 2-3 h. Vorher mit `--limit 20` pilotieren (s/Chunk, JSON-Quote).

Aufruf (aus dem backend-Verzeichnis; Ollama nativ, Qdrant-Container läuft):
  uv run python -m app.graph.extract_cli --group Horror --limit 20     # Pilot
  uv run python -m app.graph.extract_cli --group Horror                # Nachtjob
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from app.clients.ollama_client import OllamaClient
from app.clients.qdrant_client import VectorStore
from app.config import settings
from app.graph.extractor import extract_chunk
from app.graph.store import EXTRACT_FILE


def chunk_key(payload: dict) -> str:
    return f"{payload['book_id']}:{payload['chunk_index']}"


def load_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["key"]] = row
    return rows


def _summary(rows: dict[str, dict]) -> str:
    if not rows:
        return "keine Zeilen"
    n = len(rows)
    errors = sum(1 for r in rows.values() if r.get("error"))
    trunc = sum(1 for r in rows.values() if r.get("truncated"))
    ents = sum(len(r.get("entities") or []) for r in rows.values())
    rels = sum(len(r.get("relations") or []) for r in rows.values())
    ms = [r["ms"] for r in rows.values() if r.get("ms")]
    avg_s = (sum(ms) / len(ms) / 1000) if ms else 0.0
    return (f"{n} Chunks, {errors} Fehler ({errors / n:.1%}), {trunc} repariert, "
            f"Ø {ents / n:.1f} Entities, Ø {rels / n:.1f} Relationen, Ø {avg_s:.1f} s/Chunk")


async def run(
    group_id: str, book_id: str | None, out_path: Path, model: str,
    limit: int | None, retry_errors: bool,
) -> None:
    ollama = OllamaClient(settings.ollama_base_url, timeout=300.0)
    vectors = VectorStore(settings.qdrant_url, settings.qdrant_collection)
    try:
        records = await vectors.scroll_all(group_id)
        if book_id:
            records = [r for r in records if r.payload["book_id"] == book_id]
        records.sort(key=lambda r: (r.payload["book_id"], r.payload["chunk_index"]))
        if not records:
            print("Keine Chunks gefunden — --group/--book-id prüfen.")
            return

        rows = load_rows(out_path)
        if retry_errors:
            errored = [k for k, r in rows.items() if r.get("error")]
            for k in errored:
                rows.pop(k)
            out_path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows.values()),
                encoding="utf-8",
            )
            print(f"{len(errored)} Fehlerzeilen entfernt (werden neu extrahiert)")

        todo = [r for r in records if chunk_key(r.payload) not in rows]
        if limit:
            todo = todo[:limit]
        print(f"{len(records)} Chunks in Gruppe {group_id!r}: {len(rows)} im Checkpoint, "
              f"{len(todo)} zu extrahieren (Modell {model})")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        t_start = time.perf_counter()
        with out_path.open("a", encoding="utf-8") as f:
            for i, rec in enumerate(todo, start=1):
                pl = rec.payload
                t0 = time.perf_counter()
                ex = await extract_chunk(
                    ollama, model,
                    title=pl.get("book_title"), chapter=pl.get("chapter"),
                    context=pl.get("context"), text=pl["text"],
                )
                ms = int((time.perf_counter() - t0) * 1000)
                row = {
                    "key": chunk_key(pl), "book_id": pl["book_id"],
                    "chunk_index": pl["chunk_index"], "chapter": pl.get("chapter"),
                    "model": model, "entities": ex.entities, "relations": ex.relations,
                    "error": ex.error, "raw": ex.raw, "truncated": ex.truncated, "ms": ms,
                }
                rows[row["key"]] = row
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if i % 25 == 0 or i == len(todo):
                    elapsed = time.perf_counter() - t_start
                    per = elapsed / i
                    eta_min = per * (len(todo) - i) / 60
                    print(f"  {i}/{len(todo)}  Ø {per:.1f} s/Chunk  ETA {eta_min:.0f} min")
        print(f"Fertig. Checkpoint {out_path}: {_summary(rows)}")
    finally:
        await ollama.aclose()
        await vectors.aclose()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", dest="group_id", required=True, help="Qdrant-group_id (z. B. Horror)")
    ap.add_argument("--book-id", help="nur dieses Buch (Qdrant-book_id)")
    ap.add_argument("--out", help=f"Checkpoint (Default <graph_dir>/<group>/{EXTRACT_FILE})")
    ap.add_argument("--model", default=settings.chat_model)
    ap.add_argument("--limit", type=int, help="nur N Chunks (Pilot)")
    ap.add_argument("--retry-errors", action="store_true", help="Fehlerzeilen neu extrahieren")
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(settings.graph_dir) / args.group_id / EXTRACT_FILE
    asyncio.run(run(args.group_id, args.book_id, out.resolve(), args.model, args.limit,
                    args.retry_errors))


if __name__ == "__main__":
    main()
