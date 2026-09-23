"""Communities + Berichte + Berichts-Embeddings für eine Gruppe (Stufe 2, Exp 9).

Schritte einzeln schaltbar:
  (immer)      Louvain-Partition aus graph.json -> partition.json (deterministisch, Seed)
  --summarize  LLM-Bericht je Community -> communities.jsonl (Checkpoint, ~1 min/Community)
  --embed      bge-m3-Embeddings der Berichte -> community_vectors.npz
  --force      partition.json neu berechnen, auch wenn vorhanden (Berichte bleiben, sofern
               die Community-IDs stabil sind — bei geänderter Partition vorher communities.jsonl
               löschen!)

Aufruf (aus dem backend-Verzeichnis):
  uv run python -m app.graph.communities_cli --group Horror                       # nur Partition
  uv run python -m app.graph.communities_cli --group Horror --summarize --embed   # komplett
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np

from app.clients.ollama_client import OllamaClient
from app.clients.qdrant_client import VectorStore
from app.config import settings
from app.graph.communities import (
    PARTITION_FILE,
    REPORTS_FILE,
    Partition,
    detect_communities,
    load_reports,
    save_community_vectors,
)
from app.graph.community_reports import embedding_text, generate_reports
from app.graph.store import KnowledgeGraph
from app.util import l2_normalize

EMBED_BATCH = 32


async def run(args: argparse.Namespace) -> None:
    base = Path(settings.graph_dir) / args.group_id
    kg = KnowledgeGraph.load(base)
    print(f"Graph {args.group_id}: {kg.graph.number_of_nodes()} Knoten, "
          f"{kg.graph.number_of_edges()} Kanten, {kg.n_chunks} Chunks")

    if (base / PARTITION_FILE).exists() and not args.force:
        partition = Partition.load(base)
        print(f"Partition geladen ({len(partition.communities)} Communities; --force für Neuaufbau)")
    else:
        partition = detect_communities(
            kg, resolution=args.resolution, seed=args.seed, min_size=args.min_size,
            min_mentions=args.min_mentions, level=args.level, top_chunks=args.top_chunks,
        )
        partition.save(base)
        print(f"Partition: {partition.n_nodes} Knoten nach Pruning ({partition.n_dropped_nodes} "
              f"entfernt), Louvain-Stufen {partition.communities_per_level}, gewählt Stufe "
              f"{partition.level}, Modularität {partition.modularity}")
        print(f"  {len(partition.communities)} Communities (>= {args.min_size}), "
              f"{len(partition.rest)} Entities im Rest")
    for c in partition.communities[:15]:
        names = ", ".join(kg.name(e) for e in c.entities[:5])
        print(f"  {c.id} size={c.size:3d} edges={c.internal_edges:3d}  {names}")
    if len(partition.communities) > 15:
        print(f"  … +{len(partition.communities) - 15} weitere")

    if not (args.summarize or args.embed):
        return

    ollama = OllamaClient(settings.ollama_base_url, timeout=300.0)
    vectors = VectorStore(settings.qdrant_url, settings.qdrant_collection)
    try:
        reports = load_reports(base / REPORTS_FILE)
        if args.summarize:
            records = await vectors.scroll_all(args.group_id)
            chunks_by_key = {f"{r.payload['book_id']}:{r.payload['chunk_index']}": r.payload
                             for r in records}
            title = next((r.payload.get("book_title") for r in records
                          if r.payload.get("book_title")), args.group_id)
            todo = [c for c in partition.communities if c.id not in reports]
            print(f"Berichte: {len(reports)} vorhanden, {len(todo)} zu erzeugen (Modell {args.model})")
            reports = await generate_reports(
                kg, partition, chunks_by_key, ollama=ollama, model=args.model,
                path=base / REPORTS_FILE, title=title, existing=reports,
                on_progress=lambda i, n: print(f"  {i}/{n} Berichte"),
            )
            fallback = sum(1 for r in reports.values() if r.get("raw_fallback"))
            print(f"Berichte fertig: {len(reports)}, davon {fallback} ohne gültiges JSON (Rohtext)")

        if args.embed:
            ids = [c.id for c in partition.communities if c.id in reports]
            texts = [embedding_text(reports[i]) for i in ids]
            vecs: list[list[float]] = []
            for start in range(0, len(texts), EMBED_BATCH):
                vecs.extend(l2_normalize(v) for v in
                            await ollama.embed(texts[start : start + EMBED_BATCH], settings.embed_model))
            save_community_vectors(base, ids, np.asarray(vecs, dtype=np.float32))
            print(f"Embeddings: {len(ids)} Berichte -> community_vectors.npz")
    finally:
        await ollama.aclose()
        await vectors.aclose()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", dest="group_id", required=True)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-size", type=int, default=3)
    ap.add_argument("--min-mentions", type=int, default=2)
    ap.add_argument("--level", type=int, default=-1,
                    help="Louvain-Stufe (-1 = finale/gröbste, 0 = feinste)")
    ap.add_argument("--top-chunks", type=int, default=3)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--model", default=settings.chat_model)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
