"""Graph-Build: extract.jsonl -> graph.json + entity_vectors.npz (unter <graph_dir>/<group>/).

Druckt Statistiken (Knoten je Typ, Top-Grad, Top-Erwähnungen) und den Merge-Report
(welche Aliase zusammengelegt wurden, welche ambig blieben) — bitte vor dem Messlauf
lesen: falsche Merges verfälschen das Retrieval-Signal.

Aufruf (aus dem backend-Verzeichnis):
  uv run python -m app.graph.build_cli --group Horror
  uv run python -m app.graph.build_cli --group Horror --no-embed   # nur Struktur
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np

from app.clients.ollama_client import OllamaClient
from app.config import settings
from app.graph.builder import build_graph, stats
from app.graph.extract_cli import load_rows
from app.graph.store import EXTRACT_FILE, KnowledgeGraph
from app.util import l2_normalize

EMBED_BATCH = 32


async def embed_entities(kg: KnowledgeGraph, ollama: OllamaClient, embed_model: str) -> None:
    ids = sorted(kg.graph.nodes)
    texts = [kg.entity_text(e) for e in ids]
    vecs: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        vecs.extend(l2_normalize(v) for v in await ollama.embed(batch, embed_model))
        if (start // EMBED_BATCH) % 10 == 0:
            print(f"  {min(start + EMBED_BATCH, len(texts))}/{len(texts)} Entities embedded")
    kg.entity_ids = ids
    kg.entity_vectors = np.asarray(vecs, dtype=np.float32)


async def run(group_id: str, extract_path: Path, out_dir: Path, embed: bool) -> None:
    rows = list(load_rows(extract_path).values())
    if not rows:
        print(f"Keine Extraktionen in {extract_path} — erst extract_cli laufen lassen.")
        return
    model = next((r.get("model") for r in rows if r.get("model")), "")
    kg, report = build_graph(rows, group_id, n_chunks=len(rows), model=model)

    print(f"Extraktion: {report.n_rows} Chunks ({report.n_error_rows} Fehlerzeilen), "
          f"{report.n_entities_raw} Entity-Nennungen, {report.n_relations_raw} Relationen "
          f"({report.n_relations_dropped} verworfen: Endpunkt unbekannt/Self-Loop)")
    st = stats(kg)
    print(f"Graph: {st['n_nodes']} Knoten, {st['n_edges']} Kanten, {st['isolated']} isoliert, "
          f"Chunks mit Entities: {st['chunks_with_entities']}/{len(rows)}")
    print(f"  je Typ: {st['by_type']}")
    print("  Top-Grad (gewichtet): " + ", ".join(f"{n} ({d})" for n, d in st["top_degree"]))
    print("  Top-Erwähnungen:      " + ", ".join(f"{n} ({m})" for n, m in st["top_mentions"]))
    print(f"Merge-Report: {len(report.merges)} Aliase zusammengelegt, "
          f"{len(report.ambiguous)} ambig belassen")
    for src, dst in report.merges[:40]:
        print(f"    {src}  ->  {dst}")
    if len(report.merges) > 40:
        print(f"    … +{len(report.merges) - 40} weitere")
    for key, cands in report.ambiguous[:20]:
        print(f"    ambig: {key}  ~  {', '.join(cands)}")

    if embed:
        ollama = OllamaClient(settings.ollama_base_url)
        try:
            await embed_entities(kg, ollama, settings.embed_model)
        finally:
            await ollama.aclose()
    kg.save(out_dir)
    print(f"Gespeichert: {out_dir}/graph.json" + (" + entity_vectors.npz" if embed else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", dest="group_id", required=True)
    ap.add_argument("--extract", help=f"Checkpoint (Default <graph_dir>/<group>/{EXTRACT_FILE})")
    ap.add_argument("--out-dir", help="Zielverzeichnis (Default <graph_dir>/<group>)")
    ap.add_argument("--no-embed", action="store_true", help="keine Entity-Embeddings (nur Struktur)")
    args = ap.parse_args()
    base = Path(settings.graph_dir) / args.group_id
    extract = Path(args.extract) if args.extract else base / EXTRACT_FILE
    out_dir = Path(args.out_dir) if args.out_dir else base
    asyncio.run(run(args.group_id, extract.resolve(), out_dir.resolve(), not args.no_embed))


if __name__ == "__main__":
    main()
