"""Retrieval-Evaluation gegen ein Gold-Set (Recall@k, MRR) — kapitel-basiert.

Kapitel-Relevanz ist ein pragmatischer Proxy (die Antwort steht im/in den
genannten Kapitel(n)); exakte Chunk-Labels wären genauer, aber teurer. Das
Skript nutzt denselben Retriever wie die App und ist reproduzierbar.

Aufruf (aus dem backend-Verzeichnis, damit die app-Module importierbar sind):
  uv run python ../eval/retrieval_eval.py --gold ../eval/gold_dracula.jsonl \
      --group horror-classics --k 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

# app-Module verfügbar machen (Skript liegt in eval/, App in backend/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.clients.ollama_client import OllamaClient  # noqa: E402
from app.clients.qdrant_client import VectorStore  # noqa: E402
from app.config import settings  # noqa: E402
from app.rag.retriever import retrieve  # noqa: E402

_ROMAN_RE = re.compile(r"CHAPTER\s+([IVXLCDM]+)", re.IGNORECASE)


def chapter_roman(chapter: str | None) -> str | None:
    if not chapter:
        return None
    m = _ROMAN_RE.search(chapter)
    return m.group(1).upper() if m else None


async def evaluate(gold_path: str, group_id: str | None, k: int, embed_model: str) -> None:
    gold = [json.loads(line) for line in Path(gold_path).read_text().splitlines() if line.strip()]

    ollama = OllamaClient(settings.ollama_base_url)
    vectors = VectorStore(settings.qdrant_url, settings.qdrant_collection)

    recall_at = {1: 0, 3: 0, 5: 0, k: 0}
    reciprocal_ranks: list[float] = []

    try:
        for item in gold:
            relevant = {r.upper() for r in item["relevant_chapters"]}
            points = await retrieve(
                item["question"],
                ollama=ollama,
                vectors=vectors,
                embed_model=embed_model,
                top_k=k,
                group_id=group_id,
            )
            ranks = [chapter_roman((p.payload or {}).get("chapter")) for p in points]
            hit_positions = [i + 1 for i, r in enumerate(ranks) if r in relevant]
            first = hit_positions[0] if hit_positions else None

            reciprocal_ranks.append(1.0 / first if first else 0.0)
            for kk in recall_at:
                if any(r in relevant for r in ranks[:kk]):
                    recall_at[kk] += 1

            status = f"@{first}" if first else "MISS"
            print(f"  {item['id']}: {status:>5}  top-Kapitel={ranks[:5]}  (soll {sorted(relevant)})")
    finally:
        await ollama.aclose()
        await vectors.aclose()

    n = len(gold)
    print(f"\n=== Ergebnisse (n={n}, k={k}, kapitel-basiert) ===")
    for kk in sorted(recall_at):
        print(f"  Recall@{kk}: {recall_at[kk] / n:.2f}  ({recall_at[kk]}/{n})")
    print(f"  MRR:       {sum(reciprocal_ranks) / n:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(Path(__file__).resolve().parent / "gold_dracula.jsonl"))
    ap.add_argument("--group", default="horror-classics")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--embed-model", default=settings.embed_model)
    args = ap.parse_args()
    asyncio.run(evaluate(args.gold, args.group, args.k, args.embed_model))


if __name__ == "__main__":
    main()
