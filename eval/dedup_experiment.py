"""Experiment: Overlap-Dedup im Retrieval — bringt Deduplizierung netto etwas?

Durch den Chunk-Overlap (~200 Zeichen) können Top-k-Treffer dieselbe Passage doppelt
enthalten. Zwei Arme, paired auf denselben Fragen (kapitel-basierte Metrik wie die
übrigen Experimente):

  baseline: Suche liefert top_k wie bisher (Duplikate möglich)
  dedup:    Suche holt top_k + Puffer, überlappende Passagen werden dedupliziert

Zusätzlich wird der direkte Effekt gemessen: wie oft steckten Duplikate in den Top-k
und wie viele *verschiedene* Kapitel erreichen das Prompt-Fenster.

Aufruf (aus dem backend-Verzeichnis):
  uv run python ../eval/dedup_experiment.py --gold ../eval/gold_dracula.jsonl \
      --group Horror --k 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.clients.ollama_client import OllamaClient  # noqa: E402
from app.clients.qdrant_client import VectorStore  # noqa: E402
from app.config import settings  # noqa: E402
from app.rag.retriever import DEDUP_EXTRA, dedup_overlaps  # noqa: E402
from app.util import l2_normalize  # noqa: E402
from qdrant_client import models  # noqa: E402

_ROMAN_RE = re.compile(r"CHAPTER\s+([IVXLCDM]+)", re.IGNORECASE)


def chapter_roman(chapter: str | None) -> str | None:
    m = _ROMAN_RE.search(chapter or "")
    return m.group(1).upper() if m else None


def metrics(points: list, relevant: set[str], k: int) -> tuple[bool, float]:
    """(Hit@k, Reciprocal Rank) kapitel-basiert."""
    ranks = [chapter_roman((p.payload or {}).get("chapter")) for p in points[:k]]
    hits = [i + 1 for i, r in enumerate(ranks) if r in relevant]
    return bool(hits), (1 / hits[0] if hits else 0.0)


def n_duplicates(points: list) -> int:
    """Anzahl Treffer, die eine bereits höher platzierte Passage überlappen."""
    return len(points) - len(dedup_overlaps(points, len(points)))


async def run(gold_path: str, group_id: str, k: int) -> None:
    gold = [json.loads(line) for line in Path(gold_path).read_text().splitlines() if line.strip()]
    ollama = OllamaClient(settings.ollama_base_url)
    vectors = VectorStore(settings.qdrant_url, settings.qdrant_collection)
    flt = models.Filter(
        must=[models.FieldCondition(key="group_id", match=models.MatchValue(value=group_id))]
    )

    rows = []
    try:
        for item in gold:
            relevant = {r.upper() for r in item["relevant_chapters"]}
            emb = (await ollama.embed([item["question"]], settings.embed_model))[0]
            raw = await vectors.search(l2_normalize(emb), k + DEDUP_EXTRA, flt)

            base = raw[:k]
            dedup = dedup_overlaps(raw, k)

            b_hit, b_rr = metrics(base, relevant, k)
            d_hit, d_rr = metrics(dedup, relevant, k)
            rows.append({
                "id": item["id"],
                "dups_in_topk": n_duplicates(base),
                "kapitel_base": len({chapter_roman((p.payload or {}).get("chapter")) for p in base}),
                "kapitel_dedup": len({chapter_roman((p.payload or {}).get("chapter")) for p in dedup}),
                "b_hit": b_hit, "b_rr": b_rr, "d_hit": d_hit, "d_rr": d_rr,
            })
    finally:
        await ollama.aclose()
        await vectors.aclose()

    n = len(rows)
    print(f"\n=== Overlap-Dedup-Experiment (n={n}, k={k}, Puffer={DEDUP_EXTRA}, kapitel-basiert) ===")
    print(f"Fragen mit >=1 Duplikat in Top-{k}:  "
          f"{sum(1 for r in rows if r['dups_in_topk'] > 0)}/{n}  "
          f"(Ø {sum(r['dups_in_topk'] for r in rows) / n:.2f} Duplikate)")
    print(f"Ø verschiedene Kapitel in Top-{k}:   "
          f"baseline {sum(r['kapitel_base'] for r in rows) / n:.2f}  ->  "
          f"dedup {sum(r['kapitel_dedup'] for r in rows) / n:.2f}")
    print(f"Hit-Rate@{k}:  baseline {sum(r['b_hit'] for r in rows) / n:.3f}   "
          f"dedup {sum(r['d_hit'] for r in rows) / n:.3f}")
    print(f"MRR:          baseline {sum(r['b_rr'] for r in rows) / n:.3f}   "
          f"dedup {sum(r['d_rr'] for r in rows) / n:.3f}")

    better = sum(1 for r in rows if r["d_rr"] > r["b_rr"])
    worse = sum(1 for r in rows if r["d_rr"] < r["b_rr"])
    print(f"Gepaarte MRR-Bilanz: {better} besser / {worse} schlechter / {n - better - worse} gleich")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="../eval/gold_dracula.jsonl")
    ap.add_argument("--group", default="Horror")
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    asyncio.run(run(args.gold, args.group, args.k))
