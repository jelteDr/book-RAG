"""Retrieval-Evaluation gegen ein Gold-Set (Hit-Rate@k, MRR).

Unterstützt beide Label-Formate und rechnet vorhandene Arme parallel:
- v1 (`relevant_chapters`): kapitel-basiert — pragmatischer, aber grober Proxy.
- v2 (`spans` aus notebooks/gold_set_v2.ipynb): passagen-genau — Treffer, wenn ein
  gefundener Chunk einen Gold-Span (book_id + char_start/char_end) überlappt.
  v2-Items tragen `chapters` als Vergleichs-Feld, damit die Überschätzung der
  Kapitel-Metrik direkt sichtbar wird.

Das Skript nutzt denselben Retriever wie die App (inkl. Overlap-Dedup).

Aufruf (aus dem backend-Verzeichnis, damit die app-Module importierbar sind):
  uv run python ../eval/retrieval_eval.py --gold ../eval/gold_dracula.jsonl \
      --group Horror --k 8
  uv run python ../eval/retrieval_eval.py --gold ../eval/gold_v2.jsonl --group Horror
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


def span_hit(payload: dict, item: dict) -> bool:
    """Span-Metrik (Gold-Set v2): Treffer, wenn der Chunk einen Gold-Span überlappt.

    Ein Span ist {book_id, char_start, char_end} im bereinigten Buchtext — er überlebt
    Re-Chunking (Chunk-Indizes verschieben sich, Zeichen-Offsets nicht).
    """
    if item.get("book_id") and payload.get("book_id") != item["book_id"]:
        return False
    return any(
        payload.get("char_start") is not None
        and payload["char_start"] < s["char_end"]
        and s["char_start"] < payload["char_end"]
        for s in item.get("spans", [])
    )


def _first_hit(flags: list[bool]) -> int | None:
    hits = [i + 1 for i, f in enumerate(flags) if f]
    return hits[0] if hits else None


class _Arm:
    """Sammelt Hit-Rate@k + MRR für einen Metrik-Arm (kapitel- oder span-basiert)."""

    def __init__(self, k: int) -> None:
        self.hits_at = {1: 0, 3: 0, 5: 0, k: 0}
        self.rr: list[float] = []

    def add(self, flags: list[bool]) -> int | None:
        first = _first_hit(flags)
        self.rr.append(1.0 / first if first else 0.0)
        for kk in self.hits_at:
            if any(flags[:kk]):
                self.hits_at[kk] += 1
        return first

    def report(self, name: str, k: int) -> None:
        n = len(self.rr)
        if not n:
            return
        print(f"\n=== Ergebnisse (n={n}, k={k}, {name}) ===")
        for kk in sorted(self.hits_at):
            print(f"  Hit-Rate@{kk}: {self.hits_at[kk] / n:.2f}  ({self.hits_at[kk]}/{n})")
        print(f"  MRR:         {sum(self.rr) / n:.3f}")


async def evaluate(gold_path: str, group_id: str | None, k: int, embed_model: str) -> None:
    gold = [json.loads(line) for line in Path(gold_path).read_text().splitlines() if line.strip()]
    # unanswerable-Items (v2) messen Antwort-Ehrlichkeit, nicht Retrieval.
    gold = [g for g in gold if g.get("qtype") != "unanswerable"]

    ollama = OllamaClient(settings.ollama_base_url)
    vectors = VectorStore(settings.qdrant_url, settings.qdrant_collection)

    chapter_arm = _Arm(k)
    span_arm = _Arm(k)

    try:
        for item in gold:
            points = await retrieve(
                item["question"],
                ollama=ollama,
                vectors=vectors,
                embed_model=embed_model,
                top_k=k,
                group_id=group_id,
            )
            payloads = [p.payload or {} for p in points]

            # Kapitel-Arm (v1-Labels; in v2 als Vergleichs-Feld `chapters` weitergeführt).
            chapters = item.get("relevant_chapters") or item.get("chapters") or []
            first_ch = None
            if chapters:
                relevant = {r.upper() for r in chapters}
                flags = [chapter_roman(pl.get("chapter")) in relevant for pl in payloads]
                first_ch = chapter_arm.add(flags)

            # Span-Arm (v2-Labels).
            first_sp = None
            if item.get("spans"):
                flags = [span_hit(pl, item) for pl in payloads]
                first_sp = span_arm.add(flags)

            def fmt(first: int | None, present: bool) -> str:
                return ("—" if not present else f"@{first}" if first else "MISS")

            print(f"  {item['id']}: kapitel={fmt(first_ch, bool(chapters)):>5}  "
                  f"span={fmt(first_sp, bool(item.get('spans'))):>5}")
    finally:
        await ollama.aclose()
        await vectors.aclose()

    chapter_arm.report("kapitel-basiert", k)
    span_arm.report("span-basiert (v2)", k)
    if chapter_arm.rr and span_arm.rr and len(chapter_arm.rr) == len(span_arm.rr):
        diff = sum(c - s for c, s in zip(chapter_arm.rr, span_arm.rr)) / len(span_arm.rr)
        print(f"\n  Ø MRR-Differenz Kapitel−Span: {diff:+.3f} "
              f"(positiv = Kapitel-Labels überschätzen das Retrieval)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(Path(__file__).resolve().parent / "gold_dracula.jsonl"))
    ap.add_argument("--group", default="Horror")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--embed-model", default=settings.embed_model)
    args = ap.parse_args()
    asyncio.run(evaluate(args.gold, args.group, args.k, args.embed_model))


if __name__ == "__main__":
    main()
