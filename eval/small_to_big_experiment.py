"""Exp 7 — Small-to-Big: deckt der Prompt-Kontext die Gold-Passage ab?

Small-to-Big ändert NICHT, welche Chunks gefunden werden (die Suche bleibt gleich),
sondern was das Modell davon zu lesen bekommt. Deshalb misst dieses Experiment die
**Span-Abdeckung des Prompt-Kontexts**: Gilt ein Treffer als Hit, wenn sein (ggf.
erweiterter) Zeichenbereich einen Gold-Span überlappt? Dazu die Prompt-Kosten
(Ø Kontext-Zeichen), denn die Erweiterung ist nicht gratis.

Beide Arme nutzen exakt dieselben Suchergebnisse (paired): der expanded-Arm
erweitert sie lediglich per app.rag.expander.

Aufruf (aus dem backend-Verzeichnis):
  uv run python ../eval/small_to_big_experiment.py --gold ../eval/gold_v2.jsonl \
      --group Horror --k 8 --window 1 --top-n 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.clients.ollama_client import OllamaClient  # noqa: E402
from app.clients.qdrant_client import VectorStore  # noqa: E402
from app.config import settings  # noqa: E402
from app.rag.expander import expand_points  # noqa: E402
from app.rag.retriever import retrieve  # noqa: E402
from retrieval_eval import span_hit  # noqa: E402


def _first_hit(payloads: list[dict], item: dict) -> int | None:
    for i, pl in enumerate(payloads, start=1):
        if span_hit(pl, item):
            return i
    return None


def _ctx_chars(payloads: list[dict]) -> int:
    return sum(len(pl.get("text") or "") for pl in payloads)


class _Arm:
    def __init__(self, name: str) -> None:
        self.name = name
        self.first: list[int | None] = []
        self.chars: list[int] = []

    def add(self, payloads: list[dict], item: dict) -> int | None:
        first = _first_hit(payloads, item)
        self.first.append(first)
        self.chars.append(_ctx_chars(payloads))
        return first

    def report(self, k: int) -> None:
        n = len(self.first)
        hits = sum(1 for f in self.first if f)
        mrr = sum(1.0 / f for f in self.first if f) / n
        print(f"\n=== {self.name} (n={n}) ===")
        print(f"  Kontext-Hit@{k}: {hits / n:.2f}  ({hits}/{n})")
        print(f"  MRR (erster abdeckender Treffer): {mrr:.3f}")
        print(f"  Ø Prompt-Kontext: {sum(self.chars) / n:,.0f} Zeichen")


async def run(gold_path: str, group_id: str | None, k: int, window: int, top_n: int) -> None:
    gold = [json.loads(line) for line in Path(gold_path).read_text().splitlines() if line.strip()]
    gold = [g for g in gold if g.get("qtype") != "unanswerable" and g.get("spans")]

    ollama = OllamaClient(settings.ollama_base_url)
    vectors = VectorStore(settings.qdrant_url, settings.qdrant_collection)
    plain_arm, exp_arm = _Arm("plain"), _Arm(f"expanded (window={window}, top_n={top_n})")
    better = worse = same = 0

    try:
        for item in gold:
            points = await retrieve(
                item["question"], ollama=ollama, vectors=vectors,
                embed_model=settings.embed_model, top_k=k, group_id=group_id,
            )
            expanded = await expand_points(points, vectors=vectors, window=window, top_n=top_n)

            f_plain = plain_arm.add([p.payload or {} for p in points], item)
            f_exp = exp_arm.add([p.payload or {} for p in expanded], item)

            # Paired-Vergleich über den Rang des ersten abdeckenden Treffers.
            rp = 1.0 / f_plain if f_plain else 0.0
            re_ = 1.0 / f_exp if f_exp else 0.0
            better += re_ > rp
            worse += re_ < rp
            same += re_ == rp

            def fmt(f: int | None) -> str:
                return f"@{f}" if f else "MISS"

            print(f"  {item['id']}: plain={fmt(f_plain):>5}  expanded={fmt(f_exp):>5}")
    finally:
        await ollama.aclose()
        await vectors.aclose()

    plain_arm.report(k)
    exp_arm.report(k)
    print(f"\n  Paired (Rang des ersten Hits): {better} besser / {worse} schlechter / {same} gleich")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(Path(__file__).resolve().parent / "gold_v2.jsonl"))
    ap.add_argument("--group", default="Horror")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--top-n", type=int, default=3)
    args = ap.parse_args()
    asyncio.run(run(args.gold, args.group, args.k, args.window, args.top_n))


if __name__ == "__main__":
    main()
