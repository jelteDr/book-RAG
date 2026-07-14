"""Before/After-Experiment 2: Dense vs. BM25 vs. Hybrid (RRF).

Motivation aus Experiment 1: der Flaschenhals sind Eigennamen/Tail-Treffer, nicht
die Sprache. BM25 (lexikalisch) matcht Eigennamen wie "Harker", "Demeter",
"Van Helsing" direkt — auch bei deutscher Frage. Wir fusionieren Dense (bge-m3)
und BM25 per Reciprocal Rank Fusion (RRF) und vergleichen Recall@k/MRR.

Aufruf (aus backend/):
  uv run --with rank_bm25 --with matplotlib python ../eval/hybrid_experiment.py
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
from app.rag.retriever import retrieve  # noqa: E402
from retrieval_eval import chapter_roman  # noqa: E402

_TOKEN_RE = re.compile(r"[a-zA-ZäöüÄÖÜß0-9]+")
CAND = 30          # Kandidaten pro Methode
RRF_K = 60         # RRF-Konstante


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]


def hit_rate_mrr(ranked_chapters: list[str | None], relevant: set[str], ks=(1, 3, 5, 8)) -> tuple[dict, float]:
    # HINWEIS: Dies ist Hit-Rate@k (Success@k) — 1.0, sobald IRGENDEIN relevantes
    # Kapitel in Top-k liegt (nicht echtes Recall über alle Gold-Kapitel).
    hits = [i + 1 for i, c in enumerate(ranked_chapters) if c in relevant]
    first = hits[0] if hits else None
    recall = {k: 1.0 if any(c in relevant for c in ranked_chapters[:k]) else 0.0 for k in ks}
    return recall, (1.0 / first if first else 0.0)


def rrf_fuse(*ranked_id_lists: list[str]) -> list[str]:
    scores: dict[str, float] = {}
    for ids in ranked_id_lists:
        for rank, _id in enumerate(ids, start=1):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(Path(__file__).resolve().parent / "gold_dracula.jsonl"))
    ap.add_argument("--group", default="horror-classics")
    args = ap.parse_args()

    from rank_bm25 import BM25Okapi

    gold = [json.loads(x) for x in Path(args.gold).read_text().splitlines() if x.strip()]
    ollama = OllamaClient(settings.ollama_base_url)
    vectors = VectorStore(settings.qdrant_url, settings.qdrant_collection)

    # Korpus für BM25 + id->Kapitel-Map.
    records = await vectors.scroll_all(args.group)
    ids = [str(r.id) for r in records]
    id_to_chapter = {str(r.id): chapter_roman((r.payload or {}).get("chapter")) for r in records}
    bm25 = BM25Okapi([tokenize((r.payload or {}).get("text", "")) for r in records])

    methods = ("dense", "bm25", "hybrid")
    agg = {m: {"recall": {1: 0.0, 3: 0.0, 5: 0.0, 8: 0.0}, "mrr": 0.0} for m in methods}
    firsts = {m: {} for m in methods}

    try:
        for item in gold:
            relevant = {r.upper() for r in item["relevant_chapters"]}

            # Dense (identisch zum App-Retriever, inkl. Gruppen-Filter — sonst würden
            # bei mehreren Büchern Fremd-Chunks in die Top-N geraten).
            dense_pts = await retrieve(
                item["question"], ollama=ollama, vectors=vectors,
                embed_model=settings.embed_model, top_k=CAND, group_id=args.group,
            )
            dense_ids = [str(p.id) for p in dense_pts]

            # BM25.
            scores = bm25.get_scores(tokenize(item["question"]))
            bm25_ids = [ids[i] for i in sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:CAND]]

            # Hybrid (RRF).
            hybrid_ids = rrf_fuse(dense_ids, bm25_ids)[:CAND]

            for m, id_list in (("dense", dense_ids), ("bm25", bm25_ids), ("hybrid", hybrid_ids)):
                chapters = [id_to_chapter.get(i) for i in id_list]
                hr, rr = hit_rate_mrr(chapters, relevant)
                for k in agg[m]["recall"]:
                    agg[m]["recall"][k] += hr[k]
                agg[m]["mrr"] += rr
                pos = [i + 1 for i, c in enumerate(chapters) if c in relevant]
                firsts[m][item["id"]] = pos[0] if pos else None
    finally:
        await ollama.aclose()
        await vectors.aclose()

    n = len(gold)
    for m in methods:
        for k in agg[m]["recall"]:
            agg[m]["recall"][k] /= n
        agg[m]["mrr"] /= n

    print(f"{'Metrik':<12}{'dense':>9}{'bm25':>9}{'hybrid':>9}   (Hit-Rate@k = Success@k)")
    print("-" * 39)
    for k in (1, 3, 5, 8):
        row = "".join(f"{agg[m]['recall'][k]:>9.2f}" for m in methods)
        print(f"{'HitRate@' + str(k):<12}{row}")
    print(f"{'MRR':<11}" + "".join(f"{agg[m]['mrr']:>9.3f}" for m in methods))

    print("\nPro Frage — Rang des ersten Treffers (- = MISS):")
    print(f"  {'id':<5}{'dense':>7}{'bm25':>7}{'hybrid':>7}")
    for item in gold:
        i = item["id"]
        vals = "".join(f"{str(firsts[m][i] or '-'):>7}" for m in methods)
        print(f"  {i:<5}{vals}")

    _plot(agg, methods, Path(__file__).resolve().parent.parent / "results" / "hybrid_experiment.png")


def _plot(agg: dict, methods, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib fehlt — kein Chart)")
        return
    labels = ["HitRate@1", "HitRate@3", "HitRate@5", "HitRate@8", "MRR"]
    colors = {"dense": "#6ea8fe", "bm25": "#f0a868", "hybrid": "#7ed99f"}
    x = range(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for j, m in enumerate(methods):
        vals = [agg[m]["recall"][1], agg[m]["recall"][3], agg[m]["recall"][5], agg[m]["recall"][8], agg[m]["mrr"]]
        ax.bar([i + (j - 1) * w for i in x], vals, w, label=m, color=colors[m])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Wert")
    ax.set_title("Retrieval: Dense vs. BM25 vs. Hybrid (RRF) — bge-m3, Dracula, DE-Fragen")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nChart gespeichert: {out}")


if __name__ == "__main__":
    asyncio.run(main())
