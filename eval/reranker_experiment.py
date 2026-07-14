"""Before/After-Experiment 3: Dense vs. Dense + Cross-Encoder-Reranker.

Motivation aus Exp 1/2: weder Query-Sprache noch naives BM25/RRF bringen einen
belastbaren Gewinn. Ein Cross-Encoder bewertet (Frage, Passage) gemeinsam und ist
der prinzipielle Qualitäts-Hebel. Ablauf: Dense holt CAND Kandidaten, der Reranker
(bge-reranker-v2-m3, multilingual) sortiert sie neu, dann Hit-Rate@k/MRR vs. Dense.

Aufruf (aus backend/, Modelle/torch werden ephemer gezogen — erster Lauf lädt ~2-3 GB):
  uv run --with sentence-transformers --with matplotlib python ../eval/reranker_experiment.py
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
from app.rag.retriever import retrieve  # noqa: E402
from retrieval_eval import chapter_roman  # noqa: E402

CAND = 30
RERANKER = "BAAI/bge-reranker-v2-m3"


def hit_rate_mrr(chapters: list[str | None], relevant: set[str], ks=(1, 3, 5, 8)):
    # Hit-Rate@k (Success@k): 1.0, sobald ein relevantes Kapitel in Top-k liegt.
    hits = [i + 1 for i, c in enumerate(chapters) if c in relevant]
    first = hits[0] if hits else None
    return {k: 1.0 if any(c in relevant for c in chapters[:k]) else 0.0 for k in ks}, \
        (1.0 / first if first else 0.0), first


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(Path(__file__).resolve().parent / "gold_dracula.jsonl"))
    ap.add_argument("--group", default="horror-classics")
    args = ap.parse_args()

    from sentence_transformers import CrossEncoder

    gold = [json.loads(x) for x in Path(args.gold).read_text().splitlines() if x.strip()]
    print(f"Lade Reranker {RERANKER} … (erster Lauf lädt das Modell)")
    reranker = CrossEncoder(RERANKER, max_length=512)

    ollama = OllamaClient(settings.ollama_base_url)
    vectors = VectorStore(settings.qdrant_url, settings.qdrant_collection)

    methods = ("dense", "reranked")
    agg = {m: {"hr": {1: 0.0, 3: 0.0, 5: 0.0, 8: 0.0}, "mrr": 0.0} for m in methods}
    firsts = {m: {} for m in methods}
    wins = {"dense": 0, "reranked": 0, "tie": 0}

    try:
        for item in gold:
            relevant = {r.upper() for r in item["relevant_chapters"]}
            pts = await retrieve(
                item["question"], ollama=ollama, vectors=vectors,
                embed_model=settings.embed_model, top_k=CAND, group_id=args.group,
            )
            dense_ch = [chapter_roman((p.payload or {}).get("chapter")) for p in pts]

            pairs = [(item["question"], (p.payload or {}).get("text", "")) for p in pts]
            scores = reranker.predict(pairs)
            order = sorted(range(len(pts)), key=lambda i: scores[i], reverse=True)
            rerank_ch = [chapter_roman((pts[i].payload or {}).get("chapter")) for i in order]

            for m, chapters in (("dense", dense_ch), ("reranked", rerank_ch)):
                hr, rr, first = hit_rate_mrr(chapters, relevant)
                for k in agg[m]["hr"]:
                    agg[m]["hr"][k] += hr[k]
                agg[m]["mrr"] += rr
                firsts[m][item["id"]] = first

            dr, rr_ = firsts["dense"][item["id"]], firsts["reranked"][item["id"]]
            drr = 1.0 / dr if dr else 0.0
            rrr = 1.0 / rr_ if rr_ else 0.0
            wins["reranked" if rrr > drr else "dense" if drr > rrr else "tie"] += 1
    finally:
        await ollama.aclose()
        await vectors.aclose()

    n = len(gold)
    for m in methods:
        for k in agg[m]["hr"]:
            agg[m]["hr"][k] /= n
        agg[m]["mrr"] /= n

    print(f"\n{'Metrik':<12}{'dense':>9}{'reranked':>10}{'Δ':>8}   (Hit-Rate@k, n={n})")
    print("-" * 40)
    for k in (1, 3, 5, 8):
        d, r = agg["dense"]["hr"][k], agg["reranked"]["hr"][k]
        print(f"{'HitRate@' + str(k):<12}{d:>9.2f}{r:>10.2f}{r - d:>+8.2f}")
    print(f"{'MRR':<12}{agg['dense']['mrr']:>9.3f}{agg['reranked']['mrr']:>10.3f}{agg['reranked']['mrr'] - agg['dense']['mrr']:>+8.3f}")
    print(f"\nGepaarte Bilanz (erster-Treffer-Rang): reranked {wins['reranked']} : {wins['dense']} dense "
          f"({wins['tie']} unentschieden)")

    print("\nPro Frage — Rang des ersten Treffers (- = MISS):")
    print(f"  {'id':<5}{'dense':>7}{'rerank':>8}")
    for item in gold:
        i = item["id"]
        print(f"  {i:<5}{str(firsts['dense'][i] or '-'):>7}{str(firsts['reranked'][i] or '-'):>8}")

    _plot(agg, Path(__file__).resolve().parent.parent / "results" / "reranker_experiment.png", n)


def _plot(agg: dict, out: Path, n: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = ["HitRate@1", "HitRate@3", "HitRate@5", "HitRate@8", "MRR"]
    dense = [agg["dense"]["hr"][1], agg["dense"]["hr"][3], agg["dense"]["hr"][5], agg["dense"]["hr"][8], agg["dense"]["mrr"]]
    rer = [agg["reranked"]["hr"][1], agg["reranked"]["hr"][3], agg["reranked"]["hr"][5], agg["reranked"]["hr"][8], agg["reranked"]["mrr"]]
    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar([i - w / 2 for i in x], dense, w, label="dense", color="#6ea8fe")
    ax.bar([i + w / 2 for i in x], rer, w, label="dense + Reranker", color="#7ed99f")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels); ax.set_ylim(0, 1.05); ax.set_ylabel("Wert")
    ax.set_title(f"Retrieval: Dense vs. Dense+Cross-Encoder-Reranker (bge-reranker-v2-m3, n={n})")
    ax.legend()
    for i, (d, r) in enumerate(zip(dense, rer)):
        ax.text(i - w / 2, d + 0.02, f"{d:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
    fig.tight_layout(); out.parent.mkdir(exist_ok=True); fig.savefig(out, dpi=120)
    print(f"\nChart gespeichert: {out}")


if __name__ == "__main__":
    asyncio.run(main())
