"""Before/After-Experiment: Retrieval-Qualität bei deutscher vs. englischer Frage.

Hypothese: Die mäßige Baseline (Recall@5=0.80, MRR=0.517) liegt am cross-lingualen
Gap (DE-Frage → EN-Text), nicht an der Chunk-Größe. Test: dieselben Gold-Fragen
einmal auf Deutsch (`question`), einmal auf Englisch (`question_en`) retrieven und
Recall@k/MRR vergleichen. Kein Re-Embedding der Chunks nötig — nur die Query.

Aufruf (aus backend/, matplotlib ephemer via uv):
  uv run --with matplotlib python ../eval/lang_experiment.py
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


async def eval_field(gold, field, group, k, embed_model, ollama, vectors) -> dict:
    recall = {1: 0, 3: 0, 5: 0, k: 0}
    rr: list[float] = []
    firsts: dict[str, int | None] = {}
    for item in gold:
        relevant = {r.upper() for r in item["relevant_chapters"]}
        points = await retrieve(
            item[field], ollama=ollama, vectors=vectors,
            embed_model=embed_model, top_k=k, group_id=group,
        )
        ranks = [chapter_roman((p.payload or {}).get("chapter")) for p in points]
        hits = [i + 1 for i, r in enumerate(ranks) if r in relevant]
        first = hits[0] if hits else None
        firsts[item["id"]] = first
        rr.append(1.0 / first if first else 0.0)
        for kk in recall:
            if any(r in relevant for r in ranks[:kk]):
                recall[kk] += 1
    n = len(gold)
    return {"recall": {kk: recall[kk] / n for kk in recall}, "mrr": sum(rr) / n, "firsts": firsts}


def plot(de: dict, en: dict, k: int, out: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    labels = ["Recall@1", "Recall@3", "Recall@5", f"Recall@{k}", "MRR"]
    de_vals = [de["recall"][1], de["recall"][3], de["recall"][5], de["recall"][k], de["mrr"]]
    en_vals = [en["recall"][1], en["recall"][3], en["recall"][5], en["recall"][k], en["mrr"]]
    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - w / 2 for i in x], de_vals, w, label="Frage DE", color="#6ea8fe")
    ax.bar([i + w / 2 for i in x], en_vals, w, label="Frage EN", color="#f0a868")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Wert")
    ax.set_title("Retrieval: deutsche vs. englische Frage (bge-m3, Dracula)")
    ax.legend()
    for i, (d, e) in enumerate(zip(de_vals, en_vals)):
        ax.text(i - w / 2, d + 0.02, f"{d:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, e + 0.02, f"{e:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    return True


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(Path(__file__).resolve().parent / "gold_dracula.jsonl"))
    ap.add_argument("--group", default="Horror")
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    gold = [json.loads(x) for x in Path(args.gold).read_text().splitlines() if x.strip()]
    ollama = OllamaClient(settings.ollama_base_url)
    vectors = VectorStore(settings.qdrant_url, settings.qdrant_collection)
    try:
        de = await eval_field(gold, "question", args.group, args.k, settings.embed_model, ollama, vectors)
        en = await eval_field(gold, "question_en", args.group, args.k, settings.embed_model, ollama, vectors)
    finally:
        await ollama.aclose()
        await vectors.aclose()

    print(f"{'Metrik':<12}{'DE':>8}{'EN':>8}{'Δ':>8}")
    print("-" * 36)
    for kk in (1, 3, 5, args.k):
        d, e = de["recall"][kk], en["recall"][kk]
        print(f"{'Recall@' + str(kk):<12}{d:>8.2f}{e:>8.2f}{e - d:>+8.2f}")
    print(f"{'MRR':<12}{de['mrr']:>8.3f}{en['mrr']:>8.3f}{en['mrr'] - de['mrr']:>+8.3f}")

    print("\nPro Frage (Rang des ersten Treffers, - = MISS):")
    print(f"  {'id':<5}{'DE':>5}{'EN':>5}")
    for item in gold:
        i = item["id"]
        d = de["firsts"][i] or "-"
        e = en["firsts"][i] or "-"
        print(f"  {i:<5}{str(d):>5}{str(e):>5}")

    out = Path(__file__).resolve().parent.parent / "results" / "lang_experiment.png"
    if plot(de, en, args.k, out):
        print(f"\nChart gespeichert: {out}")
    else:
        print("\n(matplotlib nicht verfügbar — kein Chart; mit 'uv run --with matplotlib' ausführen.)")


if __name__ == "__main__":
    asyncio.run(main())
