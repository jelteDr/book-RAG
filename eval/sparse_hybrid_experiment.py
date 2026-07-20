"""Exp 6 — Sparse-Hybrid mit bge-m3-eigenen Sparse-Gewichten (statt BM25).

Exp 2 zeigte: klassisches BM25 ist cross-lingual gehandicapt (DE-Frage, EN-Text).
bge-m3 liefert neben dem Dense-Vektor auch **lexikalische Gewichte** über sein
multilinguales Subword-Vokabular ("learned sparse") — die Hoffnung: lexikalische
Präzision ohne den harten Sprachbruch. Gemessen wird gegen den LIVE-Index
(Horror = contextual embeddings), die eigentliche Frage lautet also:
bringt Sparse ZUSÄTZLICH zu Contextual-Dense noch etwas?

Arme (alle span-basiert, rohe Suche ohne Dedup, paired auf denselben Fragen):
  dense      – Live-Qdrant-Suche (bge-m3 dense via Ollama, contextual)
  sparse     – lexikalische bge-m3-Gewichte (FlagEmbedding, CPU) über alle Chunks
  rrf        – Reciprocal-Rank-Fusion (k=60) aus dense@50 + sparse@50
  sparse_en  – wie sparse, aber mit `question_en` (nur Items, die eins haben):
               trennt "Sparse ist schwach" von "Sparse ist nur cross-lingual schwach"

Die Sparse-Gewichte der Chunks werden gecacht (../results/sparse_weights_<group>.jsonl
— Buchtext-Derivat, gitignored, NICHT committen).

Aufruf (aus dem backend-Verzeichnis; lädt beim ersten Mal BAAI/bge-m3, ~2,3 GB):
  uv run --with FlagEmbedding python ../eval/sparse_hybrid_experiment.py \
      --gold ../eval/gold_v2.jsonl --group Horror --k 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from qdrant_client import models  # noqa: E402

from app.clients.ollama_client import OllamaClient  # noqa: E402
from app.clients.qdrant_client import VectorStore  # noqa: E402
from app.config import settings  # noqa: E402
from app.util import l2_normalize  # noqa: E402
from retrieval_eval import span_hit  # noqa: E402

RRF_K = 60
CAND = 50  # Kandidaten je Arm für die Fusion


def _key(payload: dict) -> str:
    return f"{payload['book_id']}:{payload['chunk_index']}"


def load_or_encode_sparse(model, records, cache_path: Path) -> dict[str, dict[str, float]]:
    """Lexikalische Gewichte je Chunk — aus dem Cache oder frisch encodiert (CPU)."""
    cache: dict[str, dict[str, float]] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["key"]] = row["weights"]
    todo = [r for r in records if _key(r.payload) not in cache]
    print(f"Sparse-Gewichte: {len(cache)} aus Cache, {len(todo)} zu encodieren")
    if todo:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as f:
            for start in range(0, len(todo), 16):
                part = todo[start : start + 16]
                out = model.encode(
                    [r.payload["text"] for r in part],
                    return_dense=False, return_sparse=True, return_colbert_vecs=False,
                    max_length=1024, batch_size=16,
                )
                for rec, weights in zip(part, out["lexical_weights"], strict=True):
                    w = {tok: float(v) for tok, v in weights.items()}
                    cache[_key(rec.payload)] = w
                    f.write(json.dumps({"key": _key(rec.payload), "weights": w}) + "\n")
                done = min(start + 16, len(todo))
                if done % 80 == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)} encodiert")
    return cache


def sparse_scores(query_weights: dict[str, float], chunk_weights: dict[str, dict]) -> dict[str, float]:
    """Skalarprodukt der lexikalischen Gewichte (nur gemeinsame Tokens zählen)."""
    scores: dict[str, float] = {}
    for key, cw in chunk_weights.items():
        s = sum(qv * cw[t] for t, qv in query_weights.items() if t in cw)
        if s > 0:
            scores[key] = s
    return scores


def rrf_fuse(rankings: list[list[str]]) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] += 1.0 / (RRF_K + rank)
    return sorted(scores, key=scores.get, reverse=True)


class _Arm:
    def __init__(self, name: str) -> None:
        self.name = name
        self.first: list[int | None] = []

    def add(self, payloads: list[dict], item: dict) -> int | None:
        first = next(
            (i + 1 for i, pl in enumerate(payloads) if span_hit(pl, item)), None
        )
        self.first.append(first)
        return first

    def report(self, k: int) -> None:
        n = len(self.first)
        if not n:
            return
        hit1 = sum(1 for f in self.first if f == 1)
        hitk = sum(1 for f in self.first if f)
        mrr = sum(1.0 / f for f in self.first if f) / n
        print(f"  {self.name:<10} Hit@1 {hit1 / n:.2f}   Hit@{k} {hitk / n:.2f} "
              f"({hitk}/{n})   MRR {mrr:.3f}")


async def run(gold_path: str, group_id: str, k: int) -> None:
    gold = [json.loads(x) for x in Path(gold_path).read_text().splitlines() if x.strip()]
    gold = [g for g in gold if g.get("qtype") != "unanswerable" and g.get("spans")]

    from FlagEmbedding import BGEM3FlagModel

    print("Lade BAAI/bge-m3 (CPU) …")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

    ollama = OllamaClient(settings.ollama_base_url)
    vectors = VectorStore(settings.qdrant_url, settings.qdrant_collection)
    try:
        records = await vectors.scroll_all(group_id)
        by_key = {_key(r.payload): r.payload for r in records}
        print(f"{len(records)} Chunks in Gruppe {group_id}")
        cache_path = Path(__file__).resolve().parent.parent / "results" / f"sparse_weights_{group_id}.jsonl"
        chunk_weights = load_or_encode_sparse(model, records, cache_path)

        arms = {n: _Arm(n) for n in ("dense", "sparse", "rrf")}
        arm_en = _Arm("sparse_en")
        better = worse = same = 0

        for item in gold:
            # dense: exakt der Live-Pfad (Ollama-Embedding, contextual Index).
            # WICHTIG: mit group-Filter — ohne ihn verwässern fremde Gruppen die
            # Kandidaten (derselbe latente Bug wie im Exp-2-dense-Arm!).
            q_emb = l2_normalize((await ollama.embed([item["question"]], settings.embed_model))[0])
            group_filter = models.Filter(
                must=[models.FieldCondition(key="group_id", match=models.MatchValue(value=group_id))]
            )
            dense_pts = await vectors.search(q_emb, CAND, group_filter)
            dense_keys = [_key(p.payload) for p in dense_pts if p.payload]

            q_sparse = model.encode(
                [item["question"]], return_dense=False, return_sparse=True,
                return_colbert_vecs=False, max_length=256,
            )["lexical_weights"][0]
            q_sparse = {t: float(v) for t, v in q_sparse.items()}
            s_scores = sparse_scores(q_sparse, chunk_weights)
            sparse_keys = sorted(s_scores, key=s_scores.get, reverse=True)[:CAND]

            fused_keys = rrf_fuse([dense_keys, sparse_keys])

            firsts = {}
            for name, keys in (("dense", dense_keys), ("sparse", sparse_keys), ("rrf", fused_keys)):
                firsts[name] = arms[name].add([by_key[kk] for kk in keys[:k]], item)

            if item.get("question_en"):
                qe = model.encode(
                    [item["question_en"]], return_dense=False, return_sparse=True,
                    return_colbert_vecs=False, max_length=256,
                )["lexical_weights"][0]
                se = sparse_scores({t: float(v) for t, v in qe.items()}, chunk_weights)
                keys_en = sorted(se, key=se.get, reverse=True)[:k]
                arm_en.add([by_key[kk] for kk in keys_en], item)

            rd = 1.0 / firsts["dense"] if firsts["dense"] else 0.0
            rf = 1.0 / firsts["rrf"] if firsts["rrf"] else 0.0
            better += rf > rd
            worse += rf < rd
            same += rf == rd

            def fmt(f: int | None) -> str:
                return f"@{f}" if f else "MISS"

            print(f"  {item['id']}: dense={fmt(firsts['dense']):>5}  "
                  f"sparse={fmt(firsts['sparse']):>5}  rrf={fmt(firsts['rrf']):>5}")
    finally:
        await ollama.aclose()
        await vectors.aclose()

    print(f"\n=== Ergebnisse (n={len(gold)}, k={k}, span-basiert, Gruppe {group_id}) ===")
    for arm in arms.values():
        arm.report(k)
    arm_en.report(k)
    print(f"\n  Paired rrf vs. dense (MRR-Beitrag): {better} besser / {worse} schlechter / {same} gleich")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(Path(__file__).resolve().parent / "gold_v2.jsonl"))
    ap.add_argument("--group", default="Horror")
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    asyncio.run(run(args.gold, args.group, args.k))


if __name__ == "__main__":
    main()
