"""Exp 8 — Graph-RAG lokal: hilft ein Entity-Graph dem dense Retrieval auf der Span-Metrik?

Arme (paired, dieselben 32 beantwortbaren Gold-v2-Fragen, k=8, Gruppe Horror):
  dense      — heutige Pipeline (app.rag.retriever.retrieve, contextual + dedup) = Baseline
  link       — Entity-Linking DE-Frage -> EN-Entities (bge-m3) -> 1-Hop -> Fusion
  expand     — dense-first, Seeds aus Top-3, Ko-Erwähnung >= 2 -> Fusion
  graph_only — Diagnose: Ranking rein nach Graph-Signal (analog `sparse` solo, Exp 6)
Fusion: s = cos + alpha * g̃ (dense-erhaltend; alpha=0 == dense -> Sanity-Check).

Hypothese (vorab, s. RESULTS.md Exp 8): Graph holt beiläufig erwähnte Passagen in die
Top-8 (Hit@8 rauf, MRR nicht runter), Effekt v. a. bei multi/paraphrase.
Erfolg: paired besser >= 2*schlechter UND ΔMRR >= +0.03 UND Hit@8 nicht schlechter.

Metriken: Hit@1/3/5/k, MRR (erster Span-Treffer), Cov@k (Anteil der Gold-Spans, die
ein Top-k-Chunk abdeckt — sieht bei multi-Items die zweite Passage), paired Bilanz +
Vorzeichentest, Tabelle je qtype, Sensitivität (post hoc markiert).

Aufruf (aus dem backend-Verzeichnis; Graph vorher mit build_cli gebaut):
  uv run --with matplotlib python ../eval/graph_experiment.py --gold ../eval/gold_v2.jsonl \
      --group Horror --k 8
  uv run python ../eval/graph_experiment.py --alpha 0 --no-sensitivity   # muss == dense sein
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.clients.ollama_client import OllamaClient  # noqa: E402
from app.clients.qdrant_client import VectorStore  # noqa: E402
from app.config import settings  # noqa: E402
from app.graph.builder import stats  # noqa: E402
from app.graph.retriever import GraphParams, chunk_key, graph_retrieve  # noqa: E402
from app.graph.store import KnowledgeGraph  # noqa: E402
from app.rag.retriever import embed_question, retrieve  # noqa: E402
from retrieval_eval import span_hit  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def first_hit(payloads: list[dict], item: dict) -> int | None:
    for i, pl in enumerate(payloads, start=1):
        if span_hit(pl, item):
            return i
    return None


def coverage(payloads: list[dict], item: dict) -> float:
    """Anteil der Gold-Spans, die von mindestens einem Chunk überlappt werden."""
    spans = item.get("spans") or []
    if not spans:
        return 0.0
    covered = 0
    for s in spans:
        one = {"spans": [s], "book_id": item.get("book_id")}
        if any(span_hit(pl, one) for pl in payloads):
            covered += 1
    return covered / len(spans)


class Arm:
    def __init__(self, name: str, k: int) -> None:
        self.name, self.k = name, k
        self.first: list[int | None] = []
        self.cov: list[float] = []
        self.qtypes: list[str] = []

    def add(self, payloads: list[dict], item: dict) -> int | None:
        f = first_hit(payloads, item)
        self.first.append(f)
        self.cov.append(coverage(payloads, item))
        self.qtypes.append(item.get("qtype", "fact"))
        return f

    def _idx(self, qtype: str | None) -> list[int]:
        return [i for i, q in enumerate(self.qtypes) if qtype is None or q == qtype]

    def hit(self, kk: int, qtype: str | None = None) -> float:
        idx = self._idx(qtype)
        return sum(1 for i in idx if self.first[i] and self.first[i] <= kk) / max(len(idx), 1)

    def mrr(self, qtype: str | None = None) -> float:
        idx = self._idx(qtype)
        return sum(1.0 / self.first[i] for i in idx if self.first[i]) / max(len(idx), 1)

    def cov_mean(self, qtype: str | None = None) -> float:
        idx = self._idx(qtype)
        return sum(self.cov[i] for i in idx) / max(len(idx), 1)

    def summary(self) -> dict:
        return {"hit@1": round(self.hit(1), 3), "hit@3": round(self.hit(3), 3),
                "hit@5": round(self.hit(5), 3), f"hit@{self.k}": round(self.hit(self.k), 3),
                "mrr": round(self.mrr(), 3), f"cov@{self.k}": round(self.cov_mean(), 3),
                "n": len(self.first)}


def paired(base: Arm, other: Arm) -> tuple[int, int, int, float]:
    """(besser, schlechter, gleich, p Vorzeichentest zweiseitig) über den Rang des ersten Hits."""
    better = worse = same = 0
    for fb, fo in zip(base.first, other.first, strict=True):
        rb = 1.0 / fb if fb else 0.0
        ro = 1.0 / fo if fo else 0.0
        better += ro > rb
        worse += ro < rb
        same += ro == rb
    n = better + worse
    if n == 0:
        return better, worse, same, 1.0
    m = min(better, worse)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2**n)
    return better, worse, same, p


def fmt(f: int | None) -> str:
    return f"@{f}" if f else "MISS"


async def run(args: argparse.Namespace) -> None:
    gold = [json.loads(x) for x in Path(args.gold).read_text().splitlines() if x.strip()]
    gold = [g for g in gold if g.get("qtype") != "unanswerable" and g.get("spans")]

    kg = KnowledgeGraph.load(Path(args.graph_dir) / args.group)
    if kg.entity_vectors is None:
        sys.exit("Graph ohne Entity-Vektoren — build_cli ohne --no-embed laufen lassen.")
    st = stats(kg)
    print(f"Graph {args.group}: {st['n_nodes']} Knoten, {st['n_edges']} Kanten, "
          f"{kg.n_chunks} Chunks, Modell {kg.model}")
    print("  Top-Grad: " + ", ".join(f"{n} ({d})" for n, d in st["top_degree"][:6]))

    base_params = GraphParams(alpha=args.alpha, top_m=args.top_m, min_sim=args.min_sim)
    arms: dict[str, tuple[str, GraphParams]] = {
        "link": ("link", base_params),
        "expand": ("expand", base_params),
        "graph_only": ("graph_only", base_params),
    }
    if not args.no_sensitivity:  # post hoc — im Bericht als solche markieren
        for a in (0.02, 0.05):
            arms[f"link a={a}"] = ("link", replace(base_params, alpha=a))
        for s in (0.40, 0.50):
            arms[f"link τ={s}"] = ("link", replace(base_params, min_sim=s))

    ollama = OllamaClient(settings.ollama_base_url)
    vectors = VectorStore(settings.qdrant_url, settings.qdrant_collection)
    dense_arm = Arm("dense", args.k)
    graph_arms = {name: Arm(name, args.k) for name in arms}
    items_out: list[dict] = []

    try:
        for n, item in enumerate(gold, start=1):
            qvec = await embed_question(item["question"], ollama=ollama, embed_model=settings.embed_model)
            dense = await retrieve(
                item["question"], ollama=ollama, vectors=vectors,
                embed_model=settings.embed_model, top_k=args.k, group_id=args.group,
            )
            # Der zweimal aufgetretene Eval-Bug (Exp 2/6): dense ohne Gruppenfilter.
            assert all((p.payload or {}).get("group_id") == args.group for p in dense), \
                "dense-Arm ohne Gruppenfilter!"
            dense_keys = [chunk_key(p.payload) for p in dense]
            f_dense = dense_arm.add([p.payload or {} for p in dense], item)
            row = {"id": item["id"], "qtype": item.get("qtype", "fact"), "dense": f_dense}

            if n <= 3:  # Sanity: alpha=0 muss die Baseline exakt reproduzieren
                zero = await graph_retrieve(
                    item["question"], kg=kg, mode="link", ollama=ollama, vectors=vectors,
                    embed_model=settings.embed_model, top_k=args.k, group_id=args.group,
                    params=replace(base_params, alpha=0.0), qvec=qvec,
                )
                assert [chunk_key(p.payload) for p in zero] == dense_keys, "alpha=0 != dense"

            diag: dict = {}
            line = f"  {item['id']} [{row['qtype']:>10}] dense={fmt(f_dense):>5}"
            for name, (mode, params) in arms.items():
                d: dict = {}
                pts = await graph_retrieve(
                    item["question"], kg=kg, mode=mode, ollama=ollama, vectors=vectors,
                    embed_model=settings.embed_model, top_k=args.k, group_id=args.group,
                    params=params, qvec=qvec, diagnostics=d,
                )
                f = graph_arms[name].add([p.payload or {} for p in pts], item)
                row[name] = f
                if name in ("link", "expand", "graph_only"):
                    line += f"  {name}={fmt(f):>5}"
                if name == "link":
                    diag = d
            row["linked"] = diag.get("linked", [])
            items_out.append(row)
            print(line + "  | " + ", ".join(f"{n_} ({s})" for n_, s in diag.get("linked", [])[:4]))
    finally:
        await ollama.aclose()
        await vectors.aclose()

    # --- Bericht ---------------------------------------------------------------
    k = args.k
    main_arms = [dense_arm] + [graph_arms[n] for n in ("link", "expand", "graph_only")]
    print(f"\n=== Exp 8 — Graph-RAG lokal (n={len(gold)}, k={k}, alpha={args.alpha}, "
          f"m={args.top_m}, τ={args.min_sim}) ===")
    print(f"  {'Arm':<12} {'Hit@1':>6} {'Hit@3':>6} {'Hit@5':>6} {'Hit@'+str(k):>6} {'MRR':>6} {'Cov@'+str(k):>6}")
    for arm in main_arms:
        print(f"  {arm.name:<12} {arm.hit(1):6.2f} {arm.hit(3):6.2f} {arm.hit(5):6.2f} "
              f"{arm.hit(k):6.2f} {arm.mrr():6.3f} {arm.cov_mean():6.2f}")
    paired_out = {}
    for name in ("link", "expand"):
        b, w, s, p = paired(dense_arm, graph_arms[name])
        paired_out[name] = {"better": b, "worse": w, "same": s, "p_sign": round(p, 3)}
        print(f"  Paired {name} vs dense: {b} besser / {w} schlechter / {s} gleich "
              f"(Vorzeichentest p={p:.3f}, ΔMRR={graph_arms[name].mrr() - dense_arm.mrr():+.3f})")

    qtypes = sorted(set(dense_arm.qtypes))
    print(f"\n  Je Fragetyp (Hit@{k} / MRR; n<5 nur anekdotisch):")
    print(f"  {'qtype':<12} {'n':>3}  " + "  ".join(f"{a.name:>14}" for a in main_arms[:3]))
    for qt in qtypes:
        n_qt = dense_arm.qtypes.count(qt)
        print(f"  {qt:<12} {n_qt:>3}  " + "  ".join(
            f"{a.hit(k, qt):5.2f} / {a.mrr(qt):5.3f}" for a in main_arms[:3]))

    if not args.no_sensitivity:
        print("\n  Sensitivität `link` (POST HOC, n=32 — nur Orientierung, keine Auswahl):")
        for name, arm in graph_arms.items():
            if name.startswith("link "):
                b, w, s, _ = paired(dense_arm, arm)
                print(f"    {name:<12} Hit@{k}={arm.hit(k):.2f}  MRR={arm.mrr():.3f}  "
                      f"paired {b}/{w}/{s}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "group": args.group, "k": k, "n": len(gold),
        "params": {"alpha": args.alpha, "top_m": args.top_m, "min_sim": args.min_sim},
        "graph": {"n_nodes": st["n_nodes"], "n_edges": st["n_edges"], "n_chunks": kg.n_chunks},
        "arms": {a.name: a.summary() for a in main_arms},
        "sensitivity": {n: a.summary() for n, a in graph_arms.items() if n.startswith("link ")},
        "paired_vs_dense": paired_out,
        "items": items_out,
    }
    (RESULTS_DIR / "graph_experiment.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  JSON: {RESULTS_DIR / 'graph_experiment.json'}")
    _chart(main_arms, k)


def _chart(arms: list[Arm], k: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (kein matplotlib — Chart übersprungen; `uv run --with matplotlib …`)")
        return
    metrics = [("Hit@1", lambda a: a.hit(1)), (f"Hit@{k}", lambda a: a.hit(k)),
               ("MRR", lambda a: a.mrr()), (f"Cov@{k}", lambda a: a.cov_mean())]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        xs = [j + i * width for j in range(len(metrics))]
        ys = [f(arm) for _, f in metrics]
        bars = ax.bar(xs, ys, width, label=arm.name)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks([j + width * (len(arms) - 1) / 2 for j in range(len(metrics))])
    ax.set_xticklabels([m for m, _ in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Exp 8 — Graph-RAG lokal vs. dense (span-basiert, n={len(arms[0].first)}, k={k})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = RESULTS_DIR / "graph_experiment.png"
    fig.savefig(path, dpi=150)
    print(f"  Chart: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(Path(__file__).resolve().parent / "gold_v2.jsonl"))
    ap.add_argument("--group", default="Horror")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--graph-dir", default=settings.graph_dir)
    ap.add_argument("--alpha", type=float, default=settings.graph_alpha)
    ap.add_argument("--top-m", type=int, default=settings.graph_top_m)
    ap.add_argument("--min-sim", type=float, default=settings.graph_min_sim)
    ap.add_argument("--no-sensitivity", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
