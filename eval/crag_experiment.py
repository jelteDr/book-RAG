"""Exp 11 — Corrective RAG (LangGraph) vs. dense auf dem Span-Gold-Set (Gold v2).

Arme (gleiches Modell, temp 0, gleicher Retriever: contextual + dedup):
  dense8      Baseline (Default k=8)
  dense4      LÄNGEN-KONTROLLE: CRAG filtert Passagen weg, und Exp 10 zeigte, dass weniger
              Passagen allein schon die Faithfulness heben. CRAG muss also auch gegen
              "einfach k=4" gewinnen, sonst ist der Grader teurer Umweg zum selben Effekt.
  crag        Graph mit Grader + Rewrite-Schleife, ohne Neu-Generierung (max_regen=0)
  crag_regen  wie crag + Faithfulness-Check -> 1x strenge Neu-Generierung. ACHTUNG: der
              Check nutzt dasselbe NLI-Modell wie die Messung — dieser Arm optimiert auf die
              Metrik; seine Faithfulness ist nicht unabhängig (nur als Systemwert berichtet).
  crag_soft   Exp 11b: Grader löst NUR den Retry aus, filtert nicht; in den Prompt gehen die
              ungefilterten Trefferlisten (bei Retry reihum gemischt, max. k). Ohne Retry ist
              der Prompt identisch zu dense8 -> bei temp 0 muss die Antwort gleich sein (Sanity).

Gemessen: Faithfulness (NLI, Sätze vs. Top-5 der Prompt-Passagen), ROUGE-L, Token-F1,
Refusal/False-Refusal, Span-Hit/MRR des FINALEN Prompt-Kontexts, e2e, LLM-Aufrufe;
paired vs. dense8 und dense4 (Vorzeichentest); Grader-Kalibrierung gegen die Span-Labels.

Checkpoint (resumierbar, enthält Antworttexte -> gitignored): results/crag_answers.jsonl
Ergebnis (nur Metriken je Frage, kein Buchtext): results/crag_experiment.json + .png

Aufruf: make eval-crag   (aus backend/ via uv, s. Makefile)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "eval"))

from answer_eval import Faithfulness, generate, is_refusal, token_f1  # noqa: E402
from retrieval_eval import span_hit  # noqa: E402

from app.clients.inference_router import build_inference_client  # noqa: E402
from app.clients.qdrant_client import VectorStore  # noqa: E402
from app.config import settings  # noqa: E402
from app.flow.crag import CragFlow  # noqa: E402
from app.flow.nodes import CragNodes  # noqa: E402
from app.rag.prompt_builder import build_messages  # noqa: E402
from app.rag.retriever import retrieve  # noqa: E402
from app.services.faithfulness_service import FaithfulnessService  # noqa: E402

ARMS = ["dense8", "dense4", "crag", "crag_regen", "crag_soft"]
CKPT = ROOT / "results" / "crag_answers.jsonl"


def sign_test(better: int, worse: int) -> float:
    """Zweiseitiger exakter Vorzeichentest (Gleichstände raus)."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def span_metrics(payloads: list[dict], item: dict) -> dict:
    if not item.get("spans"):
        return {}
    flags = [span_hit(pl, item) for pl in payloads]
    first = next((i + 1 for i, f in enumerate(flags) if f), None)
    return {"hit": 1.0 if first else 0.0, "mrr": 1.0 / first if first else 0.0}


async def run_dense(item, k, *, ollama, vectors, group, model) -> dict:
    t0 = time.perf_counter()
    pts = await retrieve(item["question"], ollama=ollama, vectors=vectors,
                         embed_model=settings.embed_model, top_k=k, group_id=group)
    gen = await generate(ollama, build_messages(item["question"], pts), model)
    return {
        "answer": gen["answer"], "e2e_ms": (time.perf_counter() - t0) * 1000, "llm_calls": 1,
        "context": [p.payload or {} for p in pts],
    }


async def run_crag(item, flow: CragFlow, *, group, model, k, max_regen) -> dict:
    t0 = time.perf_counter()
    st = await flow.invoke(item["question"], group_ids=[group], model=model, top_k=k,
                           max_regen=max_regen)
    trace = st.get("trace", [])
    llm = sum(1 for t in trace if t["node"] in ("grade", "rewrite", "generate") and
              not (t["node"] == "grade" and t.get("n") == 0))
    return {
        "answer": st.get("answer", ""), "e2e_ms": (time.perf_counter() - t0) * 1000,
        "llm_calls": llm, "context": [p.payload or {} for p in st.get("context", [])],
        "attempts": st.get("attempts", 0), "regen": st.get("regen", 0),
        "crag_refused": bool(st.get("refused")), "queries": st.get("queries", []),
        "graded": st.get("graded", []),
    }


def plot(summary: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Kategoriale Slots 1-5 in fester Reihenfolge (dataviz-Referenzpalette, validiert);
    # Kontrast < 3:1 bei Slot 3/4 -> Werte stehen immer als Label an den Balken.
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    panels = [
        ("faithfulness", "Faithfulness (NLI)", "{:.2f}"),
        ("hit", "Gold-Passage im Prompt (Hit)", "{:.2f}"),
        ("refusal_rate", "Refusal-Rate (unbeantwortbar)", "{:.2f}"),
        ("e2e_median_s", "e2e Median (s)", "{:.1f}"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 3.8))
    for ax, (key, title, fmt) in zip(axes, panels):
        vals = [0.0 if math.isnan(summary[a][key]) else summary[a][key] for a in ARMS]
        bars = ax.bar(range(len(ARMS)), vals, color=colors, width=0.7, edgecolor="white", linewidth=2)
        for b, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), ha="center",
                        va="bottom", fontsize=9, color="#333333", xytext=(0, 2),
                        textcoords="offset points")
        ax.set_title(title, fontsize=10, color="#333333")
        ax.set_xticks(range(len(ARMS)), ARMS, fontsize=8, rotation=20)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8, colors="#777777")
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
        # Raten/Anteile immer auf 0..1 (sonst Pseudo-Achse bei lauter Nullen), Sekunden frei.
        ax.set_ylim(0, 1.15 if key != "e2e_median_s" else max(vals + [1.0]) * 1.18)
    fig.suptitle("Exp 11/11b — Corrective RAG vs. dense (Gold v2, qwen2.5:7b)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "eval" / "gold_v2.jsonl"))
    ap.add_argument("--group", default="Horror")
    ap.add_argument("--model", default=settings.chat_model)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--limit", type=int, help="nur die ersten N Items (Smoke-Test)")
    ap.add_argument("--plot-only", action="store_true",
                    help="nur das Diagramm aus results/crag_experiment.json neu zeichnen")
    args = ap.parse_args()
    arms = args.arms.split(",")
    if args.plot_only:
        summary = json.loads((ROOT / "results" / "crag_experiment.json").read_text())["summary"]
        plot(summary, ROOT / "results" / "crag_experiment.png")
        return

    gold = [json.loads(x) for x in Path(args.gold).read_text().splitlines() if x.strip()]
    if args.limit:
        gold = gold[: args.limit]

    done: dict[tuple[str, str], dict] = {}
    if CKPT.exists():
        for line in CKPT.read_text().splitlines():
            r = json.loads(line)
            done[(r["arm"], r["id"])] = r

    print("Lade NLI-Modell (Messung) …")
    faith = Faithfulness()
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    ollama = build_inference_client(settings, timeout=300)
    vectors = VectorStore(settings.qdrant_url, settings.qdrant_collection)

    def make_flow(with_check: bool, strategy: str = "filter") -> CragFlow:
        checker = None
        if with_check:
            checker = FaithfulnessService(settings.faithfulness_model, True,
                                          settings.faithfulness_threshold)
        nodes = CragNodes(ollama, vectors, embed_model=settings.embed_model,
                          min_relevant=settings.crag_min_relevant,
                          unsupported_max=settings.crag_unsupported_max,
                          grader_model=settings.crag_grader_model, faithfulness=checker,
                          temperature=0.0, strategy=strategy)
        return CragFlow(nodes, enabled=True, max_retries=settings.crag_max_retries,
                        max_regen=settings.crag_max_regen)

    flows = {"crag": make_flow(False), "crag_regen": make_flow(True),
             "crag_soft": make_flow(False, "trigger")}

    try:
        await ollama.complete([{"role": "user", "content": "OK"}], args.model, max_tokens=1)
        ctx = await ollama.context_length(args.model)
        if ctx is None or ctx < 8000:
            sys.exit(f"Ollama läuft {args.model} mit Kontext {ctx} (< 8000) — `make ollama-ctx` + "
                     "Ollama-App neu starten, dann erneut.")
        print(f"Kontextfenster {args.model}: {ctx} Token — ok")

        CKPT.parent.mkdir(exist_ok=True)
        with CKPT.open("a") as ck:
            for item in gold:
                for arm in arms:
                    if (arm, item["id"]) in done:
                        continue
                    if arm.startswith("dense"):
                        r = await run_dense(item, int(arm[5:]), ollama=ollama, vectors=vectors,
                                            group=args.group, model=args.model)
                    else:
                        r = await run_crag(item, flows[arm], group=args.group, model=args.model,
                                           k=8, max_regen=settings.crag_max_regen if arm == "crag_regen" else 0)
                    r.update({"arm": arm, "id": item["id"]})
                    ck.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                    ck.flush()
                    done[(arm, item["id"])] = r
                    print(f"  {item['id']} {arm:<11} {r['e2e_ms'] / 1000:5.1f}s  "
                          f"calls={r['llm_calls']}  refused={is_refusal(r['answer'])}")
    finally:
        await ollama.aclose()
        await vectors.aclose()

    # --- Metriken je (Arm, Item) ---------------------------------------------------------
    rows: list[dict] = []
    for item in gold:
        unanswerable = item.get("qtype") == "unanswerable"
        for arm in arms:
            r = done[(arm, item["id"])]
            refused = is_refusal(r["answer"]) or r.get("crag_refused", False)
            row = {"arm": arm, "id": item["id"], "qtype": item.get("qtype", "fact"),
                   "refused": refused, "e2e_ms": r["e2e_ms"], "llm_calls": r["llm_calls"],
                   "n_context": len(r["context"]), "attempts": r.get("attempts"),
                   "regen": r.get("regen")}
            row.update(span_metrics(r["context"], item))
            if not unanswerable and not refused:
                chunks = [c.get("text", "") for c in r["context"]]
                row["faithfulness"] = faith.score(r["answer"], chunks)
                row["rougeL"] = rs.score(item["answer"], r["answer"])["rougeL"].fmeasure
                row["answer_f1"] = token_f1(r["answer"], item["answer"])
            rows.append(row)

    def sel(arm):
        return [x for x in rows if x["arm"] == arm]

    def mean(xs, key):
        v = [x[key] for x in xs if x.get(key) is not None]
        return sum(v) / len(v) if v else float("nan")

    summary: dict = {}
    for arm in arms:
        xs = sel(arm)
        ans = [x for x in xs if x["qtype"] != "unanswerable"]
        una = [x for x in xs if x["qtype"] == "unanswerable"]
        e2e = sorted(x["e2e_ms"] for x in xs)
        summary[arm] = {
            "faithfulness": mean(ans, "faithfulness"),
            "rougeL": mean(ans, "rougeL"), "answer_f1": mean(ans, "answer_f1"),
            "refusal_rate": sum(x["refused"] for x in una) / len(una) if una else float("nan"),
            "false_refusal_rate": sum(x["refused"] for x in ans) / len(ans),
            "hit": mean(ans, "hit"), "mrr": mean(ans, "mrr"),
            "n_context": mean(xs, "n_context"), "llm_calls": mean(xs, "llm_calls"),
            "e2e_median_s": e2e[len(e2e) // 2] / 1000,
            "retry_rate": (sum(1 for x in xs if (x["attempts"] or 0) > 1) / len(xs))
            if arm.startswith("crag") else None,
            "regen_rate": (sum(1 for x in xs if (x["regen"] or 0) > 0) / len(xs))
            if arm.startswith("crag") else None,
        }

    # Paired Faithfulness: nur Items, die in BEIDEN Armen beantwortet wurden.
    paired: dict = {}
    for arm in [a for a in arms if a.startswith("crag")]:
        for base in [a for a in arms if a.startswith("dense")]:
            a = {x["id"]: x.get("faithfulness") for x in sel(arm)}
            b = {x["id"]: x.get("faithfulness") for x in sel(base)}
            both = [i for i in a if a[i] is not None and b.get(i) is not None]
            better = sum(a[i] > b[i] + 1e-9 for i in both)
            worse = sum(a[i] < b[i] - 1e-9 for i in both)
            paired[f"{arm}_vs_{base}"] = {
                "n": len(both), "better": better, "worse": worse, "same": len(both) - better - worse,
                "delta_mean": (sum(a[i] - b[i] for i in both) / len(both)) if both else None,
                "p_sign": round(sign_test(better, worse), 3),
            }

    # Grader-Kalibrierung: Urteil je bewerteter Passage vs. Span-Overlap (nur beantwortbar).
    calib = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "relevant_on_unanswerable": 0,
             "graded_on_unanswerable": 0}
    by_id = {it["id"]: it for it in gold}
    if "crag" in arms:
        for item in gold:
            r = done[("crag", item["id"])]
            for g in r.get("graded", []):
                if not item.get("spans"):
                    calib["graded_on_unanswerable"] += 1
                    calib["relevant_on_unanswerable"] += int(g["relevant"])
                    continue
                truth = span_hit(g, by_id[item["id"]])
                key = ("tp" if truth else "fp") if g["relevant"] else ("fn" if truth else "tn")
                calib[key] += 1
        tp, fp, fn = calib["tp"], calib["fp"], calib["fn"]
        calib["precision"] = tp / (tp + fp) if tp + fp else None
        calib["recall"] = tp / (tp + fn) if tp + fn else None

    # Sanity (Exp 11b): ohne Retry ist crag_soft prompt-identisch zu dense8 -> gleiche Antwort.
    sanity = None
    if {"crag_soft", "dense8"} <= set(arms):
        single = [it["id"] for it in gold if done[("crag_soft", it["id"])].get("attempts") == 1
                  and not done[("crag_soft", it["id"])].get("crag_refused")]
        same = sum(done[("crag_soft", i)]["answer"] == done[("dense8", i)]["answer"] for i in single)
        sanity = {"single_attempt": len(single), "identical_to_dense8": same}

    print(f"\n=== Exp 11 (n={len(gold)}, Modell={args.model}) ===")
    hdr = f"{'Arm':<11} {'Faith':>6} {'ROUGE':>6} {'F1':>6} {'Refus':>6} {'FalseR':>6} " \
          f"{'Hit':>6} {'MRR':>6} {'#Ctx':>5} {'Calls':>5} {'e2e':>6}"
    print(hdr)
    for arm in arms:
        s = summary[arm]
        print(f"{arm:<11} {s['faithfulness']:6.3f} {s['rougeL']:6.3f} {s['answer_f1']:6.3f} "
              f"{s['refusal_rate']:6.2f} {s['false_refusal_rate']:6.2f} {s['hit']:6.3f} "
              f"{s['mrr']:6.3f} {s['n_context']:5.1f} {s['llm_calls']:5.1f} {s['e2e_median_s']:5.1f}s")
    print("\nPaired Faithfulness:")
    for k, v in paired.items():
        print(f"  {k}: {v['better']}:{v['worse']}:{v['same']} (n={v['n']}, "
              f"Δ={v['delta_mean'] if v['delta_mean'] is None else round(v['delta_mean'], 3)}, "
              f"p={v['p_sign']})")
    print(f"\nGrader-Kalibrierung: {calib}")
    if sanity:
        print(f"Sanity crag_soft ohne Retry == dense8: {sanity['identical_to_dense8']}/"
              f"{sanity['single_attempt']}")

    out = ROOT / "results" / "crag_experiment.json"
    out.write_text(json.dumps({"model": args.model, "n": len(gold), "summary": summary,
                               "paired": paired, "grader": calib, "sanity_soft": sanity, "rows": rows}, indent=2))
    if set(ARMS) <= set(arms):
        plot(summary, ROOT / "results" / "crag_experiment.png")
    print(f"\nErgebnis: {out}")


if __name__ == "__main__":
    asyncio.run(main())
