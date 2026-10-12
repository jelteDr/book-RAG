"""Ablation: die Vorher-Nachher-Kurve der Pipeline in EINEM einheitlichen Lauf.

Die Einzelexperimente (Exp 4/5/7/10) entstanden in verschiedenen Messepochen (Kapitel- vs.
Span-Metrik, n=21 vs. 36, teils mit abgeschnittenem Kontext). Hier wird jede Stufe
kumulativ unter identischen Bedingungen gemessen (Gold v2, qwen2.5:7b, temp 0, num_ctx >= 8000):

  S0 naiv        reine Text-Embeddings (book_chunks_plain), keine Dedup, k=8
  S1 +Dedup      Overlap-Dedup im Retrieval (Exp 4)
  S2 +Contextual LLM-Kontext im Embedding (Exp 5) — Produktiv-Collection book_chunks
  S3 k=6         weniger Passagen im Prompt (Exp 10)
  S4 +S2B        Small-to-Big, window 1 / top_n 3 (Exp 7)

Metriken wie die Suite (`answer_eval.py`): Faithfulness (NLI), ROUGE-L, Token-F1, Refusal /
False-Refusal, Span-Hit/MRR des Prompt-Kontexts, e2e; paired Faithfulness je Schritt.
Die Stufen laufen NACHEINANDER über alle Fragen (nicht verschränkt je Frage), damit Ollamas
Prefix-Cache keine Stufe bevorzugt (Lehre aus Exp 11).

Vorbereitung (einmalig, ~2 min): --build-plain baut book_chunks_plain aus den Payload-Texten
der Produktiv-Collection (gleiche IDs/Offsets, Embedding OHNE Kontext-Präfix).

Checkpoint (resumierbar, enthält Antworten -> gitignored): results/ablation_answers.jsonl
Ergebnis (nur Metriken): results/ablation.json + results/ablation.png

Aufruf: make eval-ablation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "eval"))

from answer_eval import Faithfulness, generate, is_refusal, token_f1  # noqa: E402
from retrieval_eval import span_hit  # noqa: E402

from app.clients.ollama_client import OllamaClient  # noqa: E402
from app.clients.qdrant_client import VectorStore  # noqa: E402
from app.config import settings  # noqa: E402
from app.rag.expander import expand_points  # noqa: E402
from app.rag.prompt_builder import build_messages  # noqa: E402
from app.rag.retriever import build_group_filter, embed_question, retrieve  # noqa: E402
from app.util import l2_normalize  # noqa: E402

PLAIN = "book_chunks_plain"
CKPT = ROOT / "results" / "ablation_answers.jsonl"
JUDGE_CKPT = ROOT / "results" / "ablation_judge.jsonl"  # nur Urteile (JA/NEIN), kein Text

# Korrektheit per LLM-Judge (post hoc ergänzt): Faithfulness misst Stützung durch die
# Passagen, nicht ob die Antwort STIMMT — eine Antwort kann eng an einer falschen Passage
# kleben. NLI (Antwort => Gold) war dafür unbrauchbar (korrekte Paraphrasen ~0,02).
JUDGE_PROMPT = (
    "Frage: {question}\n\nReferenzantwort: {gold}\n\nZu bewertende Antwort: {answer}\n\n"
    "Trifft die zu bewertende Antwort die Kernaussage der Referenzantwort? Andere Formulierung "
    "oder zusätzliche Details sind in Ordnung; fehlt die Kernaussage oder widerspricht die "
    "Antwort ihr, dann nicht. Antworte NUR mit JA oder NEIN."
)


async def judge(ollama: OllamaClient, model: str, question: str, gold: str, answer: str) -> bool:
    clean = re.sub(r"\s*\[\d+\]", "", answer)
    out = await ollama.complete(
        [{"role": "user", "content": JUDGE_PROMPT.format(question=question, gold=gold, answer=clean)}],
        model, temperature=0.0, max_tokens=3,
    )
    return out.strip().upper().startswith("JA")

STAGES = [
    {"key": "S0", "label": "naiv", "collection": PLAIN, "dedup": False, "k": 8, "s2b": False},
    {"key": "S1", "label": "+Dedup", "collection": PLAIN, "dedup": True, "k": 8, "s2b": False},
    {"key": "S2", "label": "+Contextual", "collection": "prod", "dedup": True, "k": 8, "s2b": False},
    {"key": "S3", "label": "k=6", "collection": "prod", "dedup": True, "k": 6, "s2b": False},
    {"key": "S4", "label": "+Small-to-Big", "collection": "prod", "dedup": True, "k": 6, "s2b": True},
]


async def build_plain(group: str) -> None:
    prod = VectorStore(settings.qdrant_url, settings.qdrant_collection)
    plain = VectorStore(settings.qdrant_url, PLAIN)
    ollama = OllamaClient(settings.ollama_base_url, timeout=300)
    try:
        await plain.ensure_collection()
        records = await prod.scroll_all(group_id=group)
        print(f"{len(records)} Chunks -> {PLAIN} (Embedding nur Chunk-Text)")
        for i in range(0, len(records), 64):
            part = records[i : i + 64]
            embs = await ollama.embed([r.payload["text"] for r in part], settings.embed_model)
            points = []
            for rec, emb in zip(part, embs, strict=True):
                payload = {k: v for k, v in rec.payload.items() if k != "context"}
                points.append({"vector": l2_normalize(emb), "payload": payload})
            await plain.upsert_chunks(points)
        print("fertig")
    finally:
        await ollama.aclose()
        await prod.aclose()
        await plain.aclose()


async def stage_points(stage: dict, question: str, *, ollama, stores: dict, group: str):
    store = stores[stage["collection"]]
    if stage["dedup"]:
        pts = await retrieve(question, ollama=ollama, vectors=store,
                             embed_model=settings.embed_model, top_k=stage["k"], group_id=group)
    else:  # Rohsuche wie vor Exp 4: top-k ohne Overlap-Puffer
        vec = await embed_question(question, ollama=ollama, embed_model=settings.embed_model)
        pts = await store.search(vec, stage["k"], build_group_filter(group))
    if stage["s2b"]:
        pts = await expand_points(pts, vectors=store, window=settings.s2b_window,
                                  top_n=settings.s2b_top_n)
    return pts


def answerable_items(gold: list[dict]) -> list[dict]:
    return [it for it in gold if it.get("qtype") != "unanswerable"]


def sign_test(better: int, worse: int) -> float:
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def plot(summary: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{s['key']}\n{s['label']}" for s in summary]
    x = range(len(summary))
    # Kategoriale Slots 1-4 (dataviz-Referenzpalette, validiert); Werte stehen als Label dran.
    series = [("correct", "Korrekt (Judge)", "#2a78d6"), ("faithfulness", "Faithfulness (NLI)", "#eb6834"),
              ("hit", "Gold im Prompt (Hit)", "#1baf7a"), ("mrr", "MRR", "#eda100")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    for key, name, color in series:
        ys = [s[key] for s in summary]
        ax.plot(x, ys, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=2, label=name)
        # Selektive Labels: nur Start und Ende (alle Werte stehen in der Tabelle/RESULTS.md).
        for xi in (0, len(ys) - 1):
            ax.annotate(f"{ys[xi]:.2f}", (xi, ys[xi]), textcoords="offset points",
                        xytext=(-14 if xi == 0 else 14, 0), ha="right" if xi == 0 else "left",
                        va="center", fontsize=8, color="#333333")
    ax.set_xticks(list(x), labels, fontsize=8)
    ax.set_xlim(-0.5, len(summary) - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("Qualität je Stufe (kumulativ)", fontsize=10, color="#333333")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.14))
    # Kosten deterministisch als Prompt-Länge: die Wanduhr-Latenz schwankte im Lauf stark
    # (S1 und S2 haben dieselbe Prompt-Größe, aber 27 s vs. 42 s) und taugt nicht als Vergleich.
    chars = [s["prompt_chars"] / 1000 for s in summary]
    bars = ax2.bar(list(x), chars, color="#9aa0a6", width=0.65, edgecolor="white", linewidth=2)
    for b, v in zip(bars, chars):
        ax2.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                     xytext=(0, 2), ha="center", fontsize=8, color="#333333")
    ax2.set_xticks(list(x), [s["key"] for s in summary], fontsize=8)
    ax2.set_ylim(0, max(chars) * 1.2)
    ax2.set_title("Prompt-Länge (Tsd. Zeichen Quellen)", fontsize=10, color="#333333")
    for a in (ax, ax2):
        a.spines[["top", "right", "left"]].set_visible(False)
        a.tick_params(axis="y", labelsize=8, colors="#777777")
        a.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        a.set_axisbelow(True)
    fig.suptitle("book-RAG — Vorher/Nachher je Pipeline-Stufe (Gold v2, n=36, qwen2.5:7b)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "eval" / "gold_v2.jsonl"))
    ap.add_argument("--group", default="Horror")
    ap.add_argument("--model", default=settings.chat_model)
    ap.add_argument("--stages", default=",".join(s["key"] for s in STAGES))
    ap.add_argument("--limit", type=int, help="nur die ersten N Items (Smoke-Test)")
    ap.add_argument("--build-plain", action="store_true", help=f"{PLAIN} aufbauen und beenden")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if args.build_plain:
        await build_plain(args.group)
        return
    if args.plot_only:
        plot(json.loads((ROOT / "results" / "ablation.json").read_text())["summary"],
             ROOT / "results" / "ablation.png")
        return

    stages = [s for s in STAGES if s["key"] in args.stages.split(",")]
    gold = [json.loads(x) for x in Path(args.gold).read_text().splitlines() if x.strip()]
    if args.limit:
        gold = gold[: args.limit]

    done: dict[tuple[str, str], dict] = {}
    if CKPT.exists():
        for line in CKPT.read_text().splitlines():
            r = json.loads(line)
            done[(r["stage"], r["id"])] = r

    ollama = OllamaClient(settings.ollama_base_url, timeout=300)
    stores = {"prod": VectorStore(settings.qdrant_url, settings.qdrant_collection),
              PLAIN: VectorStore(settings.qdrant_url, PLAIN)}
    try:
        n_plain = len(await stores[PLAIN].scroll_all(group_id=args.group))
        if n_plain == 0:
            sys.exit(f"{PLAIN} ist leer — erst `--build-plain` ausführen.")
        await ollama.complete([{"role": "user", "content": "OK"}], args.model, max_tokens=1)
        ctx = await ollama.context_length(args.model)
        if ctx is None or ctx < 8000:
            sys.exit(f"Ollama läuft {args.model} mit Kontext {ctx} (< 8000) — `make ollama-ctx` + "
                     "Ollama-App neu starten, dann erneut.")
        print(f"Kontextfenster {args.model}: {ctx} Token — ok; {PLAIN}: {n_plain} Chunks")

        CKPT.parent.mkdir(exist_ok=True)
        with CKPT.open("a") as ck:
            for stage in stages:  # Stufen nacheinander (Prefix-Cache-fair)
                print(f"\n--- {stage['key']} {stage['label']} ---")
                for item in gold:
                    if (stage["key"], item["id"]) in done:
                        continue
                    t0 = time.perf_counter()
                    pts = await stage_points(stage, item["question"], ollama=ollama,
                                             stores=stores, group=args.group)
                    gen = await generate(ollama, build_messages(item["question"], pts), args.model)
                    r = {"stage": stage["key"], "id": item["id"], "answer": gen["answer"],
                         "e2e_ms": (time.perf_counter() - t0) * 1000,
                         "context": [p.payload or {} for p in pts]}
                    ck.write(json.dumps(r, ensure_ascii=False) + "\n")
                    ck.flush()
                    done[(stage["key"], item["id"])] = r
                    print(f"  {item['id']} {r['e2e_ms'] / 1000:5.1f}s refused={is_refusal(r['answer'])}")

        # Judge: Korrektheit je beantworteter, beantwortbarer Antwort + Kalibrierung.
        verdicts: dict[str, bool] = {}
        if JUDGE_CKPT.exists():
            for line in JUDGE_CKPT.read_text().splitlines():
                v = json.loads(line)
                verdicts[v["key"]] = v["ja"]
        answerable = answerable_items(gold)
        jobs = [(f"{st['key']}:{it['id']}", it, done[(st["key"], it["id"])]["answer"])
                for st in stages for it in answerable
                if not is_refusal(done[(st["key"], it["id"])]["answer"])]
        # Kalibrierung: Referenz gegen sich selbst (soll JA) und gegen die Referenz der
        # nächsten Frage (soll NEIN).
        for i, it in enumerate(answerable):
            other = answerable[(i + 1) % len(answerable)]
            jobs.append((f"cal_pos:{it['id']}", it, it["answer"]))
            jobs.append((f"cal_neg:{it['id']}", it, other["answer"]))
        print(f"\nJudge: {sum(1 for k, *_ in jobs if k not in verdicts)} offene Urteile …")
        with JUDGE_CKPT.open("a") as jk:
            for key, it, ans in jobs:
                if key in verdicts:
                    continue
                verdicts[key] = await judge(ollama, args.model, it["question"], it["answer"], ans)
                jk.write(json.dumps({"key": key, "ja": verdicts[key]}) + "\n")
                jk.flush()
    finally:
        await ollama.aclose()
        for s in stores.values():
            await s.aclose()

    print("\nLade NLI-Modell …")
    faith = Faithfulness()
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    rows: list[dict] = []
    for stage in stages:
        for item in gold:
            r = done[(stage["key"], item["id"])]
            refused = is_refusal(r["answer"])
            row = {"stage": stage["key"], "id": item["id"], "qtype": item.get("qtype", "fact"),
                   "refused": refused, "e2e_ms": r["e2e_ms"],
                   "prompt_chars": sum(len(c.get("text", "")) for c in r["context"])}
            if item.get("spans"):
                flags = [span_hit(c, item) for c in r["context"]]
                first = next((i + 1 for i, f in enumerate(flags) if f), None)
                row["hit"] = 1.0 if first else 0.0
                row["mrr"] = 1.0 / first if first else 0.0
            if item.get("qtype") != "unanswerable" and not refused:
                chunks = [c.get("text", "") for c in r["context"]]
                row["faithfulness"] = faith.score(r["answer"], chunks)  # gefenstert (korrekt)
                row["faithfulness_trunc"] = faith.score(r["answer"], chunks, windowed=False)
                row["correct"] = 1.0 if verdicts[f"{stage['key']}:{item['id']}"] else 0.0
                row["rougeL"] = rs.score(item["answer"], r["answer"])["rougeL"].fmeasure
                row["answer_f1"] = token_f1(r["answer"], item["answer"])
            rows.append(row)

    def mean(xs, key):
        v = [x[key] for x in xs if x.get(key) is not None]
        return sum(v) / len(v) if v else float("nan")

    summary = []
    for stage in stages:
        xs = [r for r in rows if r["stage"] == stage["key"]]
        ans = [x for x in xs if x["qtype"] != "unanswerable"]
        una = [x for x in xs if x["qtype"] == "unanswerable"]
        e2e = sorted(x["e2e_ms"] for x in xs)
        summary.append({
            "key": stage["key"], "label": stage["label"],
            "faithfulness": mean(ans, "faithfulness"),
            "faithfulness_trunc": mean(ans, "faithfulness_trunc"),
            # Korrekt-Rate über ALLE beantwortbaren Fragen (Verweigerung = nicht korrekt).
            "correct": sum(x.get("correct", 0.0) for x in ans) / len(ans),
            "rougeL": mean(ans, "rougeL"),
            "answer_f1": mean(ans, "answer_f1"), "hit": mean(ans, "hit"), "mrr": mean(ans, "mrr"),
            "refusal_rate": sum(x["refused"] for x in una) / len(una) if una else float("nan"),
            "false_refusal_rate": sum(x["refused"] for x in ans) / len(ans),
            "prompt_chars": mean(xs, "prompt_chars"), "e2e_median_s": e2e[len(e2e) // 2] / 1000,
        })

    steps = []
    for prev, cur in zip(stages, stages[1:]):
        step = {"step": f"{prev['key']}->{cur['key']}"}
        for key in ("faithfulness", "correct"):
            a = {r["id"]: r.get(key) for r in rows if r["stage"] == cur["key"]}
            b = {r["id"]: r.get(key) for r in rows if r["stage"] == prev["key"]}
            both = [i for i in a if a[i] is not None and b.get(i) is not None]
            better = sum(a[i] > b[i] + 1e-9 for i in both)
            worse = sum(a[i] < b[i] - 1e-9 for i in both)
            step[key] = {"n": len(both), "better": better, "worse": worse,
                         "same": len(both) - better - worse,
                         "p_sign": round(sign_test(better, worse), 3)}
        steps.append(step)

    print(f"\n=== Ablation (n={len(gold)}, Modell={args.model}) ===")
    print(f"{'Stufe':<18} {'Faith':>6} {'F_alt':>6} {'Korrekt':>7} {'ROUGE':>6} {'F1':>6} {'Refus':>6} {'FalseR':>6} "
          f"{'Hit':>6} {'MRR':>6} {'Zeichen':>8} {'e2e':>6}")
    for s in summary:
        print(f"{s['key'] + ' ' + s['label']:<18} {s['faithfulness']:6.3f} "
              f"{s['faithfulness_trunc']:6.3f} {s['correct']:7.3f} {s['rougeL']:6.3f} "
              f"{s['answer_f1']:6.3f} {s['refusal_rate']:6.2f} {s['false_refusal_rate']:6.2f} "
              f"{s['hit']:6.3f} {s['mrr']:6.3f} {s['prompt_chars']:8.0f} {s['e2e_median_s']:5.1f}s")
    print("\nPaired je Schritt (besser:schlechter:gleich):")
    for st in steps:
        f, c = st["faithfulness"], st["correct"]
        print(f"  {st['step']}: Faith {f['better']}:{f['worse']}:{f['same']} (p={f['p_sign']})  "
              f"Korrektheit {c['better']}:{c['worse']}:{c['same']} (p={c['p_sign']})")

    cal = {
        "pos_ja_rate": mean([{"v": float(verdicts[f"cal_pos:{it['id']}"])} for it in answerable_items(gold)], "v"),
        "neg_ja_rate": mean([{"v": float(verdicts[f"cal_neg:{it['id']}"])} for it in answerable_items(gold)], "v"),
    }
    print(f"Judge-Kalibrierung: Referenz=JA {cal['pos_ja_rate']:.2f} (soll 1), "
          f"fremde Referenz=JA {cal['neg_ja_rate']:.2f} (soll 0)")

    out = ROOT / "results" / "ablation.json"
    out.write_text(json.dumps({"model": args.model, "n": len(gold), "summary": summary,
                               "steps": steps, "judge_calibration": cal, "rows": rows}, indent=2))
    if len(stages) == len(STAGES):
        plot(summary, ROOT / "results" / "ablation.png")
    print(f"\nErgebnis: {out}")


if __name__ == "__main__":
    asyncio.run(main())
