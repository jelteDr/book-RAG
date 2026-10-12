"""Modellvergleich: gleiche Retrieval-Stufe, anderes Generierungsmodell.

Die Ablation (`ablation.py`) zeigte: Retrieval liefert die Gold-Passage in ~80 % der Fragen,
aber nur ~1/3 der Antworten ist korrekt — der Engpass ist die Generierung. Hier bleibt das
Retrieval fest (Stufe S4: Contextual + Dedup, k=6, Small-to-Big) und für JEDE Frage exakt
derselbe Kontext; nur das Generierungsmodell wechselt (temp 0, num_ctx >= 8000).

Metriken: Korrekt (LLM-Judge, zwei Judges + Kalibrierung + Übereinstimmung), Faithfulness
(NLI, gefenstert), Zitatquote (Sätze mit [n] — das Frontend lebt von Zitaten), Refusal /
False-Refusal, ROUGE-L/F1, TTFT/TPS/e2e (Median), leere Antworten; paired Korrektheit vs.
qwen2.5:7b. qwen2.5:7b wird neu generiert und muss die Ablation-S4-Werte treffen (Sanity).

Ablauf (speicherschonend, 24 GB): erst alle Antworten Modell für Modell (jedes Modell einmal
laden), dann die Judges nacheinander, zuletzt NLI.

Checkpoints (gitignored, enthalten Antworten): results/model_answers.jsonl,
results/model_judge.jsonl. Ergebnis: results/model_compare.json + .png

Aufruf: make eval-models
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "eval"))

from ablation import STAGES, answerable_items, judge, sign_test, stage_points  # noqa: E402
from answer_eval import Faithfulness, generate, is_refusal, token_f1  # noqa: E402
from retrieval_eval import span_hit  # noqa: E402

from app.clients.ollama_client import OllamaClient  # noqa: E402
from app.clients.qdrant_client import VectorStore  # noqa: E402
from app.config import settings  # noqa: E402
from app.rag.prompt_builder import build_messages  # noqa: E402

BASELINE = "qwen2.5:7b-instruct-q4_k_m"
MODELS = [BASELINE, "llama3.1:8b", "qwen3:8b", "mistral:latest", "gemma4:latest", "qwen2.5:14b"]
JUDGES = ["qwen2.5:14b", BASELINE]  # erster = Haupt-Judge
S4 = next(s for s in STAGES if s["key"] == "S4")
ANS_CKPT = ROOT / "results" / "model_answers.jsonl"
JUDGE_CKPT = ROOT / "results" / "model_judge.jsonl"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_CITE_RE = re.compile(r"\[\d+\]")


def prepare(messages: list[dict], model: str) -> list[dict]:
    """qwen3 denkt sonst laut (lange <think>-Blöcke) — für den Vergleich abgeschaltet."""
    if model.startswith("qwen3"):
        messages = [*messages[:-1], {**messages[-1], "content": messages[-1]["content"] + " /no_think"}]
    return messages


def citation_rate(answer: str) -> float:
    sents = [s for s in _SENT_RE.split(answer) if len(s.strip()) > 15]
    return sum(1 for s in sents if _CITE_RE.search(s)) / len(sents) if sents else 0.0


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def short(model: str) -> str:
    return model.replace("-instruct-q4_k_m", "").replace(":latest", "")


def plot(summary: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Kategoriale Slots 1-6 in fester Reihenfolge (validiert); Farbe folgt dem Modell über alle
    # Panels, Werte stehen als Label an jedem Balken (Kontrast < 3:1 bei Slot 3-5).
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    panels = [("correct", "Korrekt (Judge qwen2.5:14b)", "{:.2f}", 1.0),
              ("faithfulness", "Faithfulness (NLI)", "{:.2f}", 1.0),
              ("citation_rate", "Zitatquote (Sätze mit [n])", "{:.2f}", 1.0),
              ("e2e_median_s", "Antwortzeit Median (s)", "{:.0f}", None)]
    names = [short(s["model"]) for s in summary]
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 4.2))
    for ax, (key, title, fmt, top) in zip(axes, panels):
        vals = [s[key] for s in summary]
        bars = ax.bar(range(len(vals)), vals, color=colors[: len(vals)], width=0.7,
                      edgecolor="white", linewidth=2)
        for b, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), ha="center",
                        va="bottom", fontsize=8, color="#333333", xytext=(0, 2),
                        textcoords="offset points")
        ax.set_title(title, fontsize=10, color="#333333")
        ax.set_xticks(range(len(vals)), names, fontsize=8, rotation=30, ha="right")
        ax.set_ylim(0, (top or max(vals) * 1.15) * (1.12 if top else 1))
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8, colors="#777777")
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Modellvergleich — gleiches Retrieval (S4), anderes Generierungsmodell "
                 "(Gold v2, n=36)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "eval" / "gold_v2.jsonl"))
    ap.add_argument("--group", default="Horror")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.plot_only:
        plot(json.loads((ROOT / "results" / "model_compare.json").read_text())["summary"],
             ROOT / "results" / "model_compare.png")
        return
    models = args.models.split(",")

    gold = [json.loads(x) for x in Path(args.gold).read_text().splitlines() if x.strip()]
    if args.limit:
        gold = gold[: args.limit]
    answerable = answerable_items(gold)

    done: dict[tuple[str, str], dict] = {}
    if ANS_CKPT.exists():
        for line in ANS_CKPT.read_text().splitlines():
            r = json.loads(line)
            done[(r["model"], r["id"])] = r
    verdicts: dict[str, bool] = {}
    if JUDGE_CKPT.exists():
        for line in JUDGE_CKPT.read_text().splitlines():
            v = json.loads(line)
            verdicts[v["key"]] = v["ja"]

    ollama = OllamaClient(settings.ollama_base_url, timeout=600)
    stores = {"prod": VectorStore(settings.qdrant_url, settings.qdrant_collection)}
    try:
        # Kontext je Frage EINMAL bestimmen -> jedes Modell sieht exakt dieselben Quellen.
        contexts = {}
        for item in gold:
            contexts[item["id"]] = await stage_points(S4, item["question"], ollama=ollama,
                                                      stores=stores, group=args.group)
        print(f"Kontexte für {len(contexts)} Fragen bestimmt (Stufe S4)")

        with ANS_CKPT.open("a") as ck:
            for model in models:
                todo = [it for it in gold if (model, it["id"]) not in done]
                if not todo:
                    continue
                await ollama.complete([{"role": "user", "content": "OK"}], model, max_tokens=1)
                ctx = await ollama.context_length(model)
                if ctx is None or ctx < 8000:
                    sys.exit(f"{model} läuft mit Kontext {ctx} (< 8000) — `make ollama-ctx` + Neustart.")
                print(f"\n--- {model} (Kontext {ctx}) ---")
                for item in todo:
                    pts = contexts[item["id"]]
                    msgs = prepare(build_messages(item["question"], pts), model)
                    gen = await generate(ollama, msgs, model)
                    answer = _THINK_RE.sub("", gen["answer"]).strip()
                    r = {"model": model, "id": item["id"], "answer": answer,
                         "ttft_ms": gen["ttft_ms"], "e2e_ms": gen["e2e_ms"], "tps": gen["tps"],
                         "context": [p.payload or {} for p in pts]}
                    ck.write(json.dumps(r, ensure_ascii=False) + "\n")
                    ck.flush()
                    done[(model, item["id"])] = r
                    print(f"  {item['id']} {gen['e2e_ms'] / 1000:5.1f}s  cite={citation_rate(answer):.2f}"
                          f"  refused={is_refusal(answer)}")

        # Judges nacheinander (jedes Judge-Modell einmal laden).
        with JUDGE_CKPT.open("a") as jk:
            for jm in JUDGES:
                jobs = [(f"{jm}|{m}|{it['id']}", it, done[(m, it["id"])]["answer"])
                        for m in models for it in answerable
                        if done[(m, it["id"])]["answer"] and not is_refusal(done[(m, it["id"])]["answer"])]
                for i, it in enumerate(answerable):
                    other = answerable[(i + 1) % len(answerable)]
                    jobs.append((f"{jm}|cal_pos|{it['id']}", it, it["answer"]))
                    jobs.append((f"{jm}|cal_neg|{it['id']}", it, other["answer"]))
                open_jobs = [j for j in jobs if j[0] not in verdicts]
                print(f"\nJudge {jm}: {len(open_jobs)} offene Urteile")
                for key, it, ans in open_jobs:
                    verdicts[key] = await judge(ollama, jm, it["question"], it["answer"], ans)
                    jk.write(json.dumps({"key": key, "ja": verdicts[key]}) + "\n")
                    jk.flush()
    finally:
        await ollama.aclose()
        await stores["prod"].aclose()

    print("\nLade NLI-Modell …")
    faith = Faithfulness()
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    def ok(jm, m, iid):
        return verdicts.get(f"{jm}|{m}|{iid}", False)

    rows = []
    for m in models:
        for item in gold:
            r = done[(m, item["id"])]
            refused = is_refusal(r["answer"])
            row = {"model": m, "id": item["id"], "qtype": item.get("qtype", "fact"),
                   "refused": refused, "empty": not r["answer"], "ttft_ms": r["ttft_ms"],
                   "e2e_ms": r["e2e_ms"], "tps": r["tps"], "citation_rate": citation_rate(r["answer"])}
            if item.get("spans"):
                row["hit"] = 1.0 if any(span_hit(c, item) for c in r["context"]) else 0.0
            if item.get("qtype") != "unanswerable":
                row["correct"] = 1.0 if ok(JUDGES[0], m, item["id"]) else 0.0
                row["correct_j2"] = 1.0 if ok(JUDGES[1], m, item["id"]) else 0.0
                if not refused and r["answer"]:
                    row["faithfulness"] = faith.score(r["answer"], [c.get("text", "") for c in r["context"]])
                    row["rougeL"] = rs.score(item["answer"], r["answer"])["rougeL"].fmeasure
                    row["answer_f1"] = token_f1(r["answer"], item["answer"])
            rows.append(row)

    def mean(xs, key):
        v = [x[key] for x in xs if x.get(key) is not None]
        return sum(v) / len(v) if v else float("nan")

    def median(xs, key):
        v = sorted(x[key] for x in xs)
        return v[len(v) // 2]

    summary = []
    for m in models:
        xs = [r for r in rows if r["model"] == m]
        ans = [x for x in xs if x["qtype"] != "unanswerable"]
        una = [x for x in xs if x["qtype"] == "unanswerable"]
        summary.append({
            "model": m, "correct": mean(ans, "correct"), "correct_j2": mean(ans, "correct_j2"),
            "faithfulness": mean(ans, "faithfulness"), "citation_rate": mean(ans, "citation_rate"),
            "rougeL": mean(ans, "rougeL"), "answer_f1": mean(ans, "answer_f1"),
            "refusal_rate": sum(x["refused"] for x in una) / len(una) if una else float("nan"),
            "false_refusal_rate": sum(x["refused"] for x in ans) / len(ans),
            "empty": sum(x["empty"] for x in xs),
            "ttft_median_s": median(xs, "ttft_ms") / 1000, "e2e_median_s": median(xs, "e2e_ms") / 1000,
            "tps_median": median(xs, "tps"),
        })

    paired = {}
    base = {r["id"]: r["correct"] for r in rows if r["model"] == BASELINE and "correct" in r}
    for m in models:
        if m == BASELINE:
            continue
        cur = {r["id"]: r["correct"] for r in rows if r["model"] == m and "correct" in r}
        better = sum(cur[i] > base[i] for i in base)
        worse = sum(cur[i] < base[i] for i in base)
        paired[m] = {"better": better, "worse": worse, "same": len(base) - better - worse,
                     "p_sign": round(sign_test(better, worse), 3)}

    cal = {jm: {"pos": mean([{"v": float(verdicts[f"{jm}|cal_pos|{it['id']}"])} for it in answerable], "v"),
                "neg": mean([{"v": float(verdicts[f"{jm}|cal_neg|{it['id']}"])} for it in answerable], "v")}
           for jm in JUDGES}
    keys = [(m, it["id"]) for m in models for it in answerable]
    j1 = [ok(JUDGES[0], m, i) for m, i in keys]
    j2 = [ok(JUDGES[1], m, i) for m, i in keys]
    agreement = {"agree": sum(a == b for a, b in zip(j1, j2)) / len(j1), "kappa": kappa(j1, j2)}

    print(f"\n=== Modellvergleich (n={len(gold)}, Retrieval S4) ===")
    print(f"{'Modell':<16} {'Korr14':>6} {'Korr7':>6} {'Faith':>6} {'Zitat':>6} {'ROUGE':>6} {'F1':>6} "
          f"{'Refus':>6} {'FalseR':>6} {'leer':>4} {'TTFT':>6} {'e2e':>6} {'TPS':>5}")
    for s in summary:
        print(f"{short(s['model']):<16} {s['correct']:6.3f} {s['correct_j2']:6.3f} {s['faithfulness']:6.3f} "
              f"{s['citation_rate']:6.2f} {s['rougeL']:6.3f} {s['answer_f1']:6.3f} {s['refusal_rate']:6.2f} "
              f"{s['false_refusal_rate']:6.2f} {s['empty']:4d} {s['ttft_median_s']:5.1f}s "
              f"{s['e2e_median_s']:5.1f}s {s['tps_median']:5.1f}")
    print("\nPaired Korrektheit vs. qwen2.5:7b (besser:schlechter:gleich):")
    for m, p in paired.items():
        print(f"  {short(m)}: {p['better']}:{p['worse']}:{p['same']} (p={p['p_sign']})")
    print(f"\nJudge-Kalibrierung: {cal}")
    print(f"Judge-Übereinstimmung 14b vs 7b: {agreement['agree']:.2f}, Kappa {agreement['kappa']:.2f}")

    out = ROOT / "results" / "model_compare.json"
    out.write_text(json.dumps({"n": len(gold), "retrieval": "S4", "judges": JUDGES, "summary": summary,
                               "paired": paired, "judge_calibration": cal,
                               "judge_agreement": agreement, "rows": rows}, indent=2))
    plot(summary, ROOT / "results" / "model_compare.png")
    print(f"\nErgebnis: {out}")


if __name__ == "__main__":
    asyncio.run(main())
