"""Exp 9 — Graph-RAG global: helfen Community-Berichte bei THEMATISCHEN Fragen?

Das Span-Gold-Set (Exp 8) misst Passagen-Retrieval; dort kann ein globaler Pfad strukturell
nicht gewinnen. Deshalb ein eigenes Gold-Set `eval/gold_global.jsonl` (thematische Fragen mit
Rubrik = 3-5 Kernpunkte) plus Kontroll-Items aus gold_v2 (Faktfragen, Rubrik = Gold-Antwort),
bei denen der globale Pfad erwartungsgemäß verlieren sollte.

Arme (paired, gleiche Fragen, temp 0):
  dense8        heutiger Pfad (k=8)                 dense16       stärkere Baseline (k=16)
  global_direct Top-m Berichte + 2 Passagen je Bericht, 1 LLM-Aufruf
  global_map    dito + Map-Step (je Bericht 1 Aufruf: relevante Punkte 0-100) -> m+1 Aufrufe

Metriken je Item: Rubrik-Abdeckung (LLM-Judge, EIN Aufruf je Kernpunkt, JA/NEIN, blind,
gemischte Reihenfolge, mit Kalibrierung: Verweigerung ≈ 0 %, Rubrik als Antwort ≈ 100 %),
NLI-Faithfulness gegen Buch-Chunks (faith_chunks, für alle Arme gleich definiert) und gegen
alle Quellen (faith_sources), TTFT/e2e/Antwortlänge/LLM-Aufrufe/Zitatquote.

Erfolgskriterien (vorab, s. RESULTS.md Exp 9): Coverage global >= dense16 + 0.15 UND paired
>= 6/10 besser UND faith_chunks >= dense16 - 0.15. Negativ: Coverage <= dense16 oder
faith_chunks < 0.35 oder Judge-Kalibrierung außerhalb [<= 20 %, >= 80 %].

Aufruf (aus backend/; Community-Index vorher via `make graph-communities`):
  uv run --with rouge-score --with transformers --with torch --with sentencepiece \\
      --with protobuf --with matplotlib python ../eval/global_graph_experiment.py \\
      --gold ../eval/gold_global.jsonl --group Horror --m 6 --control 3
  --no-nli überspringt die NLI-Phase (schnell, ohne torch); --resume nutzt den Antwort-Checkpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.clients.ollama_client import OllamaClient  # noqa: E402
from app.clients.qdrant_client import VectorStore  # noqa: E402
from app.config import settings  # noqa: E402
from app.graph.communities import CommunityIndex  # noqa: E402
from app.graph.global_search import (  # noqa: E402
    GLOBAL_SYSTEM_EXTRA,
    global_retrieve,
    is_community_source,
)
from app.rag.prompt_builder import build_messages  # noqa: E402
from app.rag.retriever import retrieve  # noqa: E402
from answer_eval import generate, is_refusal  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
ANSWERS_FILE = RESULTS_DIR / "global_graph_answers.jsonl"
ARMS = ["dense8", "dense16", "global_direct", "global_map"]
MIN_CONTEXT = 12000  # dense16 ≈ 8-10k Token, global ≈ 9k; Ollama-Default 4096 schneidet still ab
REFUSAL_TEXT = "Dazu steht in den Quellen nichts."
CONTROL_IDS = ["d02", "d03", "n09"]

JUDGE_PROMPT = (
    "Frage: {question}\n\nAntwort: {answer}\n\nKernpunkt: {point}\n\n"
    "Enthält die Antwort diesen Kernpunkt inhaltlich ausdrücklich (nicht nur andeutungsweise)? "
    "Antworte NUR mit JA oder NEIN."
)
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_CITE_RE = re.compile(r"\[\d+\]")


def citation_rate(answer: str) -> float:
    sents = [s for s in _SENT_RE.split(answer) if len(s.strip()) > 15]
    return sum(1 for s in sents if _CITE_RE.search(s)) / len(sents) if sents else 0.0


async def judge(ollama: OllamaClient, model: str, question: str, answer: str, point: str) -> bool:
    try:
        out = await ollama.complete(
            [{"role": "user", "content": JUDGE_PROMPT.format(question=question, answer=answer,
                                                              point=point)}],
            model, temperature=0.0, max_tokens=3,
        )
    except Exception:
        return False
    return out.strip().upper().startswith("JA")


async def ensure_context(ollama: OllamaClient, model: str, min_ctx: int) -> None:
    """Bricht ab, wenn das Modell mit zu kleinem Kontextfenster läuft (sonst ist jede Zahl Müll)."""
    await ollama.complete([{"role": "user", "content": "OK"}], model, max_tokens=1)  # laden
    ctx = await ollama.context_length(model)
    if ctx is None or ctx < min_ctx:
        sys.exit(f"Ollama läuft {model} mit Kontext {ctx} (< {min_ctx}): Prompts würden still "
                 "abgeschnitten. Fix: `make ollama-ctx`, dann Ollama-App beenden UND neu öffnen "
                 "(launchd-Env greift nur bei Neustart über Finder/Dock); prüfen mit `ollama ps`.")
    print(f"  Kontextfenster {model}: {ctx} Token — ok")


def load_gold(args: argparse.Namespace) -> list[dict]:
    gold = [json.loads(x) for x in Path(args.gold).read_text().splitlines() if x.strip()]
    if args.control:
        v2 = [json.loads(x) for x in Path(args.gold_v2).read_text().splitlines() if x.strip()]
        by_id = {g["id"]: g for g in v2}
        picks = [by_id[i] for i in CONTROL_IDS if i in by_id and by_id[i].get("answer")]
        if len(picks) < args.control:
            rng = random.Random(args.seed)
            pool = [g for g in v2 if g.get("qtype") == "fact" and g.get("answer") and g not in picks]
            picks += rng.sample(pool, min(args.control - len(picks), len(pool)))
        for g in picks[: args.control]:
            gold.append({"id": g["id"], "question": g["question"], "qtype": "control",
                         "rubric": [g["answer"]], "book_id": g.get("book_id")})
    return gold


def save_answers(rows: dict[tuple[str, str], dict]) -> None:
    """Checkpoint komplett neu schreiben — auch Judge-/NLI-Felder werden so resumierbar."""
    tmp = ANSWERS_FILE.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows.values()),
                   encoding="utf-8")
    tmp.replace(ANSWERS_FILE)


def load_answers() -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    if ANSWERS_FILE.exists():
        for line in ANSWERS_FILE.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r["id"], r["arm"])] = r
    return rows


async def answer_item(
    arm: str, item: dict, *, ollama: OllamaClient, vectors: VectorStore,
    index: CommunityIndex, args: argparse.Namespace,
) -> dict:
    meta: dict = {}
    if arm.startswith("dense"):
        k = args.k if arm == "dense8" else args.k_strong
        pts = await retrieve(item["question"], ollama=ollama, vectors=vectors,
                             embed_model=settings.embed_model, top_k=k, group_id=args.group)
        messages = build_messages(item["question"], pts)
    else:
        pts, meta = await global_retrieve(
            item["question"], index=index, ollama=ollama, vectors=vectors,
            embed_model=settings.embed_model, model=args.model, m=args.m,
            use_map=(arm == "global_map"), chunks_per_community=args.chunks_per_community,
        )
        messages = build_messages(item["question"], pts, system_extra=GLOBAL_SYSTEM_EXTRA,
                                  hint=meta.get("hint") or None)
    gen = await generate(ollama, messages, args.model)
    return {
        "id": item["id"], "arm": arm, "qtype": item.get("qtype", "global"),
        "answer": gen["answer"], "refused": is_refusal(gen["answer"]),
        "ttft_ms": round(gen["ttft_ms"]), "e2e_ms": round(gen["e2e_ms"]),
        "completion_tokens": gen["completion_tokens"],
        "n_llm_calls": 1 + int(meta.get("n_llm_calls", 0)),
        "prompt_chars": sum(len(m["content"]) for m in messages),
        "sources": [{"kind": "community" if is_community_source(p.payload) else "chunk",
                     "chapter": (p.payload or {}).get("chapter"),
                     "text": (p.payload or {}).get("text", "")} for p in pts],
        "community_ids": meta.get("community_ids", []),
        "map_scores": meta.get("map_scores", {}),
    }


def mean(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


async def run(args: argparse.Namespace) -> None:
    gold = load_gold(args)
    index = CommunityIndex.load(Path(args.graph_dir) / args.group)
    if index is None or index.vectors is None:
        sys.exit("Kein Community-Index (partition.json/communities.jsonl/community_vectors.npz) — "
                 "erst `make graph-communities`.")
    arms = [a for a in ARMS if a in args.arms.split(",")]
    print(f"Exp 9: {len(gold)} Fragen ({sum(1 for g in gold if g['qtype'] == 'global')} global, "
          f"{sum(1 for g in gold if g['qtype'] == 'control')} Kontrolle), Arme {arms}, "
          f"{len(index.reports)} Community-Berichte, m={args.m}")

    ollama = OllamaClient(settings.ollama_base_url, timeout=600.0)
    vectors = VectorStore(settings.qdrant_url, settings.qdrant_collection)
    rows = load_answers() if args.resume else {}
    RESULTS_DIR.mkdir(exist_ok=True)
    try:
        await ensure_context(ollama, args.model, MIN_CONTEXT)
        # --- Phase 1: Antworten (teuer) — Checkpoint je (Item, Arm) -------------------
        with ANSWERS_FILE.open("a" if args.resume else "w", encoding="utf-8") as f:
            for item in gold:
                for arm in arms:
                    if (item["id"], arm) in rows:
                        continue
                    row = await answer_item(arm, item, ollama=ollama, vectors=vectors,
                                            index=index, args=args)
                    rows[(item["id"], arm)] = row
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"  {item['id']} {arm:<13} {row['e2e_ms'] / 1000:5.1f}s "
                          f"{len(row['answer']):5d} Zeichen  calls={row['n_llm_calls']}"
                          f"{'  VERWEIGERT' if row['refused'] else ''}")

        # --- Phase 2: Judge (blind, gemischte Reihenfolge) + Kalibrierung ------------
        tasks = [(item, arm) for item in gold for arm in arms
                 if "coverage" not in rows[(item["id"], arm)]]
        random.Random(args.seed).shuffle(tasks)
        print(f"Judge ({args.judge_model}): {len(tasks)} Antworten zu bewerten "
              f"({len(gold) * len(arms) - len(tasks)} aus Checkpoint) …")
        for n, (item, arm) in enumerate(tasks, start=1):
            row = rows[(item["id"], arm)]
            verdicts = [await judge(ollama, args.judge_model, item["question"], row["answer"], p)
                        for p in item["rubric"]]
            row["per_point"] = verdicts
            row["coverage"] = sum(verdicts) / len(verdicts)
            save_answers(rows)
            if n % 10 == 0 or n == len(tasks):
                print(f"  {n}/{len(tasks)} bewertet")
        calibration = {"refusal": [], "rubric": []}
        for item in gold:
            low = [await judge(ollama, args.judge_model, item["question"], REFUSAL_TEXT, p)
                   for p in item["rubric"]]
            high = [await judge(ollama, args.judge_model, item["question"],
                                " ".join(item["rubric"]), p) for p in item["rubric"]]
            calibration["refusal"].append(sum(low) / len(low))
            calibration["rubric"].append(sum(high) / len(high))
        cal_low, cal_high = mean(calibration["refusal"]), mean(calibration["rubric"])
        print(f"  Kalibrierung: Verweigerung -> {cal_low:.0%} (soll <= 20 %), "
              f"Rubrik als Antwort -> {cal_high:.0%} (soll >= 80 %)")
    finally:
        await ollama.aclose()
        await vectors.aclose()

    # --- Phase 3: NLI-Faithfulness (Modell erst jetzt laden) ---------------------------
    if not args.no_nli:
        from answer_eval import Faithfulness  # torch/transformers erst hier

        todo = [r for r in rows.values() if "faith_sources" not in r]
        print(f"NLI-Modell laden … ({len(todo)} Antworten zu prüfen, "
              f"{len(rows) - len(todo)} aus Checkpoint; ~1 s je Satz×Passage auf CPU)")
        faith = Faithfulness()
        for n, row in enumerate(todo, start=1):
            if row["refused"]:
                row["faith_chunks"] = row["faith_sources"] = None
            else:
                chunks = [s["text"] for s in row["sources"] if s["kind"] == "chunk"]
                row["faith_chunks"] = faith.score(row["answer"], chunks) if chunks else None
                row["faith_sources"] = faith.score(row["answer"],
                                                   [s["text"] for s in row["sources"]])
            save_answers(rows)
            print(f"  NLI {n}/{len(todo)}  {row['id']} {row['arm']:<13} "
                  f"faith_chunks={fmt(row['faith_chunks'])} faith_sources={fmt(row['faith_sources'])}")
    else:
        for row in rows.values():
            row["faith_chunks"] = row["faith_sources"] = None

    # --- Bericht ---------------------------------------------------------------------
    report = _report(gold, arms, rows, cal_low, cal_high, args)
    out = {"group": args.group, "m": args.m, "k": args.k, "k_strong": args.k_strong,
           "model": args.model, "judge_model": args.judge_model,
           "communities": len(index.reports), "calibration": {"refusal": cal_low, "rubric": cal_high},
           "arms": report, "rows": [dict(r, manual_coverage=None) for r in rows.values()]}
    (RESULTS_DIR / "global_graph_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  JSON: {RESULTS_DIR / 'global_graph_eval.json'}")
    _chart(gold, arms, rows, report)


def _report(gold, arms, rows, cal_low, cal_high, args) -> dict:
    glob = [g for g in gold if g["qtype"] == "global"]
    ctrl = [g for g in gold if g["qtype"] == "control"]
    summary: dict = {}
    print(f"\n=== Exp 9 — Graph-RAG global (n_global={len(glob)}, n_control={len(ctrl)}, "
          f"m={args.m}) ===")
    print(f"  {'Arm':<14} {'Cov':>5} {'CovCtrl':>8} {'faithC':>7} {'faithS':>7} {'Zitat':>6} "
          f"{'TTFT s':>7} {'e2e s':>6} {'Zeich.':>7} {'Calls':>5} {'Verw.':>5}")
    for arm in arms:
        rs = [rows[(g["id"], arm)] for g in glob]
        rc = [rows[(g["id"], arm)] for g in ctrl]
        s = {
            "coverage": mean([r["coverage"] for r in rs]),
            "coverage_control": mean([r["coverage"] for r in rc]) if rc else None,
            "faith_chunks": mean([r["faith_chunks"] for r in rs]),
            "faith_sources": mean([r["faith_sources"] for r in rs]),
            "citation_rate": mean([citation_rate(r["answer"]) for r in rs]),
            "ttft_s": mean([r["ttft_ms"] / 1000 for r in rs]),
            "e2e_s": mean([r["e2e_ms"] / 1000 for r in rs]),
            "chars": mean([len(r["answer"]) for r in rs]),
            "llm_calls": mean([r["n_llm_calls"] for r in rs]),
            "refusals": sum(1 for r in rs if r["refused"]),
        }
        summary[arm] = s
        print(f"  {arm:<14} {fmt(s['coverage']):>5} {fmt(s['coverage_control']):>8} "
              f"{fmt(s['faith_chunks']):>7} {fmt(s['faith_sources']):>7} {fmt(s['citation_rate']):>6} "
              f"{fmt(s['ttft_s'], 1):>7} {fmt(s['e2e_s'], 1):>6} {fmt(s['chars'], 0):>7} "
              f"{fmt(s['llm_calls'], 1):>5} {s['refusals']:>5}")

    base = "dense16" if "dense16" in arms else arms[0]
    for arm in arms:
        if not arm.startswith("global"):
            continue
        b = w = e = 0
        for g in glob:
            d = rows[(g["id"], arm)]["coverage"] - rows[(g["id"], base)]["coverage"]
            b += d > 0
            w += d < 0
            e += d == 0
        summary[arm]["paired_vs_" + base] = {"better": b, "worse": w, "same": e}
        d_cov = (summary[arm]["coverage"] or 0) - (summary[base]["coverage"] or 0)
        fc, fb = summary[arm]["faith_chunks"], summary[base]["faith_chunks"]
        verdict = "unentschieden/gemischt"
        if cal_low is not None and (cal_low > 0.2 or (cal_high or 0) < 0.8):
            verdict = "NICHT INTERPRETIERBAR (Judge-Kalibrierung außerhalb des Bandes)"
        elif d_cov >= 0.15 and b >= 6 and (fc is None or fb is None or fc >= fb - 0.15):
            verdict = "POSITIV"
        elif d_cov <= 0 or (fc is not None and fc < 0.35):
            verdict = "NEGATIV"
        summary[arm]["verdict"] = verdict
        print(f"  Paired {arm} vs {base} (Coverage): {b} besser / {w} schlechter / {e} gleich, "
              f"ΔCov={d_cov:+.2f} -> {verdict}")
    print("\n  Je Frage (Coverage):")
    print(f"  {'id':<5} " + " ".join(f"{a:>13}" for a in arms))
    for g in gold:
        print(f"  {g['id']:<5} " + " ".join(f"{rows[(g['id'], a)]['coverage']:>13.2f}" for a in arms)
              + ("   (Kontrolle)" if g["qtype"] == "control" else ""))
    return summary


def _chart(gold, arms, rows, report) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (kein matplotlib — Chart übersprungen)")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    metrics = [("Coverage", "coverage"), ("Cov. Kontrolle", "coverage_control"),
               ("faith_chunks", "faith_chunks"), ("faith_sources", "faith_sources")]
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        ys = [report[arm][key] or 0 for _, key in metrics]
        bars = ax1.bar([j + i * width for j in range(len(metrics))], ys, width, label=arm)
        ax1.bar_label(bars, fmt="%.2f", fontsize=7)
    ax1.set_xticks([j + width * (len(arms) - 1) / 2 for j in range(len(metrics))])
    ax1.set_xticklabels([m for m, _ in metrics])
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Exp 9 — global vs. dense (Mittelwerte)")
    ax1.legend(fontsize=8)
    base = "dense16" if "dense16" in arms else arms[0]
    other = "global_direct" if "global_direct" in arms else arms[-1]
    glob = [g for g in gold if g["qtype"] == "global"]
    xs = [rows[(g["id"], base)]["coverage"] for g in glob]
    ys = [rows[(g["id"], other)]["coverage"] for g in glob]
    ax2.scatter(xs, ys)
    for g, x, y in zip(glob, xs, ys, strict=True):
        ax2.annotate(g["id"], (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax2.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax2.set_xlabel(f"Coverage {base}")
    ax2.set_ylabel(f"Coverage {other}")
    ax2.set_title("je Frage (über der Diagonale = global besser)")
    fig.tight_layout()
    path = RESULTS_DIR / "global_graph_experiment.png"
    fig.savefig(path, dpi=150)
    print(f"  Chart: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(Path(__file__).resolve().parent / "gold_global.jsonl"))
    ap.add_argument("--gold-v2", default=str(Path(__file__).resolve().parent / "gold_v2.jsonl"))
    ap.add_argument("--group", default="Horror")
    ap.add_argument("--graph-dir", default=settings.graph_dir)
    ap.add_argument("--m", type=int, default=settings.graph_global_m)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--k-strong", type=int, default=16)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--model", default=settings.chat_model)
    ap.add_argument("--judge-model", default=settings.chat_model)
    ap.add_argument("--chunks-per-community", type=int, default=settings.graph_global_chunks_per_community)
    ap.add_argument("--control", type=int, default=3, help="Kontroll-Items aus gold_v2 (Faktfragen)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="Antwort-Checkpoint wiederverwenden")
    ap.add_argument("--no-nli", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
