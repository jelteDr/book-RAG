"""Metrik-Suite: Antwortqualitaet + Retrieval + Serving auf dem Gold-Set.

Pro Gold-Frage wird die volle RAG-Pipeline ausgefuehrt (retrieve -> Prompt -> LLM),
dann werden gemessen:
  - ANTWORT:   ROUGE-L, Antwort-Token-F1 (vs. Gold-Antwort), Faithfulness (NLI)
  - EHRLICHKEIT: Refusal-Rate auf unanswerable-Items (sagt das Modell "steht nicht
    in den Quellen"?) und False-Refusal-Rate auf beantwortbaren Items
  - RETRIEVAL: Hit-Rate@k + MRR — span-basiert (Gold-Set v2), Kapitel als Fallback
  - SERVING:   TTFT, TPS, e2e-Latenz (aus dem Streaming)

Faithfulness ersetzt bewusst PPL: gemessen wird, ob jede Antwort-Aussage von den
abgerufenen Passagen GESTUeTZT wird (Entailment via mDeBERTa-xnli) — nicht blosse
Fluenz. Verweigerte Antworten gehen nicht in ROUGE/F1/Faithfulness ein (eine
Verweigerung stellt keine stuetzbaren Behauptungen auf), sondern in die
Refusal-Metriken.

Diese Suite ist der STANDARD-ABSCHLUSS jedes Experiments: Retrieval-Metriken
allein zeigen nicht, ob die Antwort dahinter besser wird.

Aufruf (aus backend/):
  uv run --with rouge-score --with transformers --with torch --with sentencepiece \
         --with protobuf python ../eval/answer_eval.py
Optional: --s2b aktiviert Small-to-Big (Exp 7) unabhaengig vom Env-Flag.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.clients.ollama_client import OllamaClient  # noqa: E402
from app.clients.qdrant_client import VectorStore  # noqa: E402
from app.config import settings  # noqa: E402
from app.graph.service import GraphService  # noqa: E402
from app.rag.expander import expand_points  # noqa: E402
from app.rag.prompt_builder import build_messages  # noqa: E402
from app.rag.retriever import retrieve  # noqa: E402
from retrieval_eval import chapter_roman, span_hit  # noqa: E402

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# Verweigerungs-Erkennung: typische Formulierungen des Prompt-Contracts ("Dazu
# steht in den Quellen nichts.") plus Varianten; zusaetzlich gilt eine kurze
# Antwort ganz ohne [n]-Zitat als Verweigerung.
_REFUSAL_RE = re.compile(
    r"(steht|finde|findet\s+sich|gibt\s+es)[^.]{0,60}"
    r"(quellen|texten?)[^.]{0,40}(nichts|keine|nicht)"
    r"|quellen\s+(reichen|enthalten)\s+nicht"
    r"|keine\s+(information|angabe|informationen|angaben)",
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    if _REFUSAL_RE.search(answer):
        return True
    return "[" not in answer and len(answer) < 200


def normalize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


async def generate(ollama: OllamaClient, messages: list[dict], model: str) -> dict:
    t0 = time.perf_counter()
    ttft = None
    usage: dict = {}
    parts: list[str] = []
    async for line in ollama.chat_stream(messages, model, temperature=0.0):
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if chunk.get("usage"):
            usage = chunk["usage"]
        for ch in chunk.get("choices", []):
            delta = ch.get("delta", {}).get("content")
            if delta:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                parts.append(delta)
    e2e = (time.perf_counter() - t0) * 1000
    ctoks = usage.get("completion_tokens") or 0
    decode = max(e2e - (ttft or 0), 1e-6)
    return {
        "answer": "".join(parts),
        "ttft_ms": ttft or e2e,
        "e2e_ms": e2e,
        "completion_tokens": ctoks,
        "tps": (ctoks / (decode / 1000)) if ctoks else 0.0,
    }


class Faithfulness:
    """Anteil der Antwort-Saetze, die von mind. einer Passage gestuetzt werden (Entailment)."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        self.model.eval()
        # Apple-GPU (MPS) statt CPU: die NLI-Phase war auf CPU der laengste Teil (~1 s je
        # Satz x Passage); auf MPS um ein Vielfaches schneller, Ergebnisse identisch (float32).
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)
        self.ent_idx = next(
            i for i, lab in self.model.config.id2label.items() if lab.lower().startswith("entail")
        )

    def _entail_prob(self, premise: str, hypothesis: str) -> float:
        inputs = self.tok(premise, hypothesis, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        probs = self.torch.softmax(logits, dim=-1)[0]
        return float(probs[self.ent_idx].cpu())

    def score(self, answer: str, chunks: list[str], threshold: float = 0.5) -> float:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 15]
        if not sentences:
            return 0.0
        supported = 0
        for sent in sentences:
            best = max((self._entail_prob(c, sent) for c in chunks[:5]), default=0.0)
            if best >= threshold:
                supported += 1
        return supported / len(sentences)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(Path(__file__).resolve().parent / "gold_v2.jsonl"))
    ap.add_argument("--group", default="Horror")
    ap.add_argument("--model", default=settings.chat_model)
    ap.add_argument("--k", type=int, default=settings.top_k)
    ap.add_argument("--s2b", action="store_true",
                    help="Small-to-Big (Exp 7) auf die Treffer anwenden, wie im Prod-Pfad")
    ap.add_argument("--s2b-window", type=int, default=settings.s2b_window)
    ap.add_argument("--s2b-top-n", type=int, default=settings.s2b_top_n)
    ap.add_argument("--graph", choices=["off", "link", "expand"], default="off",
                    help="Graph-RAG-Arm (Exp 8) statt dense Retrieval; Graph aus --graph-dir")
    ap.add_argument("--graph-dir", default=settings.graph_dir)
    ap.add_argument("--out", help="Ergebnis-JSON (Default results/answer_eval.json)")
    args = ap.parse_args()
    graph = None
    if args.graph != "off":
        graph = GraphService(args.graph_dir, True, "local", local_mode=args.graph,
                             alpha=settings.graph_alpha, top_m=settings.graph_top_m,
                             min_sim=settings.graph_min_sim)
        if not graph.active:
            sys.exit(f"Kein Graph unter {args.graph_dir} — erst make graph + make graph-build.")

    from rouge_score import rouge_scorer

    gold = [json.loads(x) for x in Path(args.gold).read_text().splitlines() if x.strip()]
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    print(f"Lade NLI-Modell {NLI_MODEL} …")
    faith = Faithfulness()

    ollama = OllamaClient(settings.ollama_base_url)
    vectors = VectorStore(settings.qdrant_url, settings.qdrant_collection)

    rows = []
    try:
        # Kontext-Guard: RAG-Prompts haben ~4-6k Token; Ollama-Default 4096 (2 Slots à 2048)
        # schneidet still ab und entwertet jede Messung (s. RESULTS.md, Exp 7 Nebenbefund).
        await ollama.complete([{"role": "user", "content": "OK"}], args.model, max_tokens=1)
        ctx = await ollama.context_length(args.model)
        if ctx is None or ctx < 8000:
            sys.exit(f"Ollama läuft {args.model} mit Kontext {ctx} (< 8000) — `make ollama-ctx` + "
                     "Ollama-App neu starten, dann erneut.")
        print(f"Kontextfenster {args.model}: {ctx} Token — ok")
        for item in gold:
            unanswerable = item.get("qtype") == "unanswerable"
            retriever = graph.retrieve if graph is not None else retrieve
            pts = await retriever(
                item["question"], ollama=ollama, vectors=vectors,
                embed_model=settings.embed_model, top_k=args.k, group_id=args.group,
            )
            if args.s2b:
                pts = await expand_points(
                    pts, vectors=vectors, window=args.s2b_window, top_n=args.s2b_top_n
                )
            payloads = [p.payload or {} for p in pts]
            chunks = [pl.get("text", "") for pl in payloads]

            gen = await generate(ollama, build_messages(item["question"], pts), args.model)
            refused = is_refusal(gen["answer"])

            row = {
                "id": item["id"], "qtype": item.get("qtype", "fact"), "refused": refused,
                "ttft_ms": gen["ttft_ms"], "tps": gen["tps"], "e2e_ms": gen["e2e_ms"],
            }

            # Retrieval: span-basiert (v2); Kapitel nur als Fallback fuer v1-Gold-Dateien.
            if item.get("spans"):
                flags = [span_hit(pl, item) for pl in payloads]
            else:
                relevant = {r.upper() for r in item.get("relevant_chapters") or []}
                flags = [chapter_roman(pl.get("chapter")) in relevant for pl in payloads] \
                    if relevant else []
            if flags:
                first = next((i + 1 for i, f in enumerate(flags) if f), None)
                row["hit_at_k"] = 1.0 if first else 0.0
                row["mrr"] = 1.0 / first if first else 0.0

            # Antwort-Metriken nur fuer beantwortete beantwortbare Items — eine
            # Verweigerung stellt keine Behauptungen auf, die NLI stuetzen koennte.
            if not unanswerable and not refused:
                row["rougeL"] = rs.score(item["answer"], gen["answer"])["rougeL"].fmeasure
                row["answer_f1"] = token_f1(gen["answer"], item["answer"])
                row["faithfulness"] = faith.score(gen["answer"], chunks)

            rows.append(row)
            detail = (
                f"refused={'JA' if refused else 'nein'}" if unanswerable or refused
                else f"ROUGE-L {row['rougeL']:.2f}  F1 {row['answer_f1']:.2f}  "
                     f"Faith {row['faithfulness']:.2f}"
            )
            rank = f"@{int(1 / row['mrr'])}" if row.get("mrr") else ("MISS" if flags else "—")
            print(f"  {item['id']} [{row['qtype']}]: {detail}  span={rank}  "
                  f"TTFT {gen['ttft_ms']:.0f}ms  TPS {gen['tps']:.1f}")
    finally:
        await ollama.aclose()
        await vectors.aclose()

    def mean(sel: list[dict], key: str) -> float:
        vals = [r[key] for r in sel if key in r]
        return sum(vals) / len(vals) if vals else float("nan")

    def median(sel: list[dict], key: str) -> float:
        vals = sorted(r[key] for r in sel if key in r)
        return vals[len(vals) // 2] if vals else float("nan")

    answerable = [r for r in rows if r["qtype"] != "unanswerable"]
    unanswer = [r for r in rows if r["qtype"] == "unanswerable"]
    answered = [r for r in answerable if not r["refused"]]

    print(f"\n=== Aggregat (n={len(rows)}, Modell={args.model}, k={args.k}, "
          f"s2b={'an' if args.s2b else 'aus'}, graph={args.graph}) ===")
    print(f"  ANTWORT (beantwortet, n={len(answered)}):")
    print(f"    ROUGE-L (mean):      {mean(answered, 'rougeL'):.3f}")
    print(f"    Antwort-F1 (mean):   {mean(answered, 'answer_f1'):.3f}")
    print(f"    Faithfulness (mean): {mean(answered, 'faithfulness'):.3f}")
    if unanswer:
        hon = sum(r["refused"] for r in unanswer)
        print("  EHRLICHKEIT:")
        print(f"    Refusal-Rate (unanswerable, hoeher=besser): "
              f"{hon / len(unanswer):.2f}  ({hon}/{len(unanswer)})")
    false_ref = sum(r["refused"] for r in answerable)
    print(f"    False-Refusal-Rate (beantwortbar, niedriger=besser): "
          f"{false_ref / len(answerable):.2f}  ({false_ref}/{len(answerable)})")
    print(f"  RETRIEVAL (span-basiert, n={sum(1 for r in answerable if 'mrr' in r)}):")
    print(f"    Hit-Rate@{args.k} (mean): {mean(answerable, 'hit_at_k'):.3f}")
    print(f"    MRR (mean):          {mean(answerable, 'mrr'):.3f}")
    print("  SERVING:")
    print(f"    TTFT median:         {median(rows, 'ttft_ms'):.0f} ms")
    print(f"    TPS median:          {median(rows, 'tps'):.1f} tok/s")
    print(f"    e2e median:          {median(rows, 'e2e_ms'):.0f} ms")

    out = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "results" / "answer_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"model": args.model, "k": args.k, "s2b": args.s2b, "graph": args.graph, "rows": rows},
        indent=2
    ))
    print(f"\nRoh-Ergebnisse: {out}")


if __name__ == "__main__":
    asyncio.run(main())
