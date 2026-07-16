"""Metrik-Suite: Antwortqualitaet + Retrieval + Serving auf dem Gold-Set.

Pro Gold-Frage wird die volle RAG-Pipeline ausgefuehrt (retrieve -> Prompt -> LLM),
dann werden gemessen:
  - ANTWORT:  ROUGE-L, Antwort-Token-F1 (vs. Gold-Antwort), Faithfulness (NLI)
  - RETRIEVAL: Recall@k (Kapitel-Abdeckung), MRR
  - SERVING:  TTFT, TPS, e2e-Latenz (aus dem Streaming)

Faithfulness ersetzt bewusst PPL: gemessen wird, ob jede Antwort-Aussage von den
abgerufenen Passagen GESTUeTZT wird (Entailment via mDeBERTa-xnli) — nicht blosse Fluenz.

Aufruf (aus backend/):
  uv run --with rouge-score --with transformers --with torch --with sentencepiece \
         --with protobuf python ../eval/answer_eval.py
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
from app.rag.prompt_builder import build_messages  # noqa: E402
from app.rag.retriever import retrieve  # noqa: E402
from retrieval_eval import chapter_roman  # noqa: E402

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


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
        self.ent_idx = next(
            i for i, lab in self.model.config.id2label.items() if lab.lower().startswith("entail")
        )

    def _entail_prob(self, premise: str, hypothesis: str) -> float:
        inputs = self.tok(premise, hypothesis, truncation=True, max_length=512, return_tensors="pt")
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        probs = self.torch.softmax(logits, dim=-1)[0]
        return float(probs[self.ent_idx])

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
    ap.add_argument("--gold", default=str(Path(__file__).resolve().parent / "gold_dracula.jsonl"))
    ap.add_argument("--group", default="Horror")
    ap.add_argument("--model", default=settings.chat_model)
    ap.add_argument("--k", type=int, default=settings.top_k)
    args = ap.parse_args()

    from rouge_score import rouge_scorer

    gold = [json.loads(x) for x in Path(args.gold).read_text().splitlines() if x.strip()]
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    print(f"Lade NLI-Modell {NLI_MODEL} …")
    faith = Faithfulness()

    ollama = OllamaClient(settings.ollama_base_url)
    vectors = VectorStore(settings.qdrant_url, settings.qdrant_collection)

    rows = []
    try:
        for item in gold:
            # v1: relevant_chapters; v2 (gold_v2.jsonl): chapters als Vergleichs-Feld.
            relevant = {r.upper() for r in item.get("relevant_chapters") or item.get("chapters") or []}
            pts = await retrieve(
                item["question"], ollama=ollama, vectors=vectors,
                embed_model=settings.embed_model, top_k=args.k, group_id=args.group,
            )
            chunks = [(p.payload or {}).get("text", "") for p in pts]
            chapters = [chapter_roman((p.payload or {}).get("chapter")) for p in pts]

            gen = await generate(ollama, build_messages(item["question"], pts), args.model)

            rougeL = rs.score(item["answer"], gen["answer"])["rougeL"].fmeasure
            f1 = token_f1(gen["answer"], item["answer"])
            faith_score = faith.score(gen["answer"], chunks)

            hits = [i + 1 for i, c in enumerate(chapters) if c in relevant]
            recall = len({c for c in chapters[: args.k] if c in relevant}) / len(relevant)
            mrr = 1.0 / hits[0] if hits else 0.0

            rows.append({
                "id": item["id"], "rougeL": rougeL, "answer_f1": f1, "faithfulness": faith_score,
                "recall_at_k": recall, "mrr": mrr,
                "ttft_ms": gen["ttft_ms"], "tps": gen["tps"], "e2e_ms": gen["e2e_ms"],
            })
            print(f"  {item['id']}: ROUGE-L {rougeL:.2f}  F1 {f1:.2f}  Faith {faith_score:.2f}  "
                  f"Recall@{args.k} {recall:.2f}  TTFT {gen['ttft_ms']:.0f}ms  TPS {gen['tps']:.1f}")
    finally:
        await ollama.aclose()
        await vectors.aclose()

    def mean(key: str) -> float:
        return sum(r[key] for r in rows) / len(rows)

    def median(key: str) -> float:
        vals = sorted(r[key] for r in rows)
        return vals[len(vals) // 2]

    print(f"\n=== Aggregat (n={len(rows)}, Modell={args.model}, k={args.k}) ===")
    print("  ANTWORT:")
    print(f"    ROUGE-L (mean):      {mean('rougeL'):.3f}")
    print(f"    Antwort-F1 (mean):   {mean('answer_f1'):.3f}")
    print(f"    Faithfulness (mean): {mean('faithfulness'):.3f}")
    print("  RETRIEVAL:")
    print(f"    Recall@{args.k} (mean): {mean('recall_at_k'):.3f}")
    print(f"    MRR (mean):          {mean('mrr'):.3f}")
    print("  SERVING:")
    print(f"    TTFT median:         {median('ttft_ms'):.0f} ms")
    print(f"    TPS median:          {median('tps'):.1f} tok/s")
    print(f"    e2e median:          {median('e2e_ms'):.0f} ms")

    out = Path(__file__).resolve().parent.parent / "results" / "answer_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "k": args.k, "rows": rows}, indent=2))
    print(f"\nRoh-Ergebnisse: {out}")


if __name__ == "__main__":
    asyncio.run(main())
