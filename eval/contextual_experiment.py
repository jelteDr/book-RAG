"""Exp 5 — Contextual Retrieval: hilft LLM-generierter Chunk-Kontext dem Retrieval?

Idee (nach Anthropics „Contextual Retrieval"): Ein Chunk mitten aus Kapitel 12 weiß
nicht, dass „he" Jonathan Harker ist. Beim Ingest generiert ein LLM deshalb 1–2 Sätze
Kontext („Buch X, Kapitel Y; es geht um …"), die NUR ins Embedding eingehen — der
anzeigbare Chunk-Text bleibt unverändert.

Ablauf (resumierbar, Kontexte werden nach jedem Chunk weggeschrieben):
  1. Alle Chunks der Gruppe aus der Produktiv-Collection lesen (scroll).
  2. Pro Chunk Kontext via Ollama generieren (temp=0) — Checkpoint-Datei erlaubt Resume.
  3. Kontext+Text embedden und in eine SEPARATE Collection (book_chunks_ctx) upserten.
  4. Paired Eval auf dem Gold-Set: baseline (Produktiv-Collection) vs. ctx —
     kapitel-basierte Hit-Rate@k/MRR wie Exp 1–4.

Aufruf (aus dem backend-Verzeichnis):
  uv run python ../eval/contextual_experiment.py --group Horror --k 8
  uv run python ../eval/contextual_experiment.py --group Horror --limit 8   # Smoke-Test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.clients.ollama_client import OllamaClient  # noqa: E402
from app.clients.qdrant_client import VectorStore  # noqa: E402
from app.config import settings  # noqa: E402
from app.util import l2_normalize  # noqa: E402
from qdrant_client import models  # noqa: E402

CTX_COLLECTION = "book_chunks_ctx"
_ROMAN_RE = re.compile(r"CHAPTER\s+([IVXLCDM]+)", re.IGNORECASE)

_CTX_PROMPT = (
    "Here is an excerpt from the novel \"{title}\" ({chapter}).\n\n"
    "---\n{text}\n---\n\n"
    "Write 1-2 short English sentences situating this excerpt within the novel: "
    "name the characters involved (resolve pronouns), the location and what is "
    "happening. Answer with ONLY those sentences."
)


def chapter_roman(chapter: str | None) -> str | None:
    m = _ROMAN_RE.search(chapter or "")
    return m.group(1).upper() if m else None


def _key(payload: dict) -> str:
    return f"{payload['book_id']}:{payload['chunk_index']}"


async def generate_contexts(
    ollama: OllamaClient, records: list, ctx_path: Path, model: str
) -> dict[str, str]:
    """Kontexte erzeugen; bereits vorhandene (Checkpoint-Datei) werden übersprungen."""
    contexts: dict[str, str] = {}
    if ctx_path.exists():
        for line in ctx_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                contexts[row["key"]] = row["context"]
    todo = [r for r in records if _key(r.payload) not in contexts]
    print(f"Kontexte: {len(contexts)} vorhanden, {len(todo)} zu generieren")

    with ctx_path.open("a", encoding="utf-8") as f:
        for i, rec in enumerate(todo, start=1):
            pl = rec.payload
            prompt = _CTX_PROMPT.format(
                title=pl.get("book_title") or "?",
                chapter=pl.get("chapter") or "beginning of the book",
                text=pl["text"][:2400],
            )
            try:
                ctx = await ollama.complete(
                    [{"role": "user", "content": prompt}], model, temperature=0.0, max_tokens=90
                )
            except Exception as exc:
                print(f"  WARN {_key(pl)}: {exc} — Chunk übersprungen")
                continue
            ctx = " ".join(ctx.split())
            contexts[_key(pl)] = ctx
            f.write(json.dumps({"key": _key(pl), "context": ctx}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} Kontexte generiert")
    return contexts


async def build_ctx_collection(
    ollama: OllamaClient, records: list, contexts: dict[str, str], batch: int = 32
) -> None:
    """Kontext+Text embedden und in die Experiment-Collection upserten."""
    ctx_store = VectorStore(settings.qdrant_url, CTX_COLLECTION)
    try:
        await ctx_store.ensure_collection()
        done = 0
        for start in range(0, len(records), batch):
            part = records[start : start + batch]
            texts = [f"{contexts.get(_key(r.payload), '')}\n\n{r.payload['text']}" for r in part]
            embs = await ollama.embed(texts, settings.embed_model)
            points = [
                {"vector": l2_normalize(e), "payload": r.payload}
                for r, e in zip(part, embs)
            ]
            await ctx_store.upsert_chunks(points)
            done += len(part)
            if done % (batch * 4) == 0 or done == len(records):
                print(f"  {done}/{len(records)} embedded + upserted")
    finally:
        await ctx_store.aclose()


async def evaluate(ollama: OllamaClient, gold: list[dict], group_id: str, k: int) -> None:
    base_store = VectorStore(settings.qdrant_url, settings.qdrant_collection)
    ctx_store = VectorStore(settings.qdrant_url, CTX_COLLECTION)
    flt = models.Filter(
        must=[models.FieldCondition(key="group_id", match=models.MatchValue(value=group_id))]
    )
    rows = []
    try:
        for item in gold:
            relevant = {r.upper() for r in item["relevant_chapters"]}
            emb = l2_normalize((await ollama.embed([item["question"]], settings.embed_model))[0])
            arms = {}
            for name, store in (("base", base_store), ("ctx", ctx_store)):
                pts = await store.search(emb, k, flt)
                ranks = [chapter_roman((p.payload or {}).get("chapter")) for p in pts]
                hits = [i + 1 for i, r in enumerate(ranks) if r in relevant]
                arms[name] = (bool(hits), 1 / hits[0] if hits else 0.0)
            rows.append({"id": item["id"], **{f"{n}_{m}": v for n, (h, r) in arms.items()
                                              for m, v in (("hit", h), ("rr", r))}})
    finally:
        await base_store.aclose()
        await ctx_store.aclose()

    n = len(rows)
    print(f"\n=== Exp 5: Contextual Retrieval (n={n}, k={k}, kapitel-basiert, paired) ===")
    print(f"Hit-Rate@{k}:  baseline {sum(r['base_hit'] for r in rows) / n:.3f}   "
          f"ctx {sum(r['ctx_hit'] for r in rows) / n:.3f}")
    print(f"MRR:          baseline {sum(r['base_rr'] for r in rows) / n:.3f}   "
          f"ctx {sum(r['ctx_rr'] for r in rows) / n:.3f}")
    better = sum(1 for r in rows if r["ctx_rr"] > r["base_rr"])
    worse = sum(1 for r in rows if r["ctx_rr"] < r["base_rr"])
    print(f"Gepaarte MRR-Bilanz: ctx {better} besser / {worse} schlechter / {n - better - worse} gleich")
    for r in rows:
        if r["ctx_rr"] != r["base_rr"]:
            print(f"  {r['id']}: base_rr={r['base_rr']:.2f} -> ctx_rr={r['ctx_rr']:.2f}")


async def run(gold_path: str, group_id: str, k: int, limit: int | None, chat_model: str) -> None:
    gold = [json.loads(l) for l in Path(gold_path).read_text().splitlines() if l.strip()]
    ollama = OllamaClient(settings.ollama_base_url)
    base_store = VectorStore(settings.qdrant_url, settings.qdrant_collection)
    try:
        records = await base_store.scroll_all(group_id)
        records.sort(key=lambda r: (r.payload["book_id"], r.payload["chunk_index"]))
        if limit:
            records = records[:limit]
        print(f"{len(records)} Chunks aus Gruppe '{group_id}'")

        ctx_path = Path(__file__).resolve().parent.parent / "results" / "contextual_contexts.jsonl"
        ctx_path.parent.mkdir(exist_ok=True)
        contexts = await generate_contexts(ollama, records, ctx_path, chat_model)
        await build_ctx_collection(ollama, records, contexts)
        if limit:
            print("\n(Smoke-Run mit --limit: Eval nur sinnvoll ohne Limit.)")
            return
        await evaluate(ollama, gold, group_id, k)
    finally:
        await base_store.aclose()
        await ollama.aclose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(Path(__file__).resolve().parent / "gold_dracula.jsonl"))
    ap.add_argument("--group", default="Horror")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="nur erste N Chunks (Smoke-Test)")
    ap.add_argument("--chat-model", default=settings.chat_model)
    args = ap.parse_args()
    asyncio.run(run(args.gold, args.group, args.k, args.limit, args.chat_model))
