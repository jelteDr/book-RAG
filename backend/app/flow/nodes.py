"""Knoten + Routing des Corrective-RAG-Graphen (Exp 11).

Jeder Knoten ist ein dünner Wrapper um einen vorhandenen Baustein:
  retrieve  -> app.rag.retriever.retrieve (inkl. Overlap-Dedup)
  grade     -> EIN gebündelter LLM-Aufruf (JSON-Modus) bewertet alle neuen Passagen
  rewrite   -> LLM formuliert eine passage-nahe Alternativ-Suchanfrage
  generate  -> app.rag.prompt_builder.build_messages + gestreamte Generierung
  check     -> app.rag.citations + FaithfulnessService (NLI), entscheidet über Regen
  refuse    -> ehrliche Verweigerung, wenn auch nach Retries nichts Relevantes da ist
Die Schleifengrenzen (max_retries/max_regen) stehen im Zustand, damit das Eval
Arme mit und ohne Regen auf demselben Graphen fahren kann.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from langgraph.config import get_stream_writer

from app.clients.ollama_client import OllamaClient
from app.clients.qdrant_client import VectorStore
from app.flow.state import CragState
from app.rag.citations import extract_citations
from app.rag.prompt_builder import build_messages
from app.rag.retriever import dedup_overlaps, retrieve
from app.rag.streaming import stream_deltas

REFUSAL = "Dazu finde ich in den Quellen nichts."

# Passagen werden für den Grader gekürzt: er soll Themenbezug erkennen, nicht die Antwort
# formulieren — spart bei k=8 gut die Hälfte des Grader-Prompts.
GRADE_CHARS = 1200

_GRADE_SYSTEM = (
    "Du prüfst Suchergebnisse für ein Frage-Antwort-System über Bücher. "
    "Eine Passage ist RELEVANT, wenn sie Information enthält, die hilft, die Frage zu "
    "beantworten (auch teilweise). Nur thematisch ähnlich, aber ohne Antwortbezug = nicht "
    "relevant. Fragen und Passagen können verschiedene Sprachen haben. "
    'Antworte NUR mit JSON: {"relevant": [Nummern der relevanten Passagen]}.'
)

_REWRITE_SYSTEM = (
    "Die bisherigen Suchanfragen haben in einem Buch keine passenden Passagen gefunden. "
    "Formuliere EINE neue Suchanfrage, die so klingt wie die gesuchte Textstelle selbst "
    "(konkrete Namen, Orte, Handlungen, Schlüsselwörter statt Fragewörter). Sie muss sich "
    "von den bisherigen Anfragen unterscheiden. Gib NUR die Suchanfrage aus."
)

STRICT_EXTRA = (
    "\n7. STRENG: Ein Prüfer hat Aussagen deines letzten Entwurfs als nicht belegt markiert. "
    "Schreibe nur, was eine Quelle ausdrücklich sagt — lieber kürzer als unbelegt."
)


def parse_grade(raw: str, n: int) -> list[int] | None:
    """Liest `{"relevant": [...]}`; None bei unbrauchbarer Antwort (-> fail-open)."""
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    rel = data.get("relevant") if isinstance(data, dict) else None
    if not isinstance(rel, list):
        return None
    out: list[int] = []
    for x in rel:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= n and i not in out:
            out.append(i)
    return out


def _trace(state: CragState, node: str, t0: float, **detail: Any) -> list[dict]:
    return [*state.get("trace", []), {"node": node, "ms": round((time.perf_counter() - t0) * 1000), **detail}]


class CragNodes:
    def __init__(
        self,
        ollama: OllamaClient,
        vectors: VectorStore,
        *,
        embed_model: str,
        min_relevant: int,
        unsupported_max: float,
        grader_model: str | None = None,
        faithfulness=None,
        temperature: float = 0.7,
    ) -> None:
        self._ollama = ollama
        self._vectors = vectors
        self._embed_model = embed_model
        self._min_relevant = min_relevant
        self._unsupported_max = unsupported_max
        self._grader_model = grader_model
        self._faithfulness = faithfulness
        self._temperature = temperature

    # --- Knoten -----------------------------------------------------------------

    async def retrieve(self, state: CragState) -> dict:
        t0 = time.perf_counter()
        queries = state.get("queries") or [state["retrieval_query"]]
        points = await retrieve(
            queries[-1], ollama=self._ollama, vectors=self._vectors,
            embed_model=self._embed_model, top_k=state["top_k"], group_ids=state.get("group_ids"),
        )
        # Nur Passagen, die noch nicht bewertet wurden und keine schon relevante überlappen.
        seen = set(state.get("graded_ids", []))
        relevant = state.get("relevant", [])
        fresh = [p for p in points if p.id not in seen]
        fresh = dedup_overlaps([*relevant, *fresh], len(relevant) + len(fresh))[len(relevant):]
        return {
            "queries": queries,
            "attempts": state.get("attempts", 0) + 1,
            "candidates": fresh,
            "trace": _trace(state, "retrieve", t0, query=queries[-1], n_new=len(fresh)),
        }

    async def grade(self, state: CragState) -> dict:
        t0 = time.perf_counter()
        cands = state.get("candidates", [])
        keep: list[int] | None = []
        if cands:
            passages = "\n\n".join(
                f"[{i}] {((p.payload or {}).get('text') or '')[:GRADE_CHARS]}"
                for i, p in enumerate(cands, start=1)
            )
            messages = [
                {"role": "system", "content": _GRADE_SYSTEM},
                {"role": "user", "content": f"Frage: {state['question']}\n\nPassagen:\n{passages}"},
            ]
            try:
                raw = await self._ollama.complete(
                    messages, self._grader_model or state["model"], temperature=0.0,
                    max_tokens=80, response_format={"type": "json_object"},
                )
                keep = parse_grade(raw, len(cands))
            except Exception:
                keep = None
        fail_open = keep is None
        if fail_open:  # Grader kaputt -> lieber alles behalten als die Antwort verlieren
            keep = list(range(1, len(cands) + 1))
        attempt = state.get("attempts", 1)
        graded = [
            *state.get("graded", []),
            *(
                {
                    "id": p.id, "attempt": attempt, "relevant": (i in keep),
                    "book_id": (p.payload or {}).get("book_id"),
                    "char_start": (p.payload or {}).get("char_start"),
                    "char_end": (p.payload or {}).get("char_end"),
                }
                for i, p in enumerate(cands, start=1)
            ),
        ]
        new_rel = [p for i, p in enumerate(cands, start=1) if i in keep]
        return {
            "graded_ids": [*state.get("graded_ids", []), *(p.id for p in cands)],
            "relevant": [*state.get("relevant", []), *new_rel],
            "graded": graded,
            "candidates": [],
            "trace": _trace(state, "grade", t0, n=len(cands), relevant=len(new_rel), fail_open=fail_open),
        }

    async def rewrite(self, state: CragState) -> dict:
        t0 = time.perf_counter()
        queries = state.get("queries") or [state["retrieval_query"]]
        tried = "\n".join(f"- {q}" for q in queries)
        messages = [
            {"role": "system", "content": _REWRITE_SYSTEM},
            {"role": "user", "content": f"Frage: {state['question']}\n\nBisherige Suchanfragen:\n{tried}\n\nNeue Suchanfrage:"},
        ]
        try:
            new = (await self._ollama.complete(messages, state["model"], temperature=0.0, max_tokens=60)).strip()
        except Exception:
            new = ""
        new = new.strip('"').splitlines()[0].strip() if new else state["question"]
        return {"queries": [*queries, new], "trace": _trace(state, "rewrite", t0, query=new)}

    async def generate(self, state: CragState) -> dict:
        t0 = time.perf_counter()
        writer = get_stream_writer()
        regen = state.get("regen", 0)
        retry = bool(state.get("answer"))  # zweiter Durchlauf = Regen
        if retry:
            regen += 1
            writer({"event": "reset", "reason": "unsupported"})
        # Relevante Passagen in Score-Reihenfolge, max. top_k (Prompt-Budget wie Baseline).
        context = sorted(state.get("relevant", []), key=lambda p: p.score, reverse=True)[: state["top_k"]]
        messages = build_messages(
            state["question"], context, state.get("history"),
            system_extra=STRICT_EXTRA if retry else None,
        )
        parts: list[str] = []
        usage: dict = {}
        async for delta, usage_chunk in stream_deltas(
            self._ollama, messages, state["model"], temperature=self._temperature
        ):
            if usage_chunk:
                usage = usage_chunk
            if delta:
                parts.append(delta)
                writer({"event": "token", "content": delta})
        answer = "".join(parts)
        return {
            "context": context, "answer": answer, "usage": usage, "regen": regen,
            "sources": extract_citations(answer, context),
            "trace": _trace(state, "generate", t0, n_context=len(context), strict=retry),
        }

    async def check(self, state: CragState) -> dict:
        t0 = time.perf_counter()
        sources = state.get("sources", [])
        if self._faithfulness is not None and self._faithfulness.active and sources:
            sources = await asyncio.to_thread(self._faithfulness.check, state["answer"], sources)
        unsupported = sum(1 for s in sources if s.get("supported") is False)
        return {
            "sources": sources,
            "trace": _trace(state, "check", t0, n_sources=len(sources), unsupported=unsupported),
        }

    async def refuse(self, state: CragState) -> dict:
        get_stream_writer()({"event": "token", "content": REFUSAL})
        return {
            "answer": REFUSAL, "context": [], "sources": [], "refused": True,
            "trace": _trace(state, "refuse", time.perf_counter()),
        }

    # --- Routing (bedingte Kanten) ---------------------------------------------------

    def after_grade(self, state: CragState) -> str:
        n_rel = len(state.get("relevant", []))
        if n_rel >= self._min_relevant:
            return "generate"
        if state.get("attempts", 0) <= state.get("max_retries", 1):
            return "rewrite"
        return "generate" if n_rel else "refuse"

    def after_check(self, state: CragState) -> str:
        sources = state.get("sources", [])
        if not sources or state.get("regen", 0) >= state.get("max_regen", 0):
            return "end"
        share = sum(1 for s in sources if s.get("supported") is False) / len(sources)
        return "generate" if share > self._unsupported_max else "end"
