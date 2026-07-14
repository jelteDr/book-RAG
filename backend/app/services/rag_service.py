"""RagService — kapselt den kompletten Frage→Antwort-Ablauf.

Die Route ruft nur noch `answer(...)` auf und formatiert die gelieferten Events als
SSE. Retrieval, Prompt-Bau, Streaming und Zitat-Zuordnung bleiben in eigenen,
testbaren Bausteinen (retriever/prompt_builder/citations); der Service verdrahtet sie.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

from app.clients.ollama_client import OllamaClient
from app.clients.qdrant_client import VectorStore
from app.config import Settings
from app.rag.citations import extract_citations, sources_overview
from app.rag.prompt_builder import build_messages
from app.rag.retriever import retrieve

# Ein Event ist (name, payload) — die Route macht daraus SSE.
Event = tuple[str, dict]


class RagService:
    def __init__(self, ollama: OllamaClient, vectors: VectorStore, settings: Settings) -> None:
        self._ollama = ollama
        self._vectors = vectors
        self._settings = settings

    async def answer(
        self, question: str, *, model: str | None = None, group_id: str | None = None,
        top_k: int | None = None,
    ) -> AsyncIterator[Event]:
        """Liefert nacheinander ('token'|'done'|'error', payload)-Events."""
        model = model or self._settings.chat_model
        top_k = top_k or self._settings.top_k
        t_start = time.perf_counter()

        try:
            points = await retrieve(
                question, ollama=self._ollama, vectors=self._vectors,
                embed_model=self._settings.embed_model, top_k=top_k, group_id=group_id,
            )
        except Exception as exc:  # Ollama/Qdrant nicht erreichbar
            yield "error", {"message": f"Retrieval fehlgeschlagen: {exc}"}
            return

        if not points:
            yield "token", {"content": "Dazu finde ich in den Quellen nichts."}
            yield "done", {"model": model, "sources": [], "retrieved": []}
            return

        answer_parts: list[str] = []
        ttft_ms: float | None = None
        usage: dict = {}
        try:
            async for delta, usage_chunk in self._stream_deltas(
                build_messages(question, points), model
            ):
                if usage_chunk:
                    usage = usage_chunk
                if delta:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t_start) * 1000
                    answer_parts.append(delta)
                    yield "token", {"content": delta}
        except Exception as exc:
            yield "error", {"message": f"Generierung fehlgeschlagen: {exc}"}
            return

        answer = "".join(answer_parts)
        e2e_ms = (time.perf_counter() - t_start) * 1000
        completion_tokens = usage.get("completion_tokens") or 0
        decode_ms = max(e2e_ms - (ttft_ms or 0), 1e-6)
        yield "done", {
            "model": model,
            "ttft_ms": round(ttft_ms if ttft_ms is not None else e2e_ms, 1),
            "e2e_ms": round(e2e_ms, 1),
            "tps": round(completion_tokens / (decode_ms / 1000), 1) if completion_tokens else 0.0,
            "usage": usage,
            "sources": extract_citations(answer, points),
            "retrieved": sources_overview(points),
        }

    async def _stream_deltas(
        self, messages: list[dict], model: str
    ) -> AsyncIterator[tuple[str | None, dict | None]]:
        """Parst die rohen SSE-Zeilen von Ollama zu (content-delta, usage)-Paaren."""
        async for line in self._ollama.chat_stream(messages, model):
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                return
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage")
            for choice in chunk.get("choices", []):
                yield choice.get("delta", {}).get("content"), usage
            if not chunk.get("choices") and usage:
                yield None, usage
