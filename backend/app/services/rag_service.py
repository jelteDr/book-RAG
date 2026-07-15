"""RagService — kapselt den kompletten Frage→Antwort-Ablauf.

Die Route ruft nur noch `answer(...)` auf und formatiert die gelieferten Events als
SSE. Retrieval, Prompt-Bau, Streaming und Zitat-Zuordnung bleiben in eigenen,
testbaren Bausteinen (retriever/prompt_builder/citations); der Service verdrahtet sie.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from datetime import datetime

from app.clients.ollama_client import OllamaClient
from app.clients.qdrant_client import VectorStore
from app.config import Settings
from sqlmodel import col, select

from app.db.models import ChatMessage, Conversation, QueryLog
from app.rag.citations import extract_citations, sources_overview
from app.rag.prompt_builder import build_messages
from app.rag.retriever import retrieve

# Ein Event ist (name, payload) — die Route macht daraus SSE.
Event = tuple[str, dict]


class RagService:
    def __init__(
        self,
        ollama: OllamaClient,
        vectors: VectorStore,
        settings: Settings,
        session_factory=None,
        reranker=None,
        faithfulness=None,
    ) -> None:
        self._ollama = ollama
        self._vectors = vectors
        self._settings = settings
        self._session_factory = session_factory
        self._reranker = reranker
        self._faithfulness = faithfulness

    async def answer(
        self, question: str, *, model: str | None = None, group_id: str | None = None,
        top_k: int | None = None, conversation_id: int | None = None,
    ) -> AsyncIterator[Event]:
        """Liefert nacheinander ('token'|'done'|'error', payload)-Events."""
        model = model or self._settings.chat_model
        top_k = top_k or self._settings.top_k
        t_start = time.perf_counter()

        # Unterhaltung sicherstellen + Nutzer-Nachricht speichern; prior History holen.
        conversation_id, history = await self._ensure_conversation(
            conversation_id, question, group_id
        )

        # History-aware: Folgefrage zu eigenständiger Suchanfrage umschreiben (Retrieval),
        # der Verlauf selbst geht begrenzt in die Generierung (build_messages).
        retrieval_query = question
        if history:
            retrieval_query = await self._condense(question, history, model)

        rerank = self._reranker is not None and self._reranker.active
        fetch_k = self._settings.rerank_candidates if rerank else top_k
        try:
            points = await retrieve(
                retrieval_query, ollama=self._ollama, vectors=self._vectors,
                embed_model=self._settings.embed_model, top_k=fetch_k, group_id=group_id,
            )
        except Exception as exc:  # Ollama/Qdrant nicht erreichbar
            yield "error", {"message": f"Retrieval fehlgeschlagen: {exc}"}
            return

        if rerank and points:
            # Blockierendes torch im Thread, damit der Event-Loop frei bleibt.
            points = await asyncio.to_thread(self._reranker.rerank, question, points, top_k)
        else:
            points = points[:top_k]

        if not points:
            msg = "Dazu finde ich in den Quellen nichts."
            done = {"model": model, "sources": [], "retrieved": [], "conversation_id": conversation_id}
            await self._log_query(question, model, group_id, done)
            await self._save_assistant(conversation_id, msg, model, [])
            yield "token", {"content": msg}
            yield "done", done
            return

        answer_parts: list[str] = []
        ttft_ms: float | None = None
        usage: dict = {}
        try:
            async for delta, usage_chunk in self._stream_deltas(
                build_messages(question, points, history), model
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
        done = {
            "model": model,
            "ttft_ms": round(ttft_ms if ttft_ms is not None else e2e_ms, 1),
            "e2e_ms": round(e2e_ms, 1),
            "tps": round(completion_tokens / (decode_ms / 1000), 1) if completion_tokens else 0.0,
            "usage": usage,
            "sources": extract_citations(answer, points),
            "retrieved": sources_overview(points),
            "conversation_id": conversation_id,
        }
        # Optional: ungestützte Zitate via NLI markieren (blockierend -> Thread).
        if self._faithfulness is not None and self._faithfulness.active and done["sources"]:
            done["sources"] = await asyncio.to_thread(
                self._faithfulness.check, answer, done["sources"]
            )
        await self._log_query(question, model, group_id, done)
        await self._save_assistant(conversation_id, answer, model, done["sources"])
        yield "done", done

    async def _ensure_conversation(
        self, conversation_id: int | None, question: str, group_id: str | None
    ) -> tuple[int | None, list[dict]]:
        """Legt bei Bedarf eine Unterhaltung an, speichert die Nutzer-Nachricht und
        liefert den bisherigen Verlauf (vor dieser Frage) zurück."""
        if self._session_factory is None:
            return conversation_id, []
        history: list[dict] = []
        try:
            async with self._session_factory() as session:
                if conversation_id is None:
                    conv = Conversation(title=question[:80], group_id=group_id)
                    session.add(conv)
                    await session.commit()
                    await session.refresh(conv)
                    conversation_id = conv.id
                else:
                    prior = (
                        await session.exec(
                            select(ChatMessage)
                            .where(ChatMessage.conversation_id == conversation_id)
                            .order_by(col(ChatMessage.id))
                        )
                    ).all()
                    history = [{"role": m.role, "content": m.content} for m in prior]
                    conv = await session.get(Conversation, conversation_id)
                    if conv:
                        conv.updated_at = datetime.utcnow()
                        session.add(conv)
                session.add(ChatMessage(conversation_id=conversation_id, role="user", content=question))
                await session.commit()
            return conversation_id, history
        except Exception:
            return conversation_id, history

    async def _condense(self, question: str, history: list[dict], model: str) -> str:
        """Schreibt eine Folgefrage anhand des Verlaufs zu einer eigenständigen Suchanfrage um."""
        hist_txt = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
        messages = [
            {
                "role": "system",
                "content": (
                    "Formuliere die letzte Nutzerfrage zu einer eigenständigen, kontextfreien "
                    "Suchanfrage um (nutze den Verlauf, um Bezüge wie 'er'/'das' aufzulösen). "
                    "Gib NUR die umformulierte Frage aus, ohne Erklärung."
                ),
            },
            {
                "role": "user",
                "content": f"Verlauf:\n{hist_txt}\n\nLetzte Frage: {question}\n\nEigenständige Suchanfrage:",
            },
        ]
        try:
            out = await self._ollama.complete(messages, model, temperature=0.0, max_tokens=80)
            return out.strip() or question
        except Exception:
            return question

    async def _save_assistant(
        self, conversation_id: int | None, answer: str, model: str, sources: list[dict]
    ) -> None:
        """Speichert die Assistenten-Antwort (leichte Quell-Liste, kein Chunk-Text)."""
        if self._session_factory is None or conversation_id is None:
            return
        light = [
            {
                "marker": s.get("marker"), "chapter": s.get("chapter"),
                "score": s.get("score"), "book_title": s.get("book_title"),
                "supported": s.get("supported"),
            }
            for s in sources
        ]
        try:
            async with self._session_factory() as session:
                session.add(
                    ChatMessage(
                        conversation_id=conversation_id, role="assistant",
                        content=answer, model=model, sources=light,
                    )
                )
                await session.commit()
        except Exception:
            pass

    async def _log_query(self, question: str, model: str, group_id: str | None, done: dict) -> None:
        """Schreibt eine query_log-Zeile (best-effort — darf die Antwort nie brechen)."""
        if self._session_factory is None:
            return
        try:
            usage = done.get("usage") or {}
            async with self._session_factory() as session:
                session.add(
                    QueryLog(
                        model=model,
                        group_id=group_id,
                        question=question,
                        ttft_ms=done.get("ttft_ms"),
                        e2e_ms=done.get("e2e_ms"),
                        tps=done.get("tps"),
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                        n_sources=len(done.get("sources") or []),
                    )
                )
                await session.commit()
        except Exception:
            pass

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
