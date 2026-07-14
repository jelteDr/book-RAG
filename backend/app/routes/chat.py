"""Chat-Endpunkt mit Retrieval + SSE-Streaming und serverseitiger Zitat-Zuordnung.

Ablauf: Frage einbetten -> Qdrant-Retrieval (optional nach Gruppe gefiltert) ->
Prompt mit nummerierten Quellen [1..k] -> LLM streamt Prosa mit inline [n] ->
am Ende werden die [n]-Marker auf die Quell-Chunks gemappt und als `done`-Event
mitgeschickt (inkl. TTFT/E2E/Tokens fürs Monitoring).
"""

import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.rag.citations import extract_citations, sources_overview
from app.rag.prompt_builder import build_messages
from app.rag.retriever import retrieve

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    model: str | None = None
    group_id: str | None = None
    top_k: int | None = None


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    model = req.model or settings.chat_model
    top_k = req.top_k or settings.top_k
    ollama = request.app.state.ollama
    vectors = request.app.state.vectors

    async def generate() -> AsyncIterator[str]:
        t_start = time.perf_counter()

        # 1) Retrieval.
        try:
            points = await retrieve(
                req.question,
                ollama=ollama,
                vectors=vectors,
                embed_model=settings.embed_model,
                top_k=top_k,
                group_id=req.group_id,
            )
        except Exception as exc:  # Ollama/Qdrant nicht erreichbar
            yield _sse("error", {"message": f"Retrieval fehlgeschlagen: {exc}"})
            return

        if not points:
            yield _sse("token", {"content": "Dazu finde ich in den Quellen nichts."})
            yield _sse("done", {"model": model, "sources": [], "retrieved": []})
            return

        # 2) Prompt bauen + streamen.
        messages = build_messages(req.question, points)
        answer_parts: list[str] = []
        ttft_ms: float | None = None
        usage: dict = {}

        try:
            async for line in ollama.chat_stream(messages, model):
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {}).get("content")
                    if delta:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t_start) * 1000
                        answer_parts.append(delta)
                        yield _sse("token", {"content": delta})
        except Exception as exc:
            yield _sse("error", {"message": f"Generierung fehlgeschlagen: {exc}"})
            return

        # 3) Zitate serverseitig zuordnen + Monitoring-Kennzahlen.
        answer = "".join(answer_parts)
        e2e_ms = (time.perf_counter() - t_start) * 1000
        completion_tokens = usage.get("completion_tokens") or 0
        decode_ms = max(e2e_ms - (ttft_ms or 0), 1e-6)
        tps = round(completion_tokens / (decode_ms / 1000), 1) if completion_tokens else 0.0

        yield _sse(
            "done",
            {
                "model": model,
                "ttft_ms": round(ttft_ms if ttft_ms is not None else e2e_ms, 1),
                "e2e_ms": round(e2e_ms, 1),
                "tps": tps,
                "usage": usage,
                "sources": extract_citations(answer, points),
                "retrieved": sources_overview(points),
            },
        )

    return StreamingResponse(generate(), media_type="text/event-stream")
