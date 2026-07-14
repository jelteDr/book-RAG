"""Chat-Endpunkt mit SSE-Streaming (TTFT messbar).

M0: reiner Passthrough an das LLM (noch OHNE Retrieval). Ab M1 wird hier zuerst
aus Qdrant retrievt und der Kontext mit nummerierten Quellen [n] in den Prompt
gebaut; die Zitier-Marker werden serverseitig auf Chunk-IDs gemappt.
"""

import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    model: str | None = None


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    model = req.model or settings.chat_model
    messages = [{"role": "user", "content": req.question}]

    async def generate() -> AsyncIterator[str]:
        t_start = time.perf_counter()
        ttft_ms: float | None = None
        usage: dict = {}

        async for line in request.app.state.ollama.chat_stream(messages, model):
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            # Bei include_usage kommt im finalen Chunk usage (choices dann leer).
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {}).get("content")
                if delta:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t_start) * 1000
                    yield _sse("token", {"content": delta})

        e2e_ms = (time.perf_counter() - t_start) * 1000
        yield _sse(
            "done",
            {
                "model": model,
                "ttft_ms": round(ttft_ms if ttft_ms is not None else e2e_ms, 1),
                "e2e_ms": round(e2e_ms, 1),
                "usage": usage,
            },
        )

    return StreamingResponse(generate(), media_type="text/event-stream")
