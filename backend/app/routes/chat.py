"""Chat-Route (dünn): delegiert an den RagService und formatiert dessen Events als SSE."""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    model: str | None = None
    group_id: str | None = None
    top_k: int | None = None
    conversation_id: int | None = None
    mode: str | None = None  # local | global | auto — überschreibt GRAPH_RAG_MODE (Graph-RAG)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    service = request.app.state.rag

    async def stream() -> AsyncIterator[str]:
        async for event, payload in service.answer(
            req.question, model=req.model, group_id=req.group_id, top_k=req.top_k,
            conversation_id=req.conversation_id, mode=req.mode,
        ):
            yield _sse(event, payload)

    return StreamingResponse(stream(), media_type="text/event-stream")
