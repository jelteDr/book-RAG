"""Gespeicherte Unterhaltungen: auflisten, laden, löschen."""

from fastapi import APIRouter, HTTPException
from sqlmodel import col, select

from app.db.models import ChatMessage, Conversation
from app.db.session import session_factory

router = APIRouter(tags=["conversations"])


@router.get("/conversations")
async def list_conversations() -> list[dict]:
    async with session_factory() as session:
        rows = (
            await session.exec(select(Conversation).order_by(col(Conversation.updated_at).desc()))
        ).all()
    return [
        {"id": c.id, "title": c.title, "group_id": c.group_id, "updated_at": c.updated_at.isoformat()}
        for c in rows
    ]


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: int) -> dict:
    async with session_factory() as session:
        conv = await session.get(Conversation, conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Unterhaltung nicht gefunden")
        msgs = (
            await session.exec(
                select(ChatMessage)
                .where(ChatMessage.conversation_id == conversation_id)
                .order_by(col(ChatMessage.id))
            )
        ).all()
    return {
        "id": conv.id,
        "title": conv.title,
        "group_id": conv.group_id,
        "messages": [
            {"role": m.role, "content": m.content, "model": m.model, "sources": m.sources or []}
            for m in msgs
        ],
    }


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int) -> dict:
    async with session_factory() as session:
        conv = await session.get(Conversation, conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Unterhaltung nicht gefunden")
        msgs = (
            await session.exec(
                select(ChatMessage).where(ChatMessage.conversation_id == conversation_id)
            )
        ).all()
        for m in msgs:
            await session.delete(m)
        await session.delete(conv)
        await session.commit()
    return {"deleted_conversation": conversation_id, "deleted_messages": len(msgs)}
