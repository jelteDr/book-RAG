"""Gruppen + Upload: Gruppen anlegen/löschen und Texte zu einer Gruppe ingesten."""

import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlmodel import select

from app.config import settings
from app.db.models import Book, Group
from app.db.session import session_factory
from app.ingestion.pipeline import ingest_bytes

router = APIRouter(tags=["groups"])


class GroupCreate(BaseModel):
    slug: str
    name: str
    kind: str | None = None
    description: str | None = None


def _group_dict(g: Group, n_books: int | None = None) -> dict:
    d = {"id": g.id, "slug": g.slug, "name": g.name, "kind": g.kind, "description": g.description}
    if n_books is not None:
        d["n_books"] = n_books
    return d


def _book_dict(b: Book) -> dict:
    return {
        "id": b.id, "book_key": b.book_key, "title": b.title, "author": b.author,
        "language": b.language, "n_chunks": b.n_chunks, "status": b.status,
    }


@router.post("/groups")
async def create_group(body: GroupCreate) -> dict:
    async with session_factory() as session:
        if (await session.exec(select(Group).where(Group.slug == body.slug))).first():
            raise HTTPException(status_code=409, detail=f"Gruppe existiert bereits: {body.slug}")
        group = Group(**body.model_dump())
        session.add(group)
        await session.commit()
        await session.refresh(group)
        return _group_dict(group, n_books=0)


@router.get("/groups")
async def list_groups() -> list[dict]:
    async with session_factory() as session:
        groups = (await session.exec(select(Group))).all()
        out = []
        for g in groups:
            books = (await session.exec(select(Book).where(Book.group_id == g.id))).all()
            out.append(_group_dict(g, n_books=len(books)))
        return out


@router.get("/groups/{group_id}/books")
async def group_books(group_id: int) -> list[dict]:
    async with session_factory() as session:
        books = (await session.exec(select(Book).where(Book.group_id == group_id))).all()
        return [_book_dict(b) for b in books]


@router.post("/groups/{group_id}/upload")
async def upload_to_group(
    group_id: int,
    request: Request,
    file: UploadFile = File(...),
    title: str | None = Form(None),
    author: str | None = Form(None),
    commit: bool = Form(False),
) -> dict:
    """Dry-Run (Cleaning-Report) oder Commit (ingest + Buch persistieren)."""
    async with session_factory() as session:
        group = await session.get(Group, group_id)
        if not group:
            raise HTTPException(status_code=404, detail="Gruppe nicht gefunden")

        raw = await file.read()
        file_hash = hashlib.sha256(raw).hexdigest()
        book_key = uuid.uuid4().hex

        result = await ingest_bytes(
            raw,
            book_id=book_key,
            group_id=group.slug,
            ollama=request.app.state.ollama,
            vectors=request.app.state.vectors,
            embed_model=settings.embed_model,
            dry_run=not commit,
            title=title,
            author=author,
        )

        # Fallback: kein Gutenberg-Titel + kein manueller Titel -> Dateiname (statt Buch-Key).
        resolved_title = result.title or (Path(file.filename or "").stem or None)

        if not commit:
            return {"committed": False, "title": resolved_title, "author": result.author,
                    "report": result.report}

        book = Book(
            group_id=group_id, book_key=book_key, title=resolved_title, author=result.author,
            language=result.language, n_chunks=result.n_chunks, status="committed",
            file_hash=file_hash,
        )
        session.add(book)
        await session.commit()
        await session.refresh(book)
        return {"committed": True, "book": _book_dict(book), "report": result.report}


@router.delete("/groups/{group_id}")
async def delete_group(group_id: int, request: Request) -> dict:
    """Löscht Gruppe + ihre Bücher (Qdrant-Vektoren + DB atomar)."""
    async with session_factory() as session:
        group = await session.get(Group, group_id)
        if not group:
            raise HTTPException(status_code=404, detail="Gruppe nicht gefunden")
        books = (await session.exec(select(Book).where(Book.group_id == group_id))).all()
        for book in books:
            await request.app.state.vectors.delete_by_book(book.book_key)
            await session.delete(book)
        await session.delete(group)
        await session.commit()
        return {"deleted_group": group_id, "deleted_books": len(books)}


@router.delete("/books/{book_id}")
async def delete_book(book_id: int, request: Request) -> dict:
    async with session_factory() as session:
        book = await session.get(Book, book_id)
        if not book:
            raise HTTPException(status_code=404, detail="Buch nicht gefunden")
        await request.app.state.vectors.delete_by_book(book.book_key)
        await session.delete(book)
        await session.commit()
        return {"deleted_book": book_id}
