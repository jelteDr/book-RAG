"""Sammelgruppen: mehrere Gruppen unter einem Slug bündeln (ohne Re-Ingestion).

Eine Collection ist nur eine Verweis-Ebene — die Bücher/Vektoren bleiben in ihren
Gruppen. Der Chat kann den Collection-Slug wie einen Gruppen-Slug übergeben; der
RagService expandiert ihn zu einem group_id-IN-Filter in Qdrant.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import col, select

from app.db.models import Book, Collection, CollectionMember, Group
from app.db.session import session_factory
from app.routes.groups import unique_slug

router = APIRouter(tags=["collections"])


class CollectionCreate(BaseModel):
    # Slug ist optional (UI schickt keinen mehr) — er wird aus dem Namen abgeleitet.
    slug: str | None = None
    name: str
    member_slugs: list[str]


async def _collection_dict(session, collection: Collection) -> dict:
    members = (
        await session.exec(
            select(Group)
            .join(CollectionMember, col(CollectionMember.group_id) == col(Group.id))
            .where(CollectionMember.collection_id == collection.id)
        )
    ).all()
    group_ids = [g.id for g in members]
    n_books = 0
    if group_ids:
        n_books = len(
            (await session.exec(select(Book).where(col(Book.group_id).in_(group_ids)))).all()
        )
    return {
        "id": collection.id,
        "slug": collection.slug,
        "name": collection.name,
        "members": [{"slug": g.slug, "name": g.name} for g in members],
        "n_books": n_books,
    }


@router.post("/collections")
async def create_collection(body: CollectionCreate) -> dict:
    if not body.name.strip():
        raise HTTPException(status_code=422, detail="name ist Pflicht")
    if not body.member_slugs:
        raise HTTPException(status_code=422, detail="mindestens eine Mitglieds-Gruppe angeben")
    async with session_factory() as session:
        if body.slug:
            slug = body.slug.strip()
            # Ein Namensraum für Gruppen- und Collection-Slugs (der Chat sieht nur einen String).
            if (await session.exec(select(Group).where(Group.slug == slug))).first():
                raise HTTPException(status_code=409, detail=f"Slug gehört schon einer Gruppe: {slug}")
            if (await session.exec(select(Collection).where(Collection.slug == slug))).first():
                raise HTTPException(status_code=409, detail=f"Sammelgruppe existiert bereits: {slug}")
        else:
            slug = await unique_slug(session, body.name)

        members = (
            await session.exec(select(Group).where(col(Group.slug).in_(body.member_slugs)))
        ).all()
        missing = set(body.member_slugs) - {g.slug for g in members}
        if missing:
            raise HTTPException(status_code=404, detail=f"Unbekannte Gruppen: {sorted(missing)}")

        collection = Collection(slug=slug, name=body.name.strip())
        session.add(collection)
        await session.commit()
        await session.refresh(collection)
        for g in members:
            session.add(CollectionMember(collection_id=collection.id, group_id=g.id))
        await session.commit()
        return await _collection_dict(session, collection)


@router.get("/collections")
async def list_collections() -> list[dict]:
    async with session_factory() as session:
        collections = (await session.exec(select(Collection))).all()
        return [await _collection_dict(session, c) for c in collections]


@router.delete("/collections/{collection_id}")
async def delete_collection(collection_id: int) -> dict:
    """Löscht nur die Sammelgruppe — Gruppen, Bücher und Vektoren bleiben unberührt."""
    async with session_factory() as session:
        collection = await session.get(Collection, collection_id)
        if not collection:
            raise HTTPException(status_code=404, detail="Sammelgruppe nicht gefunden")
        memberships = (
            await session.exec(
                select(CollectionMember).where(CollectionMember.collection_id == collection_id)
            )
        ).all()
        for m in memberships:
            await session.delete(m)
        # Erst die Mitglieder rausflushen (FK auf collections), dann die Collection selbst —
        # ohne deklarierte Relationships kennt SQLAlchemy die Löschreihenfolge sonst nicht.
        await session.flush()
        await session.delete(collection)
        await session.commit()
        return {"deleted_collection": collection_id}
