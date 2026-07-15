"""Async-Datenbankzugang (PostgreSQL via SQLModel/SQLAlchemy)."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import settings

# `import app.db.models` stellt sicher, dass alle Tabellen registriert sind,
# bevor create_all läuft.
from app.db import models  # noqa: F401

engine = create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)

session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    """Legt fehlende Tabellen an (KISS; Migrationen via Alembic später)."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
