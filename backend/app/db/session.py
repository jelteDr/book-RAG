"""Async-Datenbankzugang (PostgreSQL via SQLModel/SQLAlchemy)."""

from __future__ import annotations

from sqlalchemy import text
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
        # Additive Spalten für bestehende Datenbanken (create_all ergänzt keine Spalten).
        for stmt in (
            "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS parent_id INTEGER",
            "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS selected BOOLEAN NOT NULL DEFAULT TRUE",
            "CREATE INDEX IF NOT EXISTS ix_chat_messages_parent_id ON chat_messages (parent_id)",
        ):
            await conn.execute(text(stmt))
