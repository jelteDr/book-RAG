"""SQLModel-Tabellen: Modell-Metadaten und Anfrage-Log (Metriken)."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import JSON, BigInteger, Column
from sqlmodel import Field, SQLModel


class ModelInfo(SQLModel, table=True):
    """Metadaten pro (Ollama-)Modell. Auto-Felder aus /api/show, cutoff_date kuratiert."""

    __tablename__ = "model_info"

    name: str = Field(primary_key=True)
    family: str | None = None
    parameter_size: str | None = None      # z. B. "7.6B" (aus Ollama)
    # BigInteger: Parameterzahlen (Milliarden) sprengen INT32.
    parameter_count: int | None = Field(default=None, sa_type=BigInteger)
    quantization: str | None = None
    context_length: int | None = None
    cutoff_date: date | None = None        # NICHT von Ollama geliefert -> kuratiert (PATCH)
    source: str | None = None              # z. B. "ollama" / "ollama+kuratiert"
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Group(SQLModel, table=True):
    """Eine Gruppe (Genre/Franchise). `slug` ist der group_id-Wert im Qdrant-Filter."""

    __tablename__ = "groups"

    id: int | None = Field(default=None, primary_key=True)
    slug: str = Field(unique=True, index=True)   # z. B. "horror-classics" (Qdrant-group_id)
    name: str
    kind: str | None = None                       # "genre" | "franchise" | ...
    description: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Book(SQLModel, table=True):
    """Ein ingestetes Buch innerhalb einer Gruppe. `book_key` = book_id in der Qdrant-Payload."""

    __tablename__ = "books"

    id: int | None = Field(default=None, primary_key=True)
    group_id: int = Field(foreign_key="groups.id", index=True)
    book_key: str = Field(unique=True, index=True)
    title: str | None = None
    author: str | None = None
    language: str | None = None
    n_chunks: int = 0
    status: str = "committed"                      # pending | committed | failed
    file_hash: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Collection(SQLModel, table=True):
    """Eine Sammelgruppe: fasst mehrere Gruppen zusammen (z. B. Genre über Franchises).

    Kein Kopieren von Daten — beim Retrieval expandiert der Collection-Slug zu einem
    group_id-IN-Filter über die Mitglieds-Slugs. Slugs von Gruppen und Collections
    teilen sich einen Namensraum (der Chat übergibt nur einen String).
    """

    __tablename__ = "collections"

    id: int | None = Field(default=None, primary_key=True)
    slug: str = Field(unique=True, index=True)
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CollectionMember(SQLModel, table=True):
    """Mitgliedschaft einer Gruppe in einer Sammelgruppe (m:n)."""

    __tablename__ = "collection_members"

    id: int | None = Field(default=None, primary_key=True)
    collection_id: int = Field(foreign_key="collections.id", index=True)
    group_id: int = Field(foreign_key="groups.id", index=True)


class Conversation(SQLModel, table=True):
    """Eine gespeicherte Unterhaltung."""

    __tablename__ = "conversations"

    id: int | None = Field(default=None, primary_key=True)
    title: str
    group_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ChatMessage(SQLModel, table=True):
    """Eine Nachricht (user/assistant) innerhalb einer Unterhaltung."""

    __tablename__ = "chat_messages"

    id: int | None = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversations.id", index=True)
    role: str  # "user" | "assistant"
    content: str
    model: str | None = None
    # Vollständige Quell-Liste inkl. Chunk-Text (Chips bleiben nach dem Laden lesbar;
    # ältere Zeilen können noch die frühere "leichte" Liste ohne Text enthalten).
    sources: list | None = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)


class QueryLog(SQLModel, table=True):
    """Eine Zeile pro Chat-Anfrage — die Metrik-Quelle pro Modell (kein Antworttext)."""

    __tablename__ = "query_log"

    id: int | None = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=datetime.utcnow, index=True)
    model: str = Field(index=True)
    group_id: str | None = None
    question: str
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    tps: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    n_sources: int | None = None
