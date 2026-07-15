"""FastAPI-Einstiegspunkt: verdrahtet DB, Clients (Lifespan) und Routen."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.clients.ollama_client import OllamaClient
from app.clients.qdrant_client import VectorStore
from app.config import settings
from app.db.session import init_db, session_factory
from app.routes import chat, conversations, groups, health, ingest, metrics, models
from app.services.faithfulness_service import FaithfulnessService
from app.services.model_registry import sync_models
from app.services.rag_service import RagService
from app.services.reranker_service import RerankerService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Clients + Service für die Laufzeit der App anlegen.
    app.state.ollama = OllamaClient(settings.ollama_base_url)
    app.state.vectors = VectorStore(settings.qdrant_url, settings.qdrant_collection)
    app.state.reranker = RerankerService(settings.reranker_model, settings.reranker_enabled)
    app.state.faithfulness = FaithfulnessService(
        settings.faithfulness_model, settings.faithfulness_check_enabled, settings.faithfulness_threshold
    )
    app.state.rag = RagService(
        app.state.ollama, app.state.vectors, settings, session_factory,
        app.state.reranker, app.state.faithfulness,
    )

    # DB-Tabellen sicherstellen + Modell-Registry befüllen (best-effort).
    await init_db()
    try:
        await sync_models(session_factory, app.state.ollama)
    except Exception:
        pass  # Ollama evtl. nicht erreichbar — App startet trotzdem

    yield
    await app.state.ollama.aclose()
    await app.state.vectors.aclose()


app = FastAPI(title="book-RAG", version="0.2.0", lifespan=lifespan)

app.include_router(health.router)
app.include_router(models.router)
app.include_router(metrics.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(groups.router)
app.include_router(ingest.router)
