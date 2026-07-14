"""FastAPI-Einstiegspunkt: verdrahtet Clients (Lifespan) und Routen."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.clients.ollama_client import OllamaClient
from app.clients.qdrant_client import VectorStore
from app.config import settings
from app.routes import chat, health, ingest, models
from app.services.rag_service import RagService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Gemeinsame Clients + Service für die Laufzeit der App anlegen.
    app.state.ollama = OllamaClient(settings.ollama_base_url)
    app.state.vectors = VectorStore(settings.qdrant_url, settings.qdrant_collection)
    app.state.rag = RagService(app.state.ollama, app.state.vectors, settings)
    yield
    await app.state.ollama.aclose()
    await app.state.vectors.aclose()


app = FastAPI(title="book-RAG", version="0.1.0", lifespan=lifespan)

app.include_router(health.router)
app.include_router(models.router)
app.include_router(chat.router)
app.include_router(ingest.router)
