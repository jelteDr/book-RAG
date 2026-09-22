"""Health-Check: prüft Erreichbarkeit von Ollama und Qdrant."""

from fastapi import APIRouter, Request

from app.config import settings

router = APIRouter(tags=["system"])


@router.get("/health")
async def health(request: Request) -> dict:
    ollama_ok = True
    context_length = None
    try:
        await request.app.state.ollama.list_models()
        context_length = await request.app.state.ollama.context_length(settings.chat_model)
    except Exception:
        ollama_ok = False

    qdrant_ok = await request.app.state.vectors.health()

    status = "ok" if (ollama_ok and qdrant_ok) else "degraded"
    return {
        "status": status,
        "ollama": ollama_ok,
        # Kontextfenster des geladenen Chat-Modells (None = nicht geladen); < 8192 heißt:
        # RAG-Prompts werden still abgeschnitten -> `make ollama-ctx` + Ollama neu starten.
        "ollama_context_length": context_length,
        "qdrant": qdrant_ok,
        "reranker": request.app.state.reranker.active,
        "faithfulness": request.app.state.faithfulness.active,
        "graph": request.app.state.graph.status(),
    }
