"""Health-Check: prüft Erreichbarkeit von Ollama und Qdrant."""

from fastapi import APIRouter, Request

router = APIRouter(tags=["system"])


@router.get("/health")
async def health(request: Request) -> dict:
    ollama_ok = True
    try:
        await request.app.state.ollama.list_models()
    except Exception:
        ollama_ok = False

    qdrant_ok = await request.app.state.vectors.health()

    status = "ok" if (ollama_ok and qdrant_ok) else "degraded"
    return {
        "status": status,
        "ollama": ollama_ok,
        "qdrant": qdrant_ok,
        "reranker": request.app.state.reranker.active,
    }
