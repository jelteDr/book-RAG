"""Modell-Liste: verfügbare Ollama-Chat-Modelle für das Frontend-Dropdown."""

from fastapi import APIRouter, HTTPException, Request

from app.config import settings

router = APIRouter(tags=["models"])


@router.get("/models")
async def list_models(request: Request) -> dict:
    try:
        models = await request.app.state.ollama.list_models()
    except Exception as exc:  # Ollama nicht erreichbar
        raise HTTPException(status_code=503, detail=f"Ollama nicht erreichbar: {exc}") from exc

    names = [m.get("name") for m in models if m.get("name")]
    return {"default": settings.chat_model, "available": names}
