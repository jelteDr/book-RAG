"""Modell-Routen: Liste fürs Dropdown, angereicherte Metadaten, Sync & Kuratierung."""

from datetime import date

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlmodel import select

from app.config import settings
from app.db.models import ModelInfo
from app.db.session import session_factory
from app.services.model_registry import sync_models

router = APIRouter(tags=["models"])


class ModelPatch(BaseModel):
    cutoff_date: date | None = None
    parameter_count: int | None = None


@router.get("/models")
async def list_models(request: Request) -> dict:
    """Schlanke Liste fürs Frontend-Dropdown — nur chatfähige Modelle.

    Embedding-Modelle (z. B. bge-m3) liefern am Chat-Endpoint 400 Bad Request.
    Ollama meldet die Fähigkeiten pro Modell via /api/show (`capabilities`);
    ohne `completion` fliegt das Modell aus der Auswahl. Meldet eine ältere
    Ollama-Version keine capabilities, bleibt das Modell drin (kein False-Drop).
    """
    ollama = request.app.state.ollama
    try:
        tags = await ollama.list_models()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ollama nicht erreichbar: {exc}") from exc

    names = []
    for tag in tags:
        name = tag.get("name")
        if not name:
            continue
        try:
            capabilities = (await ollama.show(name)).get("capabilities")
        except Exception:
            capabilities = None  # im Zweifel nicht filtern
        if capabilities is None or "completion" in capabilities:
            names.append(name)
    return {"default": settings.chat_model, "available": names}


@router.get("/models/info")
async def models_info() -> list[dict]:
    """Angereicherte Modell-Metadaten aus der Registry (Parameter, Cutoff, Quantisierung …)."""
    async with session_factory() as session:
        rows = (await session.exec(select(ModelInfo))).all()
    return [
        {
            "name": r.name,
            "family": r.family,
            "parameter_size": r.parameter_size,
            "parameter_count": r.parameter_count,
            "quantization": r.quantization,
            "context_length": r.context_length,
            "cutoff_date": r.cutoff_date.isoformat() if r.cutoff_date else None,
            "source": r.source,
        }
        for r in rows
    ]


@router.post("/models/sync")
async def models_sync(request: Request) -> dict:
    """Synchronisiert die Registry mit den lokal verfügbaren Ollama-Modellen."""
    synced = await sync_models(session_factory, request.app.state.ollama)
    return {"synced": synced}


@router.patch("/models/{name:path}")
async def patch_model(name: str, patch: ModelPatch) -> dict:
    """Kuratiert Felder, die Ollama nicht liefert (Cutoff-Datum, Parameterzahl)."""
    async with session_factory() as session:
        info = await session.get(ModelInfo, name)
        if not info:
            raise HTTPException(status_code=404, detail=f"Modell nicht gefunden: {name}")
        if patch.cutoff_date is not None:
            info.cutoff_date = patch.cutoff_date
        if patch.parameter_count is not None:
            info.parameter_count = patch.parameter_count
        info.source = "ollama+kuratiert"
        session.add(info)
        await session.commit()
        await session.refresh(info)
    return {
        "name": info.name,
        "cutoff_date": info.cutoff_date.isoformat() if info.cutoff_date else None,
        "parameter_count": info.parameter_count,
    }
