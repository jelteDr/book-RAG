"""Synchronisiert die model_info-Tabelle mit Ollama (`/api/show`).

Auto-Felder (Familie, Parametergröße, Quantisierung, Kontextlänge) kommen aus Ollama;
`cutoff_date` (und ein manuell gesetzter `parameter_count`) sind **kuratiert** und werden
beim Sync NICHT überschrieben — Ollama liefert kein Trainings-Cutoff.
"""

from __future__ import annotations

import re
from datetime import datetime

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.clients.ollama_client import OllamaClient
from app.db.models import ModelInfo

_SIZE_RE = re.compile(r"([\d.]+)\s*([BMK])", re.IGNORECASE)
_MULT = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}


def _parse_param_count(size: str | None) -> int | None:
    if not size:
        return None
    m = _SIZE_RE.match(size.strip())
    if not m:
        return None
    return int(float(m.group(1)) * _MULT[m.group(2).upper()])


def _context_length(show: dict) -> int | None:
    info = show.get("model_info") or {}
    for key, value in info.items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    return None


async def sync_models(session_factory: async_sessionmaker, ollama: OllamaClient) -> int:
    """Upsertet alle lokal verfügbaren Modelle. Gibt die Anzahl synchronisierter zurück."""
    tags = await ollama.list_models()
    names = [t["name"] for t in tags if t.get("name")]

    count = 0
    async with session_factory() as session:
        for name in names:
            try:
                show = await ollama.show(name)
            except Exception:
                continue  # Modell nicht abfragbar -> überspringen
            details = show.get("details") or {}

            info = await session.get(ModelInfo, name) or ModelInfo(name=name)
            info.family = details.get("family")
            info.parameter_size = details.get("parameter_size")
            info.quantization = details.get("quantization_level")
            info.context_length = _context_length(show)
            if info.parameter_count is None:  # kuratierten Wert nicht überschreiben
                info.parameter_count = _parse_param_count(info.parameter_size)
            info.source = "ollama+kuratiert" if info.cutoff_date else "ollama"
            info.updated_at = datetime.utcnow()

            session.add(info)
            count += 1
        await session.commit()
    return count
