"""Optionaler Cross-Encoder-Reranker (bge-reranker-v2-m3).

Opt-in via Config (`reranker_enabled`) und lazy geladen — das schwere Modell/torch
kostet nur dann Speicher, wenn wirklich gerankt wird. Fehlt die reranker-Dependency-
Gruppe, fällt der Service still auf Dense-only zurück.
"""

from __future__ import annotations

from qdrant_client import models


class RerankerService:
    def __init__(self, model_name: str, enabled: bool) -> None:
        self._model_name = model_name
        self._enabled = enabled
        self._model = None  # lazy

    @property
    def active(self) -> bool:
        return self._enabled

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            return False  # reranker-Gruppe nicht installiert -> Dense-only
        self._model = CrossEncoder(self._model_name, max_length=512)
        return True

    def rerank(
        self, query: str, points: list[models.ScoredPoint], top_k: int
    ) -> list[models.ScoredPoint]:
        """Blockierend (torch) — in RagService via asyncio.to_thread aufrufen."""
        if not self._enabled or not points or not self._ensure_model():
            return points[:top_k]
        pairs = [(query, (p.payload or {}).get("text", "")) for p in points]
        scores = self._model.predict(pairs)
        order = sorted(range(len(points)), key=lambda i: scores[i], reverse=True)
        return [points[i] for i in order[:top_k]]
