"""GraphService: lädt Gruppen-Graphen lazy und stellt den Graph-Retrieval-Arm bereit.

Opt-in via `graph_rag_enabled`; `active` ist nur True, wenn unter `graph_dir` mindestens
ein gebauter Graph liegt. Fehlt für die angefragten Gruppen ein Graph, fällt der
Service still auf das normale dense Retrieval zurück (best-effort, wie Small-to-Big).
"""

from __future__ import annotations

from pathlib import Path

from qdrant_client import models

from app.clients.ollama_client import OllamaClient
from app.clients.qdrant_client import VectorStore
from app.graph.retriever import GraphParams, graph_retrieve
from app.graph.store import GRAPH_FILE, KnowledgeGraph, merge_graphs
from app.rag.retriever import retrieve


class GraphService:
    def __init__(
        self, graph_dir: str, enabled: bool, mode: str = "link",
        *, alpha: float = 0.03, top_m: int = 5, min_sim: float = 0.45,
    ) -> None:
        self._dir = Path(graph_dir)
        self._enabled = enabled
        self.mode = mode
        self.params = GraphParams(alpha=alpha, top_m=top_m, min_sim=min_sim)
        self._graphs: dict[str, KnowledgeGraph | None] = {}
        self._merged: dict[tuple[str, ...], KnowledgeGraph | None] = {}

    def available_groups(self) -> list[str]:
        if not self._dir.is_dir():
            return []
        return sorted(p.parent.name for p in self._dir.glob(f"*/{GRAPH_FILE}"))

    @property
    def active(self) -> bool:
        return self._enabled and bool(self.available_groups())

    def status(self) -> dict:
        return {"enabled": self._enabled, "active": self.active, "mode": self.mode,
                "groups": self.available_groups()}

    def load(self, group_id: str) -> KnowledgeGraph | None:
        if group_id not in self._graphs:
            path = self._dir / group_id
            try:
                self._graphs[group_id] = KnowledgeGraph.load(path) if (path / GRAPH_FILE).exists() else None
            except Exception:
                self._graphs[group_id] = None  # defekte Datei -> Gruppe ohne Graph
        return self._graphs[group_id]

    def graph_for(self, group_ids: list[str] | None) -> KnowledgeGraph | None:
        groups = tuple(sorted(group_ids or self.available_groups()))
        if groups not in self._merged:
            kgs = [kg for g in groups if (kg := self.load(g)) is not None]
            self._merged[groups] = merge_graphs(kgs) if kgs else None
        return self._merged[groups]

    async def retrieve(
        self,
        question: str,
        *,
        ollama: OllamaClient,
        vectors: VectorStore,
        embed_model: str,
        top_k: int,
        group_id: str | None = None,
        group_ids: list[str] | None = None,
        mode: str | None = None,
        diagnostics: dict | None = None,
    ) -> list[models.ScoredPoint]:
        """Gleiche Signatur wie `app.rag.retriever.retrieve` (+ optionaler Modus)."""
        groups = group_ids or ([group_id] if group_id else None)
        kg = self.graph_for(groups) if self.active else None
        if kg is None:
            return await retrieve(
                question, ollama=ollama, vectors=vectors, embed_model=embed_model,
                top_k=top_k, group_id=group_id, group_ids=group_ids,
            )
        return await graph_retrieve(
            question, kg=kg, mode=mode or self.mode, ollama=ollama, vectors=vectors,
            embed_model=embed_model, top_k=top_k, group_id=group_id, group_ids=group_ids,
            params=self.params, diagnostics=diagnostics,
        )
