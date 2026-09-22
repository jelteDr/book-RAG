"""GraphService: lädt Gruppen-Graphen (+ Community-Index) lazy und stellt beide Graph-Pfade bereit.

- lokal  (`retrieve`):        Entity-Linking / Nachbar-Expansion, dense-erhaltende Fusion (Exp 8)
- global (`global_retrieve`): Community-Berichte als Quellen für thematische Fragen (Exp 9)

Opt-in via `graph_rag_enabled`; `active` ist nur True, wenn unter `graph_dir` mindestens ein
gebauter Graph liegt. Fehlt für die angefragten Gruppen ein Graph bzw. Community-Index,
fällt der Service still auf den jeweils einfacheren Pfad zurück (best-effort, wie Small-to-Big).
"""

from __future__ import annotations

from pathlib import Path

from qdrant_client import models

from app.clients.ollama_client import OllamaClient
from app.clients.qdrant_client import VectorStore
from app.graph.communities import CommunityIndex
from app.graph.global_search import global_retrieve
from app.graph.retriever import GraphParams, graph_retrieve
from app.graph.store import GRAPH_FILE, KnowledgeGraph, merge_graphs
from app.rag.retriever import retrieve


class GraphService:
    def __init__(
        self,
        graph_dir: str,
        enabled: bool,
        mode: str = "local",
        *,
        local_mode: str = "link",
        alpha: float = 0.03,
        top_m: int = 5,
        min_sim: float = 0.45,
        global_m: int = 6,
        global_map: bool = False,
        chunks_per_community: int = 2,
    ) -> None:
        self._dir = Path(graph_dir)
        self._enabled = enabled
        self.mode = mode  # local | global | auto
        self.local_mode = local_mode  # link | expand
        self.params = GraphParams(alpha=alpha, top_m=top_m, min_sim=min_sim)
        self.global_m = global_m
        self.global_map = global_map
        self.chunks_per_community = chunks_per_community
        self._graphs: dict[str, KnowledgeGraph | None] = {}
        self._merged: dict[tuple[str, ...], KnowledgeGraph | None] = {}
        self._indexes: dict[str, CommunityIndex | None] = {}
        self._merged_idx: dict[tuple[str, ...], CommunityIndex | None] = {}

    # --- Verfügbarkeit ---------------------------------------------------------
    def available_groups(self) -> list[str]:
        if not self._dir.is_dir():
            return []
        return sorted(p.parent.name for p in self._dir.glob(f"*/{GRAPH_FILE}"))

    @property
    def active(self) -> bool:
        return self._enabled and bool(self.available_groups())

    def status(self) -> dict:
        groups = self.available_groups()
        n_comm = sum(
            len(ix.reports) for g in groups if (ix := self.load_index(g)) is not None
        ) if self._enabled else 0
        return {"enabled": self._enabled, "active": self.active, "mode": self.mode,
                "local_mode": self.local_mode, "groups": groups, "communities": n_comm}

    # --- Laden (lazy, gecacht) -------------------------------------------------
    def load(self, group_id: str) -> KnowledgeGraph | None:
        if group_id not in self._graphs:
            path = self._dir / group_id
            try:
                self._graphs[group_id] = (
                    KnowledgeGraph.load(path) if (path / GRAPH_FILE).exists() else None
                )
            except Exception:
                self._graphs[group_id] = None  # defekte Datei -> Gruppe ohne Graph
        return self._graphs[group_id]

    def load_index(self, group_id: str) -> CommunityIndex | None:
        if group_id not in self._indexes:
            try:
                self._indexes[group_id] = CommunityIndex.load(self._dir / group_id)
            except Exception:
                self._indexes[group_id] = None
        return self._indexes[group_id]

    def _groups(self, group_ids: list[str] | None) -> tuple[str, ...]:
        return tuple(sorted(group_ids or self.available_groups()))

    def graph_for(self, group_ids: list[str] | None) -> KnowledgeGraph | None:
        groups = self._groups(group_ids)
        if groups not in self._merged:
            kgs = [kg for g in groups if (kg := self.load(g)) is not None]
            self._merged[groups] = merge_graphs(kgs) if kgs else None
        return self._merged[groups]

    def index_for(self, group_ids: list[str] | None) -> CommunityIndex | None:
        groups = self._groups(group_ids)
        if groups not in self._merged_idx:
            ixs = [ix for g in groups if (ix := self.load_index(g)) is not None
                   and ix.vectors is not None]
            self._merged_idx[groups] = CommunityIndex.merge(ixs) if ixs else None
        return self._merged_idx[groups]

    # --- Retrieval-Pfade -------------------------------------------------------
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
        """Lokaler Graph-Arm; Signatur wie `app.rag.retriever.retrieve` (+ optionaler Arm)."""
        groups = group_ids or ([group_id] if group_id else None)
        kg = self.graph_for(groups) if self.active else None
        if kg is None:
            return await retrieve(
                question, ollama=ollama, vectors=vectors, embed_model=embed_model,
                top_k=top_k, group_id=group_id, group_ids=group_ids,
            )
        return await graph_retrieve(
            question, kg=kg, mode=mode or self.local_mode, ollama=ollama, vectors=vectors,
            embed_model=embed_model, top_k=top_k, group_id=group_id, group_ids=group_ids,
            params=self.params, diagnostics=diagnostics,
        )

    async def global_retrieve(
        self,
        question: str,
        *,
        ollama: OllamaClient,
        vectors: VectorStore,
        embed_model: str,
        model: str,
        group_id: str | None = None,
        group_ids: list[str] | None = None,
        use_map: bool | None = None,
        m: int | None = None,
    ) -> tuple[list[models.ScoredPoint], dict] | None:
        """Globaler Pfad; None, wenn für die Gruppen kein Community-Index existiert."""
        groups = group_ids or ([group_id] if group_id else None)
        index = self.index_for(groups) if self.active else None
        if index is None:
            return None
        return await global_retrieve(
            question, index=index, ollama=ollama, vectors=vectors, embed_model=embed_model,
            model=model, m=m or self.global_m,
            use_map=self.global_map if use_map is None else use_map,
            chunks_per_community=self.chunks_per_community,
        )
