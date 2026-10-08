"""Corrective-RAG-Graph (Exp 11) + Service-Fassade für RagService und Eval.

    retrieve -> grade --(genug relevant)------------------> generate -> check --ok--> END
                  |--(zu wenig, Retries übrig)-> rewrite -> retrieve     |
                  '--(nichts relevant, Retries aus)-> refuse -> END      '--zu viel unbelegt,
                                                                            Regen übrig--> generate

Gestreamt wird über LangGraphs `custom`-Stream: generate/refuse schreiben Token-Events,
vor einer Neu-Generierung ein `reset` (Frontend leert die Bubble).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from langgraph.graph import END, START, StateGraph

from app.flow.nodes import CragNodes
from app.flow.state import CragState
from app.rag.citations import sources_overview

Event = tuple[str, dict]


def build_crag_graph(nodes: CragNodes):
    g = StateGraph(CragState)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade", nodes.grade)
    g.add_node("rewrite", nodes.rewrite)
    g.add_node("generate", nodes.generate)
    g.add_node("check", nodes.check)
    g.add_node("refuse", nodes.refuse)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade", nodes.after_grade,
        {"generate": "generate", "rewrite": "rewrite", "refuse": "refuse"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "check")
    g.add_conditional_edges("check", nodes.after_check, {"generate": "generate", "end": END})
    g.add_edge("refuse", END)
    return g.compile()


class CragFlow:
    def __init__(
        self, nodes: CragNodes, *, enabled: bool, max_retries: int = 1, max_regen: int = 1
    ) -> None:
        self._enabled = enabled
        self._max_retries = max_retries
        self._max_regen = max_regen
        self.graph = build_crag_graph(nodes)

    @property
    def active(self) -> bool:
        return self._enabled

    def initial_state(
        self, question: str, *, retrieval_query: str | None = None, history: list[dict] | None = None,
        group_ids: list[str] | None = None, model: str, top_k: int,
        max_retries: int | None = None, max_regen: int | None = None,
    ) -> CragState:
        return {
            "question": question,
            "retrieval_query": retrieval_query or question,
            "history": history or [],
            "group_ids": group_ids,
            "model": model,
            "top_k": top_k,
            "max_retries": self._max_retries if max_retries is None else max_retries,
            "max_regen": self._max_regen if max_regen is None else max_regen,
            "trace": [],
        }

    async def invoke(self, question: str, **kwargs) -> CragState:
        """Kompletter Lauf ohne Streaming (Eval) — liefert den Endzustand."""
        return await self.graph.ainvoke(self.initial_state(question, **kwargs))

    async def run(self, question: str, **kwargs) -> AsyncIterator[Event]:
        """Streamt token/reset-Events und zum Schluss `done` (ohne Persistenz — das macht
        RagService). `done` trägt die Antwort unter `answer` mit, damit sie gespeichert
        werden kann."""
        t_start = time.perf_counter()
        ttft_ms: float | None = None
        final: CragState = {}
        try:
            async for mode, chunk in self.graph.astream(
                self.initial_state(question, **kwargs), stream_mode=["custom", "values"]
            ):
                if mode == "values":
                    final = chunk
                    continue
                event = chunk.pop("event")
                if event == "token" and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_start) * 1000
                if event == "reset":
                    ttft_ms = None  # TTFT zählt für die ausgelieferte Antwort
                yield event, chunk
        except Exception as exc:
            yield "error", {"message": f"Corrective RAG fehlgeschlagen: {exc}"}
            return

        e2e_ms = (time.perf_counter() - t_start) * 1000
        usage = final.get("usage") or {}
        completion_tokens = usage.get("completion_tokens") or 0
        decode_ms = max(e2e_ms - (ttft_ms or 0), 1e-6)
        yield "done", {
            "model": kwargs.get("model"),
            "answer": final.get("answer", ""),
            "ttft_ms": round(ttft_ms if ttft_ms is not None else e2e_ms, 1),
            "e2e_ms": round(e2e_ms, 1),
            "tps": round(completion_tokens / (decode_ms / 1000), 1) if completion_tokens else 0.0,
            "usage": usage,
            "sources": final.get("sources", []),
            "retrieved": sources_overview(final.get("context", [])),
            "mode": "crag",
            "crag": {
                "attempts": final.get("attempts", 0),
                "queries": final.get("queries", []),
                "n_graded": len(final.get("graded", [])),
                "n_relevant": len(final.get("relevant", [])),
                "regenerated": final.get("regen", 0) > 0,
                "refused": bool(final.get("refused")),
                "trace": final.get("trace", []),
            },
        }
