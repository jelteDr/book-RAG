"""Antwort-Varianten einer Unterhaltung: Verlauf für Folgefragen + Gruppierung für die Anzeige.

Eine Assistant-Nachricht verweist per `parent_id` auf ihre User-Nachricht; mehrere Antworten auf
dieselbe Frage (↻ „neu erzeugen“) sind Varianten, genau eine davon ist `selected`. Altdaten ohne
`parent_id` gelten als eigene Runde mit genau einer Variante.
"""

from __future__ import annotations

from typing import Any


def history_from(rows: list[Any]) -> list[dict]:
    """Verlauf für das LLM: alle User-Nachrichten + nur die ausgewählte Antwort je Runde."""
    return [
        {"role": m.role, "content": m.content}
        for m in rows
        if m.role != "assistant" or getattr(m, "selected", True) is not False
    ]


def group_turns(rows: list[Any]) -> list[dict]:
    """Nachrichten (nach id sortiert) -> Anzeige-Runden; Assistant-Runden mit allen Varianten."""
    turns: list[dict] = []
    by_parent: dict[int, dict] = {}
    for m in rows:
        if m.role != "assistant":
            turns.append({"role": m.role, "id": m.id, "content": m.content})
            continue
        variant = {"id": m.id, "content": m.content, "model": m.model, "sources": m.sources or []}
        parent = getattr(m, "parent_id", None)
        turn = by_parent.get(parent) if parent is not None else None
        if turn is None:
            turn = {"role": "assistant", "variants": [], "selected": 0}
            turns.append(turn)
            if parent is not None:
                by_parent[parent] = turn
        turn["variants"].append(variant)
        if getattr(m, "selected", True) is not False:
            turn["selected"] = len(turn["variants"]) - 1
    return turns
