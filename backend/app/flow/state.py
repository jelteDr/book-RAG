"""Zustand des Corrective-RAG-Graphen.

Kein Reducer: jeder Knoten liefert die Felder, die er ändert, komplett zurück — die
Schleifen sind kurz und so bleibt nachvollziehbar, wer was überschreibt.
"""

from __future__ import annotations

from typing import Any, TypedDict


class CragState(TypedDict, total=False):
    # Eingabe
    question: str  # Originalfrage (für Prompt + Grader)
    retrieval_query: str  # ggf. kondensierte Suchanfrage (Startwert der Suche)
    history: list[dict]
    group_ids: list[str] | None
    model: str
    top_k: int
    max_retries: int
    max_regen: int

    # Retrieval-Schleife
    queries: list[str]  # alle bisher gesuchten Anfragen (erste = retrieval_query)
    attempts: int  # Anzahl Suchläufe
    candidates: list[Any]  # ScoredPoints des LETZTEN Suchlaufs, noch unbewertet
    graded_ids: list[Any]  # IDs aller schon bewerteten Punkte (nicht doppelt bewerten)
    relevant: list[Any]  # als relevant bewertete ScoredPoints (über alle Versuche)
    graded: list[dict]  # Protokoll je bewerteter Passage (Grader-Kalibrierung im Eval)

    # Generierung + Prüfung
    context: list[Any]  # Quellen, die in den Prompt gingen ([1..n])
    answer: str
    usage: dict
    sources: list[dict]
    regen: int  # Anzahl Neu-Generierungen
    refused: bool

    trace: list[dict]  # Schritt-Protokoll (Knoten, Dauer, Detail) fürs done-Event
