"""Baut die Chat-Messages mit nummerierten Quell-Blöcken [1..k] und Zitier-Contract."""

from __future__ import annotations

from qdrant_client import models

_SYSTEM = (
    "Du bist ein präziser Assistent, der Fragen AUSSCHLIESSLICH anhand der "
    "bereitgestellten, nummerierten Quellen beantwortet. Antworte in Fließtext und "
    "setze DIREKT hinter jede Aussage den passenden Quellmarker in eckigen Klammern, "
    "z. B.: 'Harker reist nach Transsilvanien, um ein Immobiliengeschäft abzuwickeln [1].' "
    "Hänge KEINE separate Quellenliste an und nutze nur die vorhandenen Nummern [1..k]. "
    "Erfinde keine Fakten. Reichen die Quellen nicht, sage das ehrlich und rate nicht."
)


def build_messages(question: str, points: list[models.ScoredPoint]) -> list[dict]:
    # Nur die Nummer [i] als Präfix — Metadaten bleiben serverseitig, damit das
    # Modell keine "Buch:/Kapitel:"-Header in die Antwort kopiert.
    blocks = [
        f"[{i}] {(point.payload or {}).get('text', '')}"
        for i, point in enumerate(points, start=1)
    ]
    context = "\n\n".join(blocks)

    user = (
        f"Quellen:\n{context}\n\n"
        f"Frage: {question}\n\n"
        "Antworte auf Deutsch und markiere jede Aussage mit der passenden Quelle [n]."
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]
