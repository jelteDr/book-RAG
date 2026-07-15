"""Baut die Chat-Messages mit nummerierten Quell-Blöcken [1..k] und Zitier-Contract."""

from __future__ import annotations

from qdrant_client import models

_SYSTEM = (
    "Du bist ein präziser Assistent, der Fragen AUSSCHLIESSLICH anhand der "
    "bereitgestellten, nummerierten Quellen beantwortet. Regeln:\n"
    "1. Setze hinter JEDE Aussage direkt den Marker der Quelle, aus der sie stammt, "
    "z. B.: 'Harker reist nach Transsilvanien [1].'\n"
    "2. Nutze NUR die vorhandenen Nummern [1..k] — erfinde weder Nummern noch Fakten.\n"
    "3. Belege jede Aussage; was du nicht aus den Quellen belegen kannst, lässt du weg.\n"
    "4. Reichen die Quellen nicht, sage das ehrlich ('Dazu steht in den Quellen nichts.').\n"
    "5. Hänge KEINE separate Quellenliste an — die Marker im Fließtext genügen."
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
