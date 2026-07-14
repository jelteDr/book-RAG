"""Mappt die [n]-Marker aus der Antwort serverseitig auf die Quell-Chunks.

Das Modell rät NIE IDs — [n] ist nur ein Index in die dem Prompt bekannte
Reihenfolge. Halluzinierte Marker (n > Anzahl Quellen) werden verworfen.
"""

from __future__ import annotations

import re

from qdrant_client import models

_CITE_RE = re.compile(r"\[(\d+)\]")


def extract_citations(answer: str, points: list[models.ScoredPoint]) -> list[dict]:
    used = sorted({int(n) for n in _CITE_RE.findall(answer)})
    valid = [n for n in used if 1 <= n <= len(points)]

    sources = []
    for n in valid:
        point = points[n - 1]
        payload = point.payload or {}
        sources.append(
            {
                "marker": n,
                "book_title": payload.get("book_title"),
                "chapter": payload.get("chapter"),
                "score": round(point.score, 4),
                "char_start": payload.get("char_start"),
                "char_end": payload.get("char_end"),
                "text": payload.get("text"),
            }
        )
    return sources


def sources_overview(points: list[models.ScoredPoint]) -> list[dict]:
    """Alle retrievten Chunks (auch nicht zitierte) — für Transparenz im Frontend."""
    overview = []
    for i, point in enumerate(points, start=1):
        payload = point.payload or {}
        overview.append(
            {
                "marker": i,
                "book_title": payload.get("book_title"),
                "chapter": payload.get("chapter"),
                "score": round(point.score, 4),
            }
        )
    return overview
