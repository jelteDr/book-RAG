"""Small-to-Big: Antwortzeit-Erweiterung der besten Treffer um Nachbar-Chunks.

Die Suche bleibt auf den (kleinen) Original-Chunks — das hält die Treffer präzise.
Erst NACH dem Retrieval werden die besten `top_n` Treffer um ihre Nachbar-Chunks
(gleiches Buch, chunk_index ± window) erweitert, damit das Modell die Passage im
Zusammenhang liest. Die Nachbarn stehen bereits in Qdrant und sind über die
deterministischen Punkt-IDs (book_id + chunk_index) direkt adressierbar — es ist
also KEIN Re-Embedding nötig, nur ein Lookup pro erweitertem Treffer.

Aufeinanderfolgende Chunks überlappen (~200 Zeichen); beim Zusammenfügen wird der
doppelte Teil über die char-Offsets exakt abgeschnitten, sodass der erweiterte
Text die zusammenhängende Originalpassage ergibt.
"""

from __future__ import annotations

from qdrant_client import models

from app.clients.qdrant_client import VectorStore
from app.rag.retriever import dedup_overlaps

# Verbindet Textteile, zwischen denen etwas fehlt (z. B. eine übersprungene
# Kapitelüberschrift — die steht in keinem Chunk).
_GAP_MARKER = "\n[…]\n"


def merge_payload_texts(payloads: list[dict]) -> tuple[str, int, int]:
    """Fügt nach chunk_index sortierte Payloads zu einer zusammenhängenden Passage.

    Liefert (text, char_start, char_end). Voraussetzung (gilt für den Chunker):
    `payload.text` entspricht exakt `bereinigter_text[char_start:char_end]`.
    """
    ordered = sorted(payloads, key=lambda pl: pl["chunk_index"])
    parts = [ordered[0]["text"]]
    start = ordered[0]["char_start"]
    end = ordered[0]["char_end"]
    for pl in ordered[1:]:
        if pl["char_end"] <= end:
            continue  # Nachbar liegt komplett im bereits abgedeckten Bereich
        if pl["char_start"] <= end:
            parts.append(pl["text"][end - pl["char_start"]:])
        else:
            parts.append(_GAP_MARKER + pl["text"])
        end = pl["char_end"]
    return "".join(parts), start, end


async def expand_points(
    points: list[models.ScoredPoint],
    *,
    vectors: VectorStore,
    window: int = 1,
    top_n: int = 3,
) -> list[models.ScoredPoint]:
    """Erweitert die besten `top_n` Treffer um ± `window` Nachbar-Chunks.

    Nur die Spitze wird erweitert, damit das Prompt-Budget nicht explodiert
    (jeder erweiterte Treffer wird bis zu (2*window+1)-mal so lang). Erweiterte
    Fenster können danach andere Treffer abdecken — deshalb ein zweiter
    Dedup-Durchlauf über die fertigen Bereiche.
    """
    if window < 1 or top_n < 1:
        return points

    expanded: list[models.ScoredPoint] = []
    for rank, point in enumerate(points):
        payload = dict(point.payload or {})
        if (
            rank >= top_n
            or payload.get("chunk_index") is None
            or payload.get("char_start") is None
        ):
            expanded.append(point)
            continue

        idx = payload["chunk_index"]
        wanted = [i for i in range(idx - window, idx + window + 1) if i != idx and i >= 0]
        neighbors = await vectors.get_by_indices(payload["book_id"], wanted)
        group = [payload] + [
            dict(r.payload)
            for r in neighbors
            if r.payload and r.payload.get("char_start") is not None
        ]
        text, char_start, char_end = merge_payload_texts(group)
        payload.update(text=text, char_start=char_start, char_end=char_end, expanded=True)
        expanded.append(
            models.ScoredPoint(
                id=point.id, version=point.version, score=point.score,
                payload=payload, vector=None,
            )
        )

    return dedup_overlaps(expanded, len(points))
