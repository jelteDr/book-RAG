"""Strukturbewusstes Chunking: erst Kapitel erkennen, dann Absätze zu Chunks packen.

Offsets (`char_start`/`char_end`) beziehen sich auf den BEREINIGTEN Text (das
De-Wrapping hat die Positionen verschoben) — damit ein Zitat-Chip später die
richtige Stelle markiert. Chunk-Größe ist zeichenbasiert approximiert
(~4 Zeichen/Token); eine tokenizer-genaue Messung ist eine spätere Verfeinerung.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(
    r"^(CHAPTER|Chapter|KAPITEL|Kapitel|BOOK|Book)\b.{0,70}$"
)
_CHAPTER_TOKEN_RE = re.compile(r"\bCHAPTER\b|\bKAPITEL\b", re.IGNORECASE)


@dataclass
class Chunk:
    text: str
    chapter: str | None
    chunk_index: int
    char_start: int
    char_end: int


def _paragraphs_with_offsets(text: str) -> list[tuple[str, int]]:
    """Absätze (durch Leerzeile getrennt) mit ihrem Start-Offset im Text."""
    out: list[tuple[str, int]] = []
    cursor = 0
    for part in text.split("\n\n"):
        if part.strip() == "":
            cursor += len(part) + 2
            continue
        start = text.index(part, cursor)
        out.append((part, start))
        cursor = start + len(part)
    return out


def _is_heading(paragraph: str) -> bool:
    return bool(_HEADING_RE.match(paragraph)) and len(paragraph) <= 80


def _is_toc(paragraph: str) -> bool:
    # Zusammengefallenes Inhaltsverzeichnis: viele Kapitel-Marker in einem Absatz.
    return len(_CHAPTER_TOKEN_RE.findall(paragraph)) >= 3


def chunk_book(
    cleaned_text: str, *, target_chars: int = 1800, overlap_chars: int = 200
) -> list[Chunk]:
    paras = _paragraphs_with_offsets(cleaned_text)

    # In Kapitel gruppieren (Überschrift-Absätze als Titel, TOC verwerfen).
    chapters: list[tuple[str | None, list[tuple[str, int]]]] = []
    current_title: str | None = None
    current: list[tuple[str, int]] = []
    for ptext, pstart in paras:
        if _is_toc(ptext):
            continue
        if _is_heading(ptext):
            if current:
                chapters.append((current_title, current))
                current = []
            current_title = ptext.strip()
            continue
        current.append((ptext, pstart))
    if current:
        chapters.append((current_title, current))

    # Innerhalb jedes Kapitels Absätze zu Chunks packen (mit Overlap).
    chunks: list[Chunk] = []
    idx = 0
    for title, plist in chapters:
        i = 0
        while i < len(plist):
            buf: list[tuple[str, int]] = []
            clen = 0
            j = i
            while j < len(plist) and clen < target_chars:
                buf.append(plist[j])
                clen += len(plist[j][0]) + 2
                j += 1

            text = "\n\n".join(p[0] for p in buf)
            char_start = buf[0][1]
            char_end = buf[-1][1] + len(buf[-1][0])
            chunks.append(Chunk(text, title, idx, char_start, char_end))
            idx += 1

            if j >= len(plist):
                break

            # Overlap: einige Absätze am Ende erneut in den nächsten Chunk nehmen.
            back = 0
            ov = 0
            k = j - 1
            while k > i and ov < overlap_chars:
                ov += len(plist[k][0])
                k -= 1
                back += 1
            i = max(j - back, i + 1)  # Fortschritt garantieren

    return chunks
