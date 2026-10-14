"""Strukturbewusstes Chunking: erst Kapitel erkennen, dann Absätze zu Chunks packen.

Offsets (`char_start`/`char_end`) beziehen sich auf den BEREINIGTEN Text (das
De-Wrapping hat die Positionen verschoben) — damit ein Zitat-Chip später die
richtige Stelle markiert. Chunk-Größe ist zeichenbasiert approximiert
(~4 Zeichen/Token); eine tokenizer-genaue Messung ist eine spätere Verfeinerung.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_HEADING_RE = re.compile(
    r"^(CHAPTER|Chapter|KAPITEL|Kapitel|BOOK|Book)\b.{0,70}$"
)
_CHAPTER_TOKEN_RE = re.compile(r"\bCHAPTER\b|\bKAPITEL\b", re.IGNORECASE)
# Satzzeichen, die in einer Überschrift nicht vorkommen (sonst ist es gerufene Rede o. Ä.).
_SENTENCE_PUNCT = set('.!?,;:"“”„')


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


def _is_caps_heading(paragraph: str) -> bool:
    """Überschrift ganz in Großbuchstaben ohne Kapitel-Wort, z. B. „THE RIDDLE HOUSE“
    (Harry Potter) oder POV-Namen wie „BRAN“ / „PROLOGUE“ (Game of Thrones)."""
    s = paragraph.strip()
    return (
        3 <= len(s) <= 60
        and "\n" not in s
        and s == s.upper()
        and sum(c.isalpha() for c in s) >= 3
        and not any(c in _SENTENCE_PUNCT for c in s)
    )


_MARKER_RE = re.compile(r"^(?:[IVXLC]+|[-–—]?\s*\d+|[^A-Za-zÀ-ÿ]*)$")


def _normalize(paragraph: str) -> str:
    return re.sub(r"[\d\s|]+", " ", paragraph).strip().lower()


def _heading_flags(paras: list[tuple[str, int]]) -> list[bool]:
    """Welche Absätze sind Kapitel-Überschriften?

    Kapitel-Wort-Überschriften haben Vorrang. Nur Bücher OHNE solche Überschriften nutzen
    Großbuchstaben-Überschriften — und auch die nur direkt nach einem Kapitel-Marker
    (römische/arabische Zahl, Trennzeichen) oder einer wiederkehrenden Kopf-/Fußzeile
    („Page | 19 Harry Potter and the …“). So werden „DUDLEY DEMENTED“ (nach „I“) oder
    „THE SCAR“ (nach der Fußzeile) Kapitel, nicht aber „WHAM“ oder Zeitungsschlagzeilen
    mitten im Erzähltext. Ohne diese Regel würden auch Untertitel wie
    „JONATHAN HARKER'S JOURNAL“ (Dracula, unter „CHAPTER I“) zu eigenen Kapiteln.
    """
    texts = [p for p, _ in paras]
    if any(_is_heading(p) for p in texts):
        return [_is_heading(p) for p in texts]
    counts = Counter(_normalize(p) for p in texts if len(p) <= 150)
    boilerplate = {t for t, n in counts.items() if n >= 5 and t}

    def after_marker(i: int) -> bool:
        if i == 0:
            return False
        prev = texts[i - 1].strip()
        return bool(_MARKER_RE.match(prev)) or _normalize(prev) in boilerplate

    return [_is_caps_heading(p) and after_marker(i) for i, p in enumerate(texts)]


def chapter_starts(cleaned_text: str) -> list[tuple[int, str]]:
    """(Offset, Titel) aller Kapitel-Überschriften im bereinigten Text — damit lassen sich
    vorhandene Chunks nachträglich einem Kapitel zuordnen, ohne neu zu chunken."""
    paras = _paragraphs_with_offsets(cleaned_text)
    flags = _heading_flags(paras)
    return [(start, p.strip()) for (p, start), f in zip(paras, flags) if f and not _is_toc(p)]


def _is_toc(paragraph: str) -> bool:
    # Zusammengefallenes Inhaltsverzeichnis: viele Kapitel-Marker in einem Absatz.
    return len(_CHAPTER_TOKEN_RE.findall(paragraph)) >= 3


def chunk_book(
    cleaned_text: str, *, target_chars: int = 1800, overlap_chars: int = 200
) -> list[Chunk]:
    paras = _paragraphs_with_offsets(cleaned_text)
    flags = _heading_flags(paras)

    # In Kapitel gruppieren (Überschrift-Absätze als Titel, TOC verwerfen).
    chapters: list[tuple[str | None, list[tuple[str, int]]]] = []
    current_title: str | None = None
    current: list[tuple[str, int]] = []
    for (ptext, pstart), is_heading in zip(paras, flags):
        if _is_toc(ptext):
            continue
        if is_heading:
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
