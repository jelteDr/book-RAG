"""Einlesen & Metadaten: robustes Encoding-Sniffing und Gutenberg-Aufbereitung.

- `read_text`: liest Bytes und dekodiert mit charset-normalizer (robuster als
  blindes UTF-8 — GoT-Bände sind teils us-ascii / unknown-8bit).
- `parse_gutenberg`: schneidet den Text auf den Inhalt zwischen den
  `*** START/END OF …`-Markern und liest Titel/Autor/Sprache aus dem Header.
  Fällt sauber zurück, wenn keine Marker vorhanden sind.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from charset_normalizer import from_bytes

_START_RE = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE)
_END_RE = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE)
_META_RE = {
    "title": re.compile(r"^Title:\s*(.+)$", re.MULTILINE),
    "author": re.compile(r"^Author:\s*(.+)$", re.MULTILINE),
    "language": re.compile(r"^Language:\s*(.+)$", re.MULTILINE),
}


@dataclass
class ParsedBook:
    body: str
    title: str | None
    author: str | None
    language: str | None
    had_gutenberg_markers: bool


def decode_bytes(raw: bytes) -> str:
    """Dekodiert Rohbytes mit erkanntem Encoding (charset-normalizer)."""
    match = from_bytes(raw).best()
    if match is None:  # Fallback: UTF-8 mit Ersatzzeichen
        return raw.decode("utf-8", errors="replace")
    return str(match)


def read_text(path: str | Path) -> str:
    """Liest eine Textdatei und dekodiert mit erkanntem Encoding."""
    return decode_bytes(Path(path).read_bytes())


def parse_gutenberg(text: str) -> ParsedBook:
    """Trennt Header/Boilerplate ab und extrahiert Metadaten."""
    header = text[:5000]
    title = _first(_META_RE["title"], header)
    author = _first(_META_RE["author"], header)
    language = _first(_META_RE["language"], header)

    start = _START_RE.search(text)
    end = _END_RE.search(text)
    if start and end and start.end() < end.start():
        body = text[start.end() : end.start()]
        had_markers = True
    else:
        # Kein Gutenberg-Rahmen (z. B. Kaggle-Dump) -> ganzen Text nehmen.
        body = text
        had_markers = False

    return ParsedBook(
        body=body.strip(),
        title=title,
        author=author,
        language=language,
        had_gutenberg_markers=had_markers,
    )


def _first(pattern: re.Pattern[str], text: str) -> str | None:
    m = pattern.search(text)
    return m.group(1).strip() if m else None
