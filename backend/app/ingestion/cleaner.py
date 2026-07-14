"""Deterministische Textbereinigung + Cleaning-Report (Dry-Run-Prüf-Gate).

Reihenfolge (bewusst): Unicode-Normalisierung -> Zeichen-Normalisierung ->
De-Wrapping (harte Umbrüche mergen, De-Hyphenation) -> Artefakte (Seitenzahlen,
überzählige Leerzeilen). Der Report macht die Bereinigung überprüfbar, BEVOR
etwas in DB/Index landet.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Einzelzeichen-Ersetzungen (Smart Quotes, Dashes, Ligaturen, Zero-Width).
_TRANSLATE = {
    # Anführungszeichen -> gerade
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "«": '"', "»": '"',
    # Bindestriche/Dashes -> Bindestrich-Minus (em-dash — bleibt erhalten)
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "―": "-",
    # Auslassungspunkte
    "…": "...",
    # Zero-Width / BOM -> entfernen
    "​": "", "‌": "", "‍": "", "﻿": "",
    # geschütztes Leerzeichen -> normales
    " ": " ",
}
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
}
_TABLE = {ord(k): v for k, v in {**_TRANSLATE, **_LIGATURES}.items()}

# Verbleibende C0-Steuerzeichen außer \n und \t.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_PAGENUM_RE = re.compile(r"^\s*\d{1,4}\s*$")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")


@dataclass
class CleaningReport:
    original_chars: int
    cleaned_chars: int
    lines_merged: int = 0
    pagenumber_lines_removed: int = 0
    control_chars_removed: int = 0
    special_chars_normalized: int = 0
    nonascii_ratio_after: float = 0.0
    dropped_ratio: float = 0.0
    warnings: list[str] = field(default_factory=list)
    before_sample: str = ""
    after_sample: str = ""

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def clean(text: str, *, max_dropped_ratio: float = 0.4) -> tuple[str, CleaningReport]:
    original_len = len(text)
    before_sample = _sample(text)

    # 1) Unicode-Normalisierung (NFC).
    text = unicodedata.normalize("NFC", text)

    # 2) Steuerzeichen zählen + entfernen.
    control_removed = len(_CONTROL_RE.findall(text))
    text = _CONTROL_RE.sub("", text)

    # 3) Zeichen-Normalisierung (Smart Quotes/Dashes/Ligaturen/Zero-Width).
    special_count = sum(text.count(chr(cp)) for cp in _TABLE)
    text = text.translate(_TABLE)

    # 4) Zeilen normalisieren (CRLF -> LF) und De-Wrapping.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned, merged, pagenums = _dewrap(text)

    # 5) Whitespace glätten, überzählige Leerzeilen auf max. eine reduzieren.
    cleaned = _MULTISPACE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    nonascii = sum(1 for c in cleaned if ord(c) > 127)
    report = CleaningReport(
        original_chars=original_len,
        cleaned_chars=len(cleaned),
        lines_merged=merged,
        pagenumber_lines_removed=pagenums,
        control_chars_removed=control_removed,
        special_chars_normalized=special_count,
        nonascii_ratio_after=round(nonascii / max(len(cleaned), 1), 4),
        dropped_ratio=round(1 - len(cleaned) / max(original_len, 1), 4),
        before_sample=before_sample,
        after_sample=_sample(cleaned),
    )

    # Guardrails.
    if report.dropped_ratio > max_dropped_ratio:
        report.warnings.append(
            f"Hoher Textverlust ({report.dropped_ratio:.0%}) — Verdacht auf falsches "
            f"Encoding oder kaputte Datei."
        )
    if report.nonascii_ratio_after > 0.30:
        report.warnings.append(
            f"Hoher Nicht-ASCII-Anteil ({report.nonascii_ratio_after:.0%}) — Encoding prüfen."
        )
    return cleaned, report


def _dewrap(text: str) -> tuple[str, int, int]:
    """Führt harte Zeilenumbrüche innerhalb von Absätzen zusammen.

    Absätze sind durch Leerzeilen getrennt. Innerhalb eines Absatzes werden
    Zeilen mit Leerzeichen verbunden; endet eine Zeile auf 'wort-' und die
    nächste beginnt klein, wird de-hyphenisiert. Reine Zahlenzeilen (Seitenzahlen)
    werden verworfen.
    """
    lines = text.split("\n")
    paragraphs: list[str] = []
    current: list[str] = []
    merged = 0
    pagenums = 0

    def flush() -> None:
        nonlocal merged
        if not current:
            return
        buf = current[0]
        for nxt in current[1:]:
            if re.search(r"[A-Za-zÀ-ÿ]-$", buf) and nxt[:1].islower():
                buf = buf[:-1] + nxt  # De-Hyphenation
            else:
                buf = buf + " " + nxt
            merged += 1
        paragraphs.append(buf.strip())
        current.clear()

    for line in lines:
        stripped = line.strip()
        if stripped == "":
            flush()
        elif _PAGENUM_RE.match(stripped):
            pagenums += 1  # Seitenzahl -> überspringen
        else:
            current.append(stripped)
    flush()

    return "\n\n".join(p for p in paragraphs if p), merged, pagenums


def _sample(text: str, n: int = 300) -> str:
    """Kurzes, einzeiliges Sample (Reports können in gitignore-te Pfade; hier gekürzt)."""
    return " ".join(text[:n].split())
