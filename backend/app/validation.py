"""Eingabe-Validierung für Dokumente (vor dem Chunking) und Chat-Fragen.

Gibt jeweils eine Liste von Fehlermeldungen zurück (leer = gültig).
"""

from __future__ import annotations

import re

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

DOC_MIN_CHARS = 200
QUESTION_MIN_CHARS = 2
QUESTION_MAX_CHARS = 2000


def validate_document(cleaned: str) -> list[str]:
    """Prüft den BEREINIGTEN Text vor dem Chunking: Länge + gültige Zeichen."""
    errors: list[str] = []
    stripped = cleaned.strip()
    if len(stripped) < DOC_MIN_CHARS:
        errors.append(f"Text zu kurz ({len(stripped)} Zeichen, mindestens {DOC_MIN_CHARS}).")

    replacement = cleaned.count("�")
    if replacement > max(10, int(0.001 * len(cleaned))):
        errors.append(
            f"Zu viele unlesbare Zeichen ({replacement}) — vermutlich falsches Encoding."
        )

    non_space = sum(1 for c in cleaned if not c.isspace())
    letters = sum(1 for c in cleaned if c.isalpha())
    if non_space and letters / non_space < 0.5:
        errors.append("Zu geringer Buchstabenanteil — vermutlich kein sinnvoller Text.")

    return errors


def validate_question(question: str) -> list[str]:
    """Prüft eine Chat-Frage: Länge + keine Steuerzeichen."""
    errors: list[str] = []
    stripped = question.strip()
    if len(stripped) < QUESTION_MIN_CHARS:
        errors.append(f"Frage zu kurz (mindestens {QUESTION_MIN_CHARS} Zeichen).")
    if len(stripped) > QUESTION_MAX_CHARS:
        errors.append(f"Frage zu lang (maximal {QUESTION_MAX_CHARS} Zeichen).")
    if _CONTROL_RE.search(question):
        errors.append("Die Frage enthält ungültige Steuerzeichen.")
    return errors
