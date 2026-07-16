"""Kleine Hilfsfunktionen."""

from __future__ import annotations

import math
import re
import unicodedata


def slugify(text: str) -> str:
    """Macht aus einem Anzeigenamen einen URL-/Filter-tauglichen Slug.

    „Härry Pötter!" -> "harry-potter". Fällt auf "gruppe" zurück, wenn nichts übrig bleibt.
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text or "gruppe"


def l2_normalize(vector: list[float]) -> list[float]:
    """L2-Normalisierung eines Vektors (für stabile Cosine-Ähnlichkeit)."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]
