"""Kleine Hilfsfunktionen."""

from __future__ import annotations

import math


def l2_normalize(vector: list[float]) -> list[float]:
    """L2-Normalisierung eines Vektors (für stabile Cosine-Ähnlichkeit)."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]
