"""Routing lokal/global (nur bei GRAPH_RAG_MODE=auto).

Stufe 1: billige Regex-Heuristik auf der deutschen Frage — thematische Signalwörter
(Thema, Entwicklung, Beziehung zwischen, welche Rolle …) -> global; klassische Faktfrage
(Wann/Wo/Wer/Wie viele …) -> local. Stufe 2 nur bei Unentschieden: 1-Token-Klassifikator
(LOCAL/GLOBAL, ~1 s). Im Zweifel local — das ist der gemessene Standardpfad.
"""

from __future__ import annotations

import re

from app.clients.ollama_client import OllamaClient

_GLOBAL_RE = re.compile(
    r"\b(thema|themen|motiv|entwickl\w*|im verlauf|gesamten roman|ganzen roman|"
    r"beziehung(en)? zwischen|verhältnis zwischen|welche rolle|welche bedeutung|"
    r"erzähl\w*|symbol\w*|vergleich\w*|kontrast\w*|unterschied\w*|verändert sich|"
    r"über den roman|charakterisier\w*|dargestellt|inszeniert)\b",
    re.IGNORECASE,
)
_LOCAL_RE = re.compile(
    r"^\s*(wann|wo|wer|wem|wen|wie viele|wie heißt|wie hieß|welches schiff|"
    r"in welchem kapitel|an welchem tag)\b",
    re.IGNORECASE,
)
_CLASSIFY_SYSTEM = (
    "Classify the user's question about a novel. Answer with exactly one word:\n"
    "LOCAL  = asks for a specific fact, event, name, place or passage.\n"
    "GLOBAL = asks about themes, character development, relationships across the book, "
    "narrative technique or a comparison spanning the whole novel."
)


def heuristic_mode(question: str) -> str | None:
    if _GLOBAL_RE.search(question):
        return "global"
    if _LOCAL_RE.search(question):
        return "local"
    return None


async def decide_mode(question: str, *, ollama: OllamaClient, model: str) -> str:
    mode = heuristic_mode(question)
    if mode:
        return mode
    try:
        out = await ollama.complete(
            [{"role": "system", "content": _CLASSIFY_SYSTEM},
             {"role": "user", "content": question}],
            model, temperature=0.0, max_tokens=3,
        )
        return "global" if "GLOBAL" in out.upper() else "local"
    except Exception:
        return "local"
