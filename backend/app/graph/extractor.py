"""Entity-/Relations-Extraktion je Chunk (Graph-RAG, Exp 8).

Ein LLM-Aufruf pro Chunk liefert ein kleines JSON mit Entities (Figuren, Orte,
Organisationen, Objekte, Ereignisse) und Relationen zwischen ihnen. Der Parser ist
bewusst tolerant (Code-Fences, Text um das JSON herum, unbekannte Typen), damit ein
verrauschter 7B-Output nie den ganzen Lauf abbricht — ein Chunk ohne verwertbares
JSON wird als leer + Fehlertext protokolliert (Muster: ingestion/contextualizer.py).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.clients.ollama_client import OllamaClient

ENTITY_TYPES = ("PERSON", "PLACE", "ORGANISATION", "OBJECT", "EVENT")
MAX_ENTITIES = 8
MAX_RELATIONS = 8
MAX_DESC_CHARS = 120
MAX_TOKENS = 600  # Pilot: 450 schnitt ~40 % der Antworten ab (pretty-printed JSON)

# Häufige Abweichungen des Modells auf die kanonischen Typen abbilden.
_TYPE_ALIASES = {
    "PERSON": "PERSON", "CHARACTER": "PERSON", "PEOPLE": "PERSON", "HUMAN": "PERSON",
    "PLACE": "PLACE", "LOCATION": "PLACE", "GEO": "PLACE", "BUILDING": "PLACE",
    "ORGANISATION": "ORGANISATION", "ORGANIZATION": "ORGANISATION", "GROUP": "ORGANISATION",
    "OBJECT": "OBJECT", "ITEM": "OBJECT", "THING": "OBJECT", "ANIMAL": "OBJECT",
    "EVENT": "EVENT", "ACTION": "EVENT",
}

EXTRACT_PROMPT = (
    'You extract a knowledge graph from an excerpt of the novel "{title}" ({chapter}).\n'
    "{context_line}"
    "---\n{text}\n---\n\n"
    "Return ONLY a compact JSON object (one line, no whitespace) of exactly this shape:\n"
    '{{"entities":[{{"name":"...","type":"PERSON|PLACE|ORGANISATION|OBJECT|EVENT","desc":"..."}}],'
    '"relations":[{{"source":"...","target":"...","rel":"..."}}]}}\n'
    "Rules:\n"
    f"- At most {MAX_ENTITIES} entities and {MAX_RELATIONS} relations; "
    "only named entities that actually occur in the excerpt (no generic nouns like "
    '"train" or "breakfast").\n'
    "- name: the fullest canonical name used anywhere in the excerpt or context "
    '(e.g. "Jonathan Harker", not "Harker" or "he"; "Count Dracula", not "the Count"). '
    "Never a pronoun.\n"
    "- desc: at most 10 words, grounded in the excerpt.\n"
    "- source and target must be names from entities; rel is 1-3 words.\n"
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class Extraction:
    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    error: str | None = None
    raw: str | None = None  # nur bei Fehlern (Debug)
    truncated: bool = False  # Antwort war abgeschnitten und wurde repariert (Teilverlust)

    def as_dict(self) -> dict:
        return {"entities": self.entities, "relations": self.relations}


def _clean_str(value, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _normalize_type(value) -> str:
    key = str(value or "").strip().upper()
    return _TYPE_ALIASES.get(key, "OTHER")


def _repair_truncated(text: str) -> dict | None:
    """Abgeschnittenes JSON retten: am letzten vollständigen Objekt kappen und schließen."""
    cut = len(text)
    for _ in range(20):
        cut = text.rfind("}", 0, cut)
        if cut <= 0:
            return None
        head = text[: cut + 1]
        for suffix in ("]}", '],"relations":[]}'):
            try:
                data = json.loads(head + suffix)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "entities" in data:
                return data
    return None


def parse_extraction(raw: str) -> Extraction:
    """Tolerantes Parsen der Modellantwort; wirft ValueError, wenn kein JSON-Objekt darin ist."""
    text = raw.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("kein JSON-Objekt in der Antwort")
    text = text[start:]
    truncated = False
    try:
        data = json.loads(text[: text.rfind("}") + 1])
    except json.JSONDecodeError as exc:
        data = _repair_truncated(text)
        if data is None:
            raise ValueError(f"ungültiges JSON: {exc.msg}") from exc
        truncated = True
    if not isinstance(data, dict):
        raise ValueError("JSON ist kein Objekt")

    entities: list[dict] = []
    seen: set[str] = set()
    for ent in data.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        name = _clean_str(ent.get("name"), 80)
        if not name or name.casefold() in seen:
            continue
        seen.add(name.casefold())
        entities.append({
            "name": name,
            "type": _normalize_type(ent.get("type")),
            "desc": _clean_str(ent.get("desc") or ent.get("description"), MAX_DESC_CHARS),
        })
        if len(entities) == MAX_ENTITIES:
            break

    relations: list[dict] = []
    for rel in data.get("relations") or []:
        if not isinstance(rel, dict):
            continue
        source = _clean_str(rel.get("source"), 80)
        target = _clean_str(rel.get("target"), 80)
        # Nur Relationen zwischen extrahierten Entities (sonst hängen Kanten im Leeren).
        if not source or not target or source.casefold() == target.casefold():
            continue
        if source.casefold() not in seen or target.casefold() not in seen:
            continue
        relations.append({
            "source": source,
            "target": target,
            "rel": _clean_str(rel.get("rel") or rel.get("relation"), 40),
        })
        if len(relations) == MAX_RELATIONS:
            break
    return Extraction(entities=entities, relations=relations, truncated=truncated)


def build_prompt(*, title: str | None, chapter: str | None, context: str | None, text: str) -> str:
    context_line = f"Context: {context}\n" if context else ""
    return EXTRACT_PROMPT.format(
        title=title or "?",
        chapter=chapter or "beginning of the book",
        context_line=context_line,
        text=text[:2400],
    )


async def extract_chunk(
    ollama: OllamaClient,
    model: str,
    *,
    title: str | None,
    chapter: str | None,
    context: str | None,
    text: str,
) -> Extraction:
    """Ein Chunk -> Extraction. Ein Retry bei Parse-Fehler; bei Ollama-Fehlern leer + error."""
    prompt = build_prompt(title=title, chapter=chapter, context=context, text=text)
    messages = [{"role": "user", "content": prompt}]
    raw = ""
    error: str | None = None
    for attempt in range(2):
        try:
            raw = await ollama.complete(
                messages, model, temperature=0.0, max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # Ollama nicht erreichbar / Timeout
            return Extraction(error=f"ollama: {exc}", raw=None)
        try:
            return parse_extraction(raw)
        except ValueError as exc:
            error = f"parse: {exc}"
            if attempt == 0:
                messages = [{"role": "user", "content": prompt + "\nReturn ONLY valid JSON."}]
    return Extraction(error=error, raw=raw[:2000])
