"""Community-Berichte (Microsoft-GraphRAG-Stil) per LLM — resumierbarer Checkpoint.

Je Community bekommt das Modell die Entities (Name, Typ, Beschreibung), die internen
Relationen und 2-3 repräsentative Buchauszüge und schreibt einen englischen Bericht
(title / summary / findings). Englisch, weil der Quelltext englisch ist — bge-m3 und
das NLI-Modell sind multilingual, die Antwort an den Nutzer bleibt deutsch.

Jeder Bericht ist ein Buchtext-Derivat -> data/graph/<group>/communities.jsonl (gitignored).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from app.clients.ollama_client import OllamaClient
from app.graph.communities import Community, Partition
from app.graph.store import KnowledgeGraph

MAX_ENTITIES = 40
MAX_RELATIONS = 60
EXCERPT_CHARS = 700
MAX_TOKENS = 700
PROMPT_BUDGET_CHARS = 18000  # ~4,5k Token Eingabe; passt in num_ctx 16384 mit Reserve

REPORT_SYSTEM = (
    'You write an analytical report about a group of related entities from the novel "{title}". '
    "Stay strictly grounded in the given entity descriptions, relationships and text excerpts; "
    "do not invent facts or events. Output JSON only."
)

REPORT_USER = (
    "ENTITIES:\n{entities}\n\n"
    "RELATIONSHIPS:\n{relations}\n\n"
    "TEXT EXCERPTS:\n{excerpts}\n\n"
    "Write a report as compact JSON (one line): "
    '{{"title":"...","summary":"...","findings":[{{"summary":"...","explanation":"..."}}]}}\n'
    "- title: a short descriptive name for this group of entities.\n"
    "- summary: 3-5 sentences on how these entities relate and what happens between them.\n"
    "- findings: 3-6 key findings; each explanation names the entities involved and stays "
    "within the given material.\n"
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _clean(value, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def build_report_messages(
    kg: KnowledgeGraph, community: Community, chunks_by_key: dict[str, dict], *, title: str
) -> list[dict]:
    ents = community.entities[:MAX_ENTITIES]
    ent_lines = []
    for eid in ents:
        node = kg.graph.nodes[eid]
        desc = f": {node['desc']}" if node.get("desc") else ""
        ent_lines.append(f"- {node['name']} ({node['type']}){desc}")

    member = set(community.entities)
    edges = [
        (u, v, d) for u, v, d in kg.graph.edges(ents, data=True) if u in member and v in member
    ]
    edges.sort(key=lambda e: (-e[2].get("weight", 1), e[0], e[1]))
    rel_lines = []
    for u, v, d in edges[:MAX_RELATIONS]:
        labels = sorted({r.get("rel", "") for r in d.get("rels", []) if r.get("rel")})[:2]
        label = ", ".join(labels) or "related to"
        rel_lines.append(f"- {kg.name(u)} -- {label} -- {kg.name(v)} (in {d.get('weight', 1)} passages)")

    excerpts = []
    for key in community.chunk_keys:
        pl = chunks_by_key.get(key)
        if pl:
            excerpts.append(f"[{pl.get('chapter') or '?'}] {pl.get('text', '')[:EXCERPT_CHARS]}")

    # Budget: erst Auszüge, dann Relationen, dann Entities kürzen.
    def render() -> str:
        return REPORT_USER.format(
            entities="\n".join(ent_lines) or "-", relations="\n".join(rel_lines) or "-",
            excerpts="\n\n".join(excerpts) or "-",
        )

    user = render()
    while len(user) > PROMPT_BUDGET_CHARS and (excerpts or len(rel_lines) > 10 or len(ent_lines) > 10):
        if excerpts:
            excerpts.pop()
        elif len(rel_lines) > 10:
            rel_lines = rel_lines[: max(10, len(rel_lines) // 2)]
        else:
            ent_lines = ent_lines[: max(10, len(ent_lines) // 2)]
        user = render()
    return [
        {"role": "system", "content": REPORT_SYSTEM.format(title=title)},
        {"role": "user", "content": user},
    ]


def parse_report(raw: str) -> dict:
    """Tolerant: Fences/Prosa um das JSON; wirft ValueError ohne verwertbares Objekt."""
    text = raw.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("kein JSON-Objekt")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"ungültiges JSON: {exc.msg}") from exc
    if not isinstance(data, dict) or not data.get("summary"):
        raise ValueError("kein summary-Feld")
    findings = []
    for f in data.get("findings") or []:
        if isinstance(f, dict) and (f.get("summary") or f.get("explanation")):
            findings.append({"summary": _clean(f.get("summary"), 300),
                             "explanation": _clean(f.get("explanation"), 800)})
        elif isinstance(f, str) and f.strip():
            findings.append({"summary": _clean(f, 300), "explanation": ""})
    return {"title": _clean(data.get("title"), 120) or "Untitled community",
            "summary": _clean(data.get("summary"), 2000), "findings": findings[:6]}


def report_to_text(report: dict) -> str:
    """Der Text, den Prompt, Zitat-Chip und NLI sehen."""
    lines = [report.get("title", ""), "", report.get("summary", "")]
    if report.get("findings"):
        lines += ["", "Key findings:"]
        for f in report["findings"]:
            expl = f" — {f['explanation']}" if f.get("explanation") else ""
            lines.append(f"- {f['summary']}{expl}")
    return "\n".join(lines).strip()


def embedding_text(report: dict) -> str:
    """Kompakter Text fürs Embedding (Titel + Summary + Finding-Summaries)."""
    parts = [report.get("title", ""), report.get("summary", "")]
    parts += [f.get("summary", "") for f in report.get("findings", [])]
    return "\n".join(p for p in parts if p)


async def generate_report(ollama: OllamaClient, model: str, messages: list[dict]) -> tuple[dict, bool]:
    """(Bericht, raw_fallback). Ein Retry bei Parse-Fehler; danach Rohtext als summary."""
    raw = ""
    for attempt in range(2):
        raw = await ollama.complete(
            messages, model, temperature=0.0, max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        try:
            return parse_report(raw), False
        except ValueError:
            if attempt == 0:
                messages = messages[:1] + [
                    {"role": "user", "content": messages[1]["content"] + "\nReturn ONLY valid JSON."}
                ]
    return {"title": "Untitled community", "summary": _clean(raw, 2000), "findings": []}, True


async def generate_reports(
    kg: KnowledgeGraph,
    partition: Partition,
    chunks_by_key: dict[str, dict],
    *,
    ollama: OllamaClient,
    model: str,
    path: Path,
    title: str,
    existing: dict[str, dict] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, dict]:
    """Berichte für alle Communities ohne Checkpoint-Zeile; append+flush je Bericht."""
    reports = dict(existing or {})
    todo = [c for c in partition.communities if c.id not in reports]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for i, community in enumerate(todo, start=1):
            messages = build_report_messages(kg, community, chunks_by_key, title=title)
            report, fallback = await generate_report(ollama, model, messages)
            row = {
                "community_id": community.id, "level": community.level, "size": community.size,
                **report, "text": report_to_text(report),
                "entities": [kg.name(e) for e in community.entities[:10]],
                "chunk_keys": community.chunk_keys, "book_title": title, "model": model,
                "prompt_chars": len(messages[1]["content"]), "raw_fallback": fallback,
            }
            reports[community.id] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if on_progress:
                on_progress(i, len(todo))
    return reports
