"""Graph-Build: Extraktions-Zeilen (extract.jsonl) -> KnowledgeGraph.

Normalisierung ist regelbasiert und konservativ:
  - Schlüssel = "<TYP>:<normalisierter name>" (casefold, Akzente weg, Titel/Artikel
    wie "Count", "Dr.", "the" am Anfang entfernt) -> "Count Dracula" == "Dracula".
  - Alias-Merge NUR für PERSON/ORGANISATION: ein kürzerer Name wird in einen längeren
    gemergt, wenn seine Tokens ein zusammenhängendes Präfix/Suffix von GENAU EINEM
    längeren Namen gleichen Typs sind ("Lucy" -> "Lucy Westenra"). "Harker" bleibt
    ein eigener Knoten, weil Jonathan UND Mina Harker existieren (ambig -> Report).
    Orte/Objekte werden nicht gemergt ("London" ist nicht "London Bridge").
  - Gattungsbegriffe ("inn", "wolves", "train") extrahiert das 7B-Modell trotz Prompt
    als Entities. Namen ohne Großbuchstaben werden deshalb standardmäßig verworfen
    (englische Eigennamen sind großgeschrieben; `keep_common_nouns=True` behält sie).
  - Hubs (Dracula in ~40 % der Chunks) werden nicht gekappt, sondern über
    idf = ln(N / n_mentions) im Retrieval gedämpft.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import networkx as nx

from app.graph.store import KnowledgeGraph

_TITLES = {
    "the", "a", "an", "count", "countess", "dr", "doctor", "mr", "mrs", "ms", "miss",
    "professor", "prof", "lord", "sir", "lady", "madam", "madame", "captain", "mister",
}
_MERGE_TYPES = {"PERSON", "ORGANISATION"}
MAX_DESCS = 3
MAX_DESC_TOTAL = 300
MAX_REL_EXAMPLES = 5


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"'s\b", "", text)  # Possessiv: "landlord's wife" != "landlord"
    text = re.sub(r"[^a-z0-9\s]", " ", text.casefold())
    tokens = text.split()
    while tokens and tokens[0] in _TITLES:
        tokens.pop(0)
    return " ".join(tokens)


def entity_key(name: str, entity_type: str) -> str:
    return f"{entity_type}:{normalize_name(name)}"


def merge_aliases(keys: list[str]) -> tuple[dict[str, str], list[tuple[str, list[str]]]]:
    """Alias-Regel (s. Modul-Docstring). Liefert (mapping kurz->lang, ambige Fälle)."""
    by_type: dict[str, list[tuple[str, list[str]]]] = {}
    for key in keys:
        etype, _, norm = key.partition(":")
        if etype in _MERGE_TYPES and norm:
            by_type.setdefault(etype, []).append((key, norm.split()))

    mapping: dict[str, str] = {}
    ambiguous: list[tuple[str, list[str]]] = []
    for entries in by_type.values():
        for key, tokens in entries:
            n = len(tokens)
            candidates = [
                other for other, otoks in entries
                if len(otoks) > n and (otoks[:n] == tokens or otoks[-n:] == tokens)
            ]
            if len(candidates) == 1:
                mapping[key] = candidates[0]
            elif len(candidates) > 1:
                ambiguous.append((key, sorted(candidates)))

    # Ketten auflösen ("lucy" -> "lucy westenra" -> ...), Zyklen sind konstruktiv unmöglich
    # (Ziel ist immer länger), trotzdem begrenzen.
    for key in list(mapping):
        target, hops = mapping[key], 0
        while target in mapping and hops < 10:
            target, hops = mapping[target], hops + 1
        mapping[key] = target
    return mapping, ambiguous


@dataclass
class BuildReport:
    n_rows: int = 0
    n_error_rows: int = 0
    n_entities_raw: int = 0
    n_relations_raw: int = 0
    n_relations_dropped: int = 0  # Endpunkt nicht auflösbar / Self-Loop
    n_common_nouns_dropped: int = 0  # Entities ohne Großbuchstaben (Gattungsbegriffe)
    merges: list[tuple[str, str]] = field(default_factory=list)
    ambiguous: list[tuple[str, list[str]]] = field(default_factory=list)


def build_graph(
    rows: list[dict], group_id: str, n_chunks: int, model: str = "",
    *, keep_common_nouns: bool = False,
) -> tuple[KnowledgeGraph, BuildReport]:
    report = BuildReport(n_rows=len(rows))
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str], dict] = {}
    book_ids: set[str] = set()

    def node(key: str, etype: str) -> dict:
        return nodes.setdefault(
            key, {"type": etype, "surface": Counter(), "descs": [], "chunks": set()}
        )

    for row in rows:
        if row.get("error"):
            report.n_error_rows += 1
        chunk_key = row["key"]
        book_ids.add(row.get("book_id") or chunk_key.split(":")[0])
        name_to_key: dict[str, str] = {}
        for ent in row.get("entities") or []:
            report.n_entities_raw += 1
            key = entity_key(ent["name"], ent["type"])
            if key.endswith(":"):  # Name bestand nur aus Titeln/Interpunktion
                continue
            name_to_key[ent["name"].casefold()] = key
            data = node(key, ent["type"])
            data["surface"][ent["name"]] += 1
            data["chunks"].add(chunk_key)
            desc = ent.get("desc") or ""
            if desc and desc not in data["descs"] and len(data["descs"]) < MAX_DESCS:
                data["descs"].append(desc)
        for rel in row.get("relations") or []:
            report.n_relations_raw += 1
            u = name_to_key.get(rel["source"].casefold())
            v = name_to_key.get(rel["target"].casefold())
            if not u or not v or u == v:
                report.n_relations_dropped += 1
                continue
            edge = edges.setdefault(tuple(sorted((u, v))), {"chunks": set(), "rels": []})
            edge["chunks"].add(chunk_key)
            if len(edge["rels"]) < MAX_REL_EXAMPLES:
                edge["rels"].append({"rel": rel.get("rel", ""), "chunk": chunk_key})

    # Gattungsbegriffe verwerfen (häufigste Oberflächenform ohne Großbuchstaben).
    if not keep_common_nouns:
        common = [k for k, d in nodes.items()
                  if not any(ch.isupper() for ch in d["surface"].most_common(1)[0][0])]
        for k in common:
            nodes.pop(k)
        report.n_common_nouns_dropped = len(common)
        dropped_edges = [e for e in edges if e[0] in common or e[1] in common]
        for e in dropped_edges:
            report.n_relations_dropped += len(edges.pop(e)["chunks"])

    # Alias-Merge anwenden.
    mapping, report.ambiguous = merge_aliases(sorted(nodes))
    report.merges = sorted(mapping.items())
    for src, dst in mapping.items():
        target = nodes[dst]
        source = nodes.pop(src)
        target["surface"].update(source["surface"])
        target["chunks"] |= source["chunks"]
        for desc in source["descs"]:
            if desc not in target["descs"] and len(target["descs"]) < MAX_DESCS:
                target["descs"].append(desc)
    merged_edges: dict[tuple[str, str], dict] = {}
    for (u, v), data in edges.items():
        u2, v2 = mapping.get(u, u), mapping.get(v, v)
        if u2 == v2:
            report.n_relations_dropped += len(data["chunks"])
            continue
        target = merged_edges.setdefault(tuple(sorted((u2, v2))), {"chunks": set(), "rels": []})
        target["chunks"] |= data["chunks"]
        target["rels"] = (target["rels"] + data["rels"])[:MAX_REL_EXAMPLES]

    # networkx-Graph in sortierter Reihenfolge (deterministisch für Louvain in Stufe 2).
    graph = nx.Graph()
    mentions: dict[str, list[str]] = {}
    chunk_entities: dict[str, list[str]] = {}
    for key in sorted(nodes):
        data = nodes[key]
        name, _ = data["surface"].most_common(1)[0]
        aliases = sorted(s for s in data["surface"] if s != name)
        n_mentions = len(data["chunks"])
        graph.add_node(
            key,
            name=name,
            type=data["type"],
            aliases=aliases,
            desc="; ".join(data["descs"])[:MAX_DESC_TOTAL],
            n_mentions=n_mentions,
            idf=round(math.log(max(n_chunks, 1) / n_mentions), 4) if n_mentions else 0.0,
        )
        mentions[key] = sorted(data["chunks"])
        for chunk_key in mentions[key]:
            chunk_entities.setdefault(chunk_key, []).append(key)
    for (u, v) in sorted(merged_edges):
        data = merged_edges[(u, v)]
        graph.add_edge(u, v, weight=len(data["chunks"]), rels=data["rels"])

    kg = KnowledgeGraph(
        group_id=group_id, book_ids=sorted(book_ids), n_chunks=n_chunks, graph=graph,
        mentions=mentions, chunk_entities=chunk_entities, model=model,
    )
    return kg, report


def stats(kg: KnowledgeGraph, top: int = 10) -> dict:
    g = kg.graph
    by_type = Counter(d["type"] for _, d in g.nodes(data=True))
    degree = sorted(g.degree(weight="weight"), key=lambda x: x[1], reverse=True)[:top]
    by_mentions = sorted(g.nodes(data="n_mentions"), key=lambda x: x[1], reverse=True)[:top]
    return {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "by_type": dict(by_type),
        "isolated": sum(1 for n in g.nodes if g.degree(n) == 0),
        "top_degree": [(kg.name(n), int(d)) for n, d in degree],
        "top_mentions": [(kg.name(n), int(m)) for n, m in by_mentions],
        "chunks_with_entities": len(kg.chunk_entities),
    }
