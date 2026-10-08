"""Unit-Tests für den Corrective-RAG-Graphen (app.flow) — Fake-Ollama/-Qdrant, kein Netz.

Aufruf (aus backend/): uv run pytest tests -q
"""

from __future__ import annotations

import asyncio
import json

from qdrant_client import models

from app.flow.crag import CragFlow
from app.flow.nodes import REFUSAL, CragNodes, parse_grade


def _pt(i: int, score: float = 0.5) -> models.ScoredPoint:
    return models.ScoredPoint(
        id=i, version=0, score=score,
        payload={"text": f"Passage {i}", "book_id": "b", "char_start": i * 1000, "char_end": i * 1000 + 500},
    )


class FakeOllama:
    """`grades`: Antworten des Graders der Reihe nach; `answers`: Antworten der Generierung."""

    def __init__(self, grades: list[str], answers: list[str] | None = None) -> None:
        self.grades = list(grades)
        self.answers = list(answers or ["Antwort [1]."])
        self.rewrites = 0

    async def embed(self, texts, model):
        return [[1.0, 0.0] for _ in texts]

    async def complete(self, messages, model, temperature=0.0, max_tokens=None, response_format=None):
        if response_format:  # Grader
            return self.grades.pop(0)
        self.rewrites += 1
        return f"neue Anfrage {self.rewrites}"

    async def chat_stream(self, messages, model, temperature=0.7):
        text = self.answers.pop(0)
        for part in (text[: len(text) // 2], text[len(text) // 2 :]):
            yield "data: " + json.dumps({"choices": [{"delta": {"content": part}}]})
        yield "data: " + json.dumps({"choices": [], "usage": {"completion_tokens": 4}})
        yield "data: [DONE]"


class FakeVectors:
    """Liefert je Suchlauf die nächste Trefferliste."""

    def __init__(self, batches: list[list[models.ScoredPoint]]) -> None:
        self.batches = list(batches)
        self.calls = 0

    async def search(self, vec, limit, query_filter):
        self.calls += 1
        return self.batches.pop(0) if self.batches else []


class FakeFaith:
    """Markiert die ersten `n_bad` Prüfungen als komplett unbelegt."""

    active = True

    def __init__(self, n_bad: int) -> None:
        self.n_bad = n_bad

    def check(self, answer, sources):
        ok = self.n_bad <= 0
        self.n_bad -= 1
        return [{**s, "supported": ok} for s in sources]


def _flow(ollama, vectors, faith=None, max_retries=1, max_regen=1) -> CragFlow:
    nodes = CragNodes(ollama, vectors, embed_model="e", min_relevant=2, unsupported_max=0.5,
                      faithfulness=faith)
    return CragFlow(nodes, enabled=True, max_retries=max_retries, max_regen=max_regen)


def _run(flow: CragFlow, **kw) -> list[tuple[str, dict]]:
    async def go():
        return [e async for e in flow.run("Frage?", model="m", top_k=4, **kw)]
    return asyncio.run(go())


def test_parse_grade():
    assert parse_grade('{"relevant": [2, 1, 2, 9, "3"]}', 4) == [2, 1, 3]
    assert parse_grade("Sure: ```{\"relevant\": []}```", 4) == []
    assert parse_grade("kaputt", 4) is None
    assert parse_grade('{"relevant": "1"}', 4) is None


def test_enough_relevant_generates_directly():
    ollama = FakeOllama(['{"relevant":[1,3]}'])
    vectors = FakeVectors([[_pt(1, .9), _pt(2, .8), _pt(3, .7)]])
    events = _run(_flow(ollama, vectors))
    names = [e for e, _ in events]
    assert names[-1] == "done" and "token" in names and "reset" not in names
    done = events[-1][1]
    assert done["crag"]["attempts"] == 1 and done["crag"]["n_relevant"] == 2
    assert done["answer"] == "Antwort [1]."
    # Nur relevante Passagen im Prompt -> [1] zeigt auf Passage 1, [2] wäre Passage 3.
    assert done["sources"][0]["text"] == "Passage 1"
    assert [t["node"] for t in done["crag"]["trace"]] == ["retrieve", "grade", "generate", "check"]


def test_rewrite_then_refuse_when_nothing_relevant():
    ollama = FakeOllama(['{"relevant":[]}', '{"relevant":[]}'])
    vectors = FakeVectors([[_pt(1)], [_pt(2)]])
    events = _run(_flow(ollama, vectors, max_retries=1))
    done = events[-1][1]
    assert vectors.calls == 2 and ollama.rewrites == 1
    assert done["crag"]["refused"] is True and done["answer"] == REFUSAL
    assert done["crag"]["queries"] == ["Frage?", "neue Anfrage 1"]
    assert events[0] == ("token", {"content": REFUSAL})


def test_retry_collects_relevant_across_attempts_and_skips_graded():
    ollama = FakeOllama(['{"relevant":[1]}', '{"relevant":[1]}'])
    # Zweiter Lauf liefert Passage 1 erneut -> darf nicht nochmal bewertet werden.
    vectors = FakeVectors([[_pt(1, .9), _pt(2, .8)], [_pt(1, .9), _pt(5, .6)]])
    done = _run(_flow(ollama, vectors))[-1][1]
    assert done["crag"]["attempts"] == 2
    assert done["crag"]["n_graded"] == 3  # 1, 2, 5
    assert done["crag"]["n_relevant"] == 2  # 1, 5


def test_retries_exhausted_with_one_relevant_still_answers():
    ollama = FakeOllama(['{"relevant":[1]}', '{"relevant":[]}'])
    vectors = FakeVectors([[_pt(1)], [_pt(2)]])
    done = _run(_flow(ollama, vectors))[-1][1]
    assert done["crag"]["refused"] is False and done["crag"]["n_relevant"] == 1


def test_broken_grader_fails_open():
    ollama = FakeOllama(["kein json"])
    vectors = FakeVectors([[_pt(1), _pt(2), _pt(3)]])
    done = _run(_flow(ollama, vectors))[-1][1]
    assert done["crag"]["n_relevant"] == 3
    assert done["crag"]["trace"][1]["fail_open"] is True


def test_unsupported_answer_is_regenerated_once_with_reset():
    ollama = FakeOllama(['{"relevant":[1,2]}'], ["Erst [1].", "Dann [1].", "nie"])
    vectors = FakeVectors([[_pt(1), _pt(2)]])
    events = _run(_flow(ollama, vectors, faith=FakeFaith(n_bad=5), max_regen=1))
    names = [e for e, _ in events]
    assert names.count("reset") == 1
    assert names.index("reset") < len(names) - 1
    done = events[-1][1]
    assert done["crag"]["regenerated"] is True and done["answer"] == "Dann [1]."
    assert [t["node"] for t in done["crag"]["trace"]].count("generate") == 2  # Grenze greift


def test_no_regen_when_disabled():
    ollama = FakeOllama(['{"relevant":[1,2]}'])
    vectors = FakeVectors([[_pt(1), _pt(2)]])
    events = _run(_flow(ollama, vectors, faith=FakeFaith(n_bad=5), max_regen=0))
    assert "reset" not in [e for e, _ in events]
