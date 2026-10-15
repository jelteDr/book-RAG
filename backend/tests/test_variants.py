"""Antwort-Varianten: Verlauf (nur ausgewählte Antworten) und Gruppierung für die Anzeige."""

from __future__ import annotations

from types import SimpleNamespace

from app.services.conversation_view import group_turns, history_from


def _m(id, role, content, parent_id=None, selected=True):
    return SimpleNamespace(id=id, role=role, content=content, model="m", sources=[],
                           parent_id=parent_id, selected=selected)


ROWS = [
    _m(1, "user", "Frage 1"),
    _m(2, "assistant", "Antwort 1a", parent_id=1, selected=False),
    _m(3, "assistant", "Antwort 1b", parent_id=1, selected=True),
    _m(4, "user", "Frage 2"),
    _m(5, "assistant", "Antwort 2", parent_id=4),
]


def test_history_contains_only_selected_variant():
    assert [h["content"] for h in history_from(ROWS)] == ["Frage 1", "Antwort 1b", "Frage 2", "Antwort 2"]


def test_group_turns_collects_variants_and_selection():
    turns = group_turns(ROWS)
    assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant"]
    assert [v["content"] for v in turns[1]["variants"]] == ["Antwort 1a", "Antwort 1b"]
    assert turns[1]["selected"] == 1
    assert len(turns[3]["variants"]) == 1 and turns[3]["selected"] == 0


def test_legacy_rows_without_parent_are_single_variant_turns():
    legacy = [_m(1, "user", "F"), _m(2, "assistant", "A"), _m(3, "user", "G"), _m(4, "assistant", "B")]
    turns = group_turns(legacy)
    assert [len(t.get("variants", [])) for t in turns] == [0, 1, 0, 1]
    assert [h["content"] for h in history_from(legacy)] == ["F", "A", "G", "B"]
