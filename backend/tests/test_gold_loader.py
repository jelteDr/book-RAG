"""Gold-Set aus mehreren Dateien/Büchern laden (eval/retrieval_eval.load_gold)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from retrieval_eval import load_gold  # noqa: E402


def _write(path: Path, items: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(i) for i in items) + "\n")
    return str(path)


def test_multiple_files_and_group_fallback(tmp_path):
    a = _write(tmp_path / "a.jsonl", [{"id": "d01", "question": "?"}])
    b = _write(tmp_path / "b.jsonl", [{"id": "f01", "question": "?", "group": "Frankenstein"}])
    items = load_gold(f"{a}, {b}", "Horror")
    assert [(i["id"], i["group"]) for i in items] == [("d01", "Horror"), ("f01", "Frankenstein")]


def test_duplicate_ids_are_rejected(tmp_path):
    a = _write(tmp_path / "a.jsonl", [{"id": "x1", "question": "?"}])
    with pytest.raises(ValueError, match="doppelt"):
        load_gold(f"{a},{a}", "Horror")
