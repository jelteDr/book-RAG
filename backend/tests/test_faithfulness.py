"""Unit-Tests für den gefensterten Faithfulness-Check — Fake-Tokenizer/-NLI, kein HF-Download.

Aufruf (aus backend/): uv run pytest tests -q
"""

from __future__ import annotations

from app.services.faithfulness_service import FaithfulnessService


class FakeTokenizer:
    """Ein Wort = ein Token; `decode` setzt die Wörter wieder zusammen."""

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        self.vocab = getattr(self, "vocab", {})
        ids = [self.vocab.setdefault(w, len(self.vocab)) for w in text.split()]
        return {"input_ids": ids}

    def decode(self, ids: list[int]) -> str:
        inv = {i: w for w, i in self.vocab.items()}
        return " ".join(inv[i] for i in ids)


def _svc() -> FaithfulnessService:
    svc = FaithfulnessService("fake", enabled=True)
    svc._tok = FakeTokenizer()
    svc._model = object()  # _ensure_model() gilt damit als geladen
    return svc


def _text(n: int, tail: str = "") -> str:
    return " ".join(f"w{i}" for i in range(n)) + (f" {tail}" if tail else "")


def test_kurze_praemisse_bleibt_ein_fenster():
    svc = _svc()
    premise = _text(FaithfulnessService.WINDOW)
    assert svc._windows(premise) == [premise]


def test_fenster_ueberlappen_und_decken_ende_ab():
    svc = _svc()
    n = 1300  # ~ Small-to-Big-Passage
    wins = svc._windows(_text(n))
    tokens = [w.split() for w in wins]
    assert all(len(t) <= FaithfulnessService.WINDOW for t in tokens)
    # Schritt STRIDE → 50 % Überlappung zwischen Nachbarfenstern
    assert tokens[1][0] == f"w{FaithfulnessService.STRIDE}"
    assert tokens[0][FaithfulnessService.STRIDE:] == tokens[1][: FaithfulnessService.WINDOW - FaithfulnessService.STRIDE]
    # Letztes Token der Prämisse liegt in einem Fenster
    assert tokens[-1][-1] == f"w{n - 1}"


def test_belegstelle_am_ende_wird_gefunden():
    """Vorher abgeschnitten → ⚠; mit Fenstern liegt der Beleg im letzten Fenster."""
    svc = _svc()
    # Fake-NLI: Entailment nur, wenn das Fenster den Beleg enthält.
    svc._entail_prob = lambda premise, hyp: 0.9 if "BELEG" in premise else 0.1
    sources = [
        {"marker": 1, "text": _text(1000, "BELEG")},
        {"marker": 2, "text": _text(1000)},
    ]
    out = svc.check("Die Aussage stimmt [1]. Eine andere [2].", sources)
    assert out[0]["supported"] is True and out[0]["support_score"] == 0.9
    assert out[1]["supported"] is False


def test_deaktiviert_laesst_quellen_unveraendert():
    svc = FaithfulnessService("fake", enabled=False)
    sources = [{"marker": 1, "text": "x"}]
    assert svc.check("a [1].", sources) == [{"marker": 1, "text": "x"}]
