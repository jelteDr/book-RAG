"""Optionaler Faithfulness-Check: markiert ungestützte Zitate via NLI (Entailment).

Für jede zitierte Quelle wird geprüft, ob ihr Chunk die Antwort-Aussage(n) mit dem
zugehörigen Marker STÜTZT (Entailment via mDeBERTa-xnli). Opt-in, lazy geladen; ohne
transformers/torch (reranker-Gruppe) still deaktiviert.

Lange Chunks werden in überlappende Token-Fenster geteilt (Maximum über die Fenster), statt
sie bei 512 Token abzuschneiden — gleiches Verfahren wie `eval/answer_eval.py` (`Faithfulness`).
"""

from __future__ import annotations

import re


class FaithfulnessService:
    # Fenster für lange Prämissen: Chunks haben im Median ~630 Token (Small-to-Big-Passagen
    # ~1300), das Modell nimmt Prämisse + Satz nur bis 512 Token. Bisher wurde still
    # abgeschnitten (98-100 % der Passagen betroffen; s. eval/RESULTS.md, Abschnitt Ablation):
    # stand die Belegstelle hinten, galt ein korrektes Zitat als ungestützt (⚠ im Frontend) und
    # der CRAG-Knoten `check` regenerierte auf ein falsches Signal. 380 Token je Fenster lassen
    # Platz für den Satz, 50 % Überlappung, damit keine Belegstelle an einer Grenze zerfällt.
    WINDOW, STRIDE = 380, 190

    def __init__(self, model_name: str, enabled: bool, threshold: float = 0.5) -> None:
        self._model_name = model_name
        self._enabled = enabled
        self._threshold = threshold
        self._model = None
        self._tok = None
        self._ent_idx = 0

    @property
    def active(self) -> bool:
        return self._enabled

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            return False
        self._tok = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        self._model.eval()
        self._ent_idx = next(
            i for i, lab in self._model.config.id2label.items() if lab.lower().startswith("entail")
        )
        return True

    def _windows(self, premise: str) -> list[str]:
        """Teilt die Prämisse in überlappende Fenster à WINDOW Token (Schritt STRIDE)."""
        ids = self._tok(premise, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.WINDOW:
            return [premise]
        starts = range(0, max(len(ids) - self.WINDOW, 0) + self.STRIDE, self.STRIDE)
        return [self._tok.decode(ids[s : s + self.WINDOW]) for s in starts]

    def _entail_prob(self, premise: str, hypothesis: str) -> float:
        import torch

        inputs = self._tok(premise, hypothesis, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(self._model(**inputs).logits, dim=-1)[0]
        return float(probs[self._ent_idx])

    def _entail_windowed(self, premise: str, hypothesis: str) -> float:
        return max(self._entail_prob(w, hypothesis) for w in self._windows(premise))

    def check(self, answer: str, sources: list[dict]) -> list[dict]:
        """Ergänzt jede Quelle um `supported: bool` (True, wenn sie ihre Aussage stützt)."""
        if not self._enabled or not sources or not self._ensure_model():
            return sources

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        for src in sources:
            marker = src.get("marker")
            # Sätze, die genau diesen Marker zitieren (Fallback: ganze Antwort).
            cited = [s for s in sentences if f"[{marker}]" in s] or [answer]
            premise = src.get("text", "")
            best = max(self._entail_windowed(premise, sent) for sent in cited)
            src["supported"] = best >= self._threshold
            src["support_score"] = round(best, 3)
        return sources
