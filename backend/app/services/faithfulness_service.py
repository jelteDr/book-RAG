"""Optionaler Faithfulness-Check: markiert ungestützte Zitate via NLI (Entailment).

Für jede zitierte Quelle wird geprüft, ob ihr Chunk die Antwort-Aussage(n) mit dem
zugehörigen Marker STÜTZT (Entailment via mDeBERTa-xnli). Opt-in, lazy geladen; ohne
transformers/torch (reranker-Gruppe) still deaktiviert.
"""

from __future__ import annotations

import re


class FaithfulnessService:
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

    def check(self, answer: str, sources: list[dict]) -> list[dict]:
        """Ergänzt jede Quelle um `supported: bool` (True, wenn sie ihre Aussage stützt)."""
        if not self._enabled or not sources or not self._ensure_model():
            return sources

        import torch

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        for src in sources:
            marker = src.get("marker")
            # Sätze, die genau diesen Marker zitieren (Fallback: ganze Antwort).
            cited = [s for s in sentences if f"[{marker}]" in s] or [answer]
            best = 0.0
            for sent in cited:
                inputs = self._tok(
                    src.get("text", ""), sent, truncation=True, max_length=512, return_tensors="pt"
                )
                with torch.no_grad():
                    probs = torch.softmax(self._model(**inputs).logits, dim=-1)[0]
                best = max(best, float(probs[self._ent_idx]))
            src["supported"] = best >= self._threshold
            src["support_score"] = round(best, 3)
        return sources
