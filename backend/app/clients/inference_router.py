"""Verteilt Inferenz-Aufrufe auf Ollama und (optional) vllm-metal.

Hat dieselbe Schnittstelle wie `OllamaClient`, damit RagService, Graph, CRAG und
Eval-Skripte unverändert bleiben. Regeln:

- **Chat/Completion** gehen an vllm-metal, wenn das angefragte Modell dort bedient
  wird — sonst an Ollama. Das Modell-Dropdown zeigt beide, die Auswahl IST der Schalter.
- **Embeddings** gehen IMMER an Ollama: der Qdrant-Index wurde mit Ollamas bge-m3
  (GGUF) gebaut; andere Gewichte liefern leicht andere Vektoren.
- Ist vllm-metal aus oder nicht erreichbar, verhält sich der Router wie Ollama allein.
"""

import time
from collections.abc import AsyncIterator

from app.clients.ollama_client import OllamaClient
from app.clients.vllm_client import VLLMClient

_CACHE_TTL_S = 30.0  # so lange gilt die vLLM-Modellliste, bevor sie neu abgefragt wird


class InferenceRouter:
    def __init__(self, ollama: OllamaClient, vllm: VLLMClient | None = None) -> None:
        self.ollama = ollama
        self.vllm = vllm
        self._vllm_models: set[str] = set()
        self._vllm_checked = 0.0

    async def aclose(self) -> None:
        await self.ollama.aclose()
        if self.vllm:
            await self.vllm.aclose()

    async def vllm_models(self) -> set[str]:
        """Von vllm-metal bediente Modelle (kurz gecacht; leer, wenn aus/unerreichbar)."""
        if self.vllm is None:
            return set()
        if time.monotonic() - self._vllm_checked > _CACHE_TTL_S:
            try:
                self._vllm_models = {m["name"] for m in await self.vllm.list_models()}
            except Exception:
                self._vllm_models = set()
            self._vllm_checked = time.monotonic()
        return self._vllm_models

    async def backend_for(self, model: str) -> OllamaClient | VLLMClient:
        if self.vllm and model in await self.vllm_models():
            return self.vllm
        return self.ollama

    async def vllm_status(self) -> bool | None:
        """None = vllm-metal nicht aktiviert, sonst Erreichbarkeit (ungecacht)."""
        if self.vllm is None:
            return None
        try:
            await self.vllm.list_models()
            return True
        except Exception:
            return False

    async def chat_stream(
        self, messages: list[dict], model: str, temperature: float = 0.7, extra: dict | None = None
    ) -> AsyncIterator[str]:
        backend = await self.backend_for(model)
        async for line in backend.chat_stream(messages, model, temperature=temperature, extra=extra):
            yield line

    async def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        backend = await self.backend_for(model)
        return await backend.complete(
            messages, model, temperature=temperature, max_tokens=max_tokens,
            response_format=response_format,
        )

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        return await self.ollama.embed(texts, model)

    async def list_models(self) -> list[dict]:
        """Ollama-Modelle + vllm-metal-Modelle (Ollama-Fehler schlagen durch, vLLM-Fehler nicht)."""
        models = await self.ollama.list_models()
        return models + [{"name": n, "model": n} for n in sorted(await self.vllm_models())]

    async def context_length(self, model: str) -> int | None:
        return await (await self.backend_for(model)).context_length(model)

    async def show(self, model: str) -> dict:
        return await (await self.backend_for(model)).show(model)


def build_inference_client(settings, timeout: float = 120.0) -> InferenceRouter:
    """Router nach Konfiguration — vllm-metal nur, wenn VLLM_ENABLED=true."""
    vllm = VLLMClient(settings.vllm_base_url, timeout=timeout) if settings.vllm_enabled else None
    return InferenceRouter(OllamaClient(settings.ollama_base_url, timeout=timeout), vllm)
