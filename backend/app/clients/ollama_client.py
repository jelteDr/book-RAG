"""Async-Client für Ollama.

Chat und Embeddings laufen über den **OpenAI-kompatiblen `/v1`-Endpunkt**
(liefert Streaming + `usage`-Tokens); die Modell-Liste über die native
Ollama-API `/api/tags`.
"""

from collections.abc import AsyncIterator

import httpx


class OllamaClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[dict]:
        """Lokal verfügbare Modelle (native API `/api/tags`)."""
        resp = await self._client.get("/api/tags")
        resp.raise_for_status()
        return resp.json().get("models", [])

    async def show(self, model: str) -> dict:
        """Detail-Metadaten eines Modells (native API `/api/show`)."""
        resp = await self._client.post("/api/show", json={"model": model})
        resp.raise_for_status()
        return resp.json()

    async def chat_stream(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Streamt die rohen SSE-Zeilen der Chat-Completion (OpenAI-kompatibel).

        `stream_options.include_usage` sorgt dafür, dass am Ende ein Chunk mit
        `usage` (prompt/completion_tokens) für das Monitoring kommt.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "stream_options": {"include_usage": True},
        }
        async with self._client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                yield line

    async def complete(
        self, messages: list[dict], model: str, temperature: float = 0.0, max_tokens: int | None = None
    ) -> str:
        """Einzelne, nicht-gestreamte Completion (z. B. für Query-Condensation)."""
        payload: dict = {"model": model, "messages": messages, "stream": False, "temperature": temperature}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        resp = await self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Erzeugt Embeddings (OpenAI-kompatibel `/v1/embeddings`)."""
        resp = await self._client.post(
            "/v1/embeddings", json={"model": model, "input": texts}
        )
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]
