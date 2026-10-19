"""Gemeinsame Basis für OpenAI-kompatible Inferenz-Server (Ollama, vllm-metal).

Chat, Completion und Embeddings laufen bei beiden Servern über `/v1` — nur die
Modell-Metadaten (Liste, Kontextlänge, Details) sind serverspezifisch und stecken
in den Unterklassen.
"""

from collections.abc import AsyncIterator

import httpx


class OpenAICompatClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _adapt_extra(self, extra: dict | None) -> dict:
        """Hook für serverspezifische Zusatzparameter (z. B. Denken abschalten)."""
        return extra or {}

    async def chat_stream(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        extra: dict | None = None,
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
            # z. B. {"reasoning_effort": "none"}: schaltet bei Thinking-Modellen (qwen3, gemma4)
            # das Denken ab — bei Ollama 0.34 wirkt nur dieser Parameter über /v1.
            **self._adapt_extra(extra),
        }
        async with self._client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                yield line

    async def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Einzelne, nicht-gestreamte Completion (z. B. für Query-Condensation).

        `response_format={"type": "json_object"}` schaltet den JSON-Modus ein
        (Graph-Extraktion) — nur setzen, wenn der Prompt explizit JSON verlangt.
        """
        payload: dict = {"model": model, "messages": messages, "stream": False, "temperature": temperature}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if response_format:
            payload["response_format"] = response_format
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
