"""Parst Ollamas OpenAI-kompatible SSE-Zeilen zu (content-delta, usage)-Paaren.

Eigene Funktion, damit RagService und der Corrective-RAG-Graph (app.flow) dieselbe
Logik nutzen.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from app.clients.ollama_client import OllamaClient


async def stream_deltas(
    ollama: OllamaClient, messages: list[dict], model: str, temperature: float = 0.7
) -> AsyncIterator[tuple[str | None, dict | None]]:
    async for line in ollama.chat_stream(messages, model, temperature=temperature):
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            return
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        usage = chunk.get("usage")
        for choice in chunk.get("choices", []):
            yield choice.get("delta", {}).get("content"), usage
        if not chunk.get("choices") and usage:
            yield None, usage
