"""Async-Client für Ollama.

Chat und Embeddings laufen über den **OpenAI-kompatiblen `/v1`-Endpunkt**
(liefert Streaming + `usage`-Tokens, s. `OpenAICompatClient`); Modell-Liste
und -Metadaten über die native Ollama-API (`/api/tags`, `/api/ps`, `/api/show`).
"""

from app.clients.openai_compat import OpenAICompatClient


class OllamaClient(OpenAICompatClient):
    async def list_models(self) -> list[dict]:
        """Lokal verfügbare Modelle (native API `/api/tags`)."""
        resp = await self._client.get("/api/tags")
        resp.raise_for_status()
        return resp.json().get("models", [])

    async def context_length(self, model: str) -> int | None:
        """Kontextfenster (num_ctx) des GELADENEN Modells laut `/api/ps`; None, wenn nicht geladen.

        Ollama teilt OLLAMA_CONTEXT_LENGTH auf OLLAMA_NUM_PARALLEL Slots auf und schneidet
        längere Prompts still ab (Server-Log: „truncating input prompt") — das hat in diesem
        Projekt schon dreimal Messungen entwertet. Eval-Skripte prüfen deshalb vorab.
        """
        resp = await self._client.get("/api/ps")
        resp.raise_for_status()
        for m in resp.json().get("models", []):
            if m.get("name") == model or m.get("model") == model:
                return m.get("context_length")
        return None

    async def show(self, model: str) -> dict:
        """Detail-Metadaten eines Modells (native API `/api/show`)."""
        resp = await self._client.post("/api/show", json={"model": model})
        resp.raise_for_status()
        return resp.json()
