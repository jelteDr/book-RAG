"""Async-Client für vllm-metal (vLLM auf der Apple-GPU via MLX).

Chat/Completion laufen wie bei Ollama über `/v1` (s. `OpenAICompatClient`). vLLM hat
keine native Metadaten-API — Liste, Kontextlänge und Details kommen aus `/v1/models`
und werden auf die Form gebracht, die Registry und Routen von Ollama kennen.

Ein vLLM-Prozess bedient genau EIN Modell (Modellname = HF-Repo-ID, z. B.
`mlx-community/Qwen2.5-7B-Instruct-4bit`, oder `--served-model-name`).
"""

import re

from app.clients.openai_compat import OpenAICompatClient

# Quantisierung steckt bei HF/MLX-Repos nur im Namen (z. B. "-4bit", "-AWQ", "Q4_K_M").
_QUANT_RE = re.compile(r"(\d+bit|awq|gptq|fp8|q\d_k_[ms]|q8_0)", re.IGNORECASE)


class VLLMClient(OpenAICompatClient):
    def _adapt_extra(self, extra: dict | None) -> dict:
        # Ollama schaltet das Denken über `reasoning_effort: "none"` ab, vLLM über das
        # Chat-Template (Qwen3 & Co. werten `enable_thinking` aus).
        extra = dict(extra or {})
        if extra.pop("reasoning_effort", None) == "none":
            extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        return extra

    async def _models(self) -> list[dict]:
        resp = await self._client.get("/v1/models")
        resp.raise_for_status()
        return resp.json().get("data", [])

    async def list_models(self) -> list[dict]:
        """Bediente Modelle im Format von Ollamas `/api/tags` (`name`, `model`)."""
        return [{"name": m["id"], "model": m["id"]} for m in await self._models()]

    async def context_length(self, model: str) -> int | None:
        """`max_model_len` des Modells — bei vLLM fest ab Serverstart (`--max-model-len`)."""
        for m in await self._models():
            if m.get("id") == model:
                return m.get("max_model_len")
        return None

    async def show(self, model: str) -> dict:
        """Metadaten im Format von Ollamas `/api/show` (soweit vLLM sie kennt)."""
        ctx = await self.context_length(model)
        quant = _QUANT_RE.search(model)
        return {
            "backend": "vllm",
            "capabilities": ["completion"],
            "details": {
                "family": model.rsplit("/", 1)[-1].split("-")[0].lower(),
                "quantization_level": quant.group(1) if quant else None,
            },
            "model_info": {"vllm.context_length": ctx} if ctx else {},
        }
