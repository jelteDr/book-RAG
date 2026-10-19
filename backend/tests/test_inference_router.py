"""Unit-Tests für InferenceRouter + VLLMClient — Fake-Server via httpx.MockTransport, kein Netz.

Aufruf (aus backend/): uv run pytest tests -q
"""

from __future__ import annotations

import asyncio
import json

import httpx

from app.clients.inference_router import InferenceRouter
from app.clients.ollama_client import OllamaClient
from app.clients.vllm_client import VLLMClient

VLLM_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


def _chat_reply(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def _ollama(calls: list) -> OllamaClient:
    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(("ollama", req.url.path, json.loads(req.content or b"{}")))
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b"}]})
        if req.url.path == "/v1/embeddings":
            return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})
        return httpx.Response(200, json=_chat_reply("von ollama"))

    client = OllamaClient("http://ollama")
    client._client = httpx.AsyncClient(base_url="http://ollama", transport=httpx.MockTransport(handler))
    return client


def _vllm(calls: list, up: bool = True) -> VLLMClient:
    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(("vllm", req.url.path, json.loads(req.content or b"{}")))
        if not up:
            raise httpx.ConnectError("down", request=req)
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": VLLM_MODEL, "max_model_len": 16384}]})
        return httpx.Response(200, json=_chat_reply("von vllm"))

    client = VLLMClient("http://vllm")
    client._client = httpx.AsyncClient(base_url="http://vllm", transport=httpx.MockTransport(handler))
    return client


def test_chat_routes_by_model_and_embeddings_stay_on_ollama():
    calls: list = []
    router = InferenceRouter(_ollama(calls), _vllm(calls))

    async def go():
        a = await router.complete([{"role": "user", "content": "hi"}], VLLM_MODEL)
        b = await router.complete([{"role": "user", "content": "hi"}], "qwen2.5:7b")
        await router.embed(["text"], "bge-m3")
        return a, b

    a, b = asyncio.run(go())
    assert (a, b) == ("von vllm", "von ollama")
    assert ("ollama", "/v1/embeddings") in [(s, p) for s, p, _ in calls]
    assert not any(s == "vllm" and p == "/v1/embeddings" for s, p, _ in calls)


def test_list_models_merges_both_and_survives_vllm_outage():
    calls: list = []
    up = InferenceRouter(_ollama(calls), _vllm(calls))
    down = InferenceRouter(_ollama(calls), _vllm(calls, up=False))

    names_up = [m["name"] for m in asyncio.run(up.list_models())]
    names_down = [m["name"] for m in asyncio.run(down.list_models())]
    assert names_up == ["qwen2.5:7b", VLLM_MODEL]
    assert names_down == ["qwen2.5:7b"]
    assert asyncio.run(down.vllm_status()) is False
    assert asyncio.run(InferenceRouter(_ollama(calls)).vllm_status()) is None


def test_vllm_metadata_in_ollama_shape():
    calls: list = []
    router = InferenceRouter(_ollama(calls), _vllm(calls))
    show = asyncio.run(router.show(VLLM_MODEL))
    assert asyncio.run(router.context_length(VLLM_MODEL)) == 16384
    assert show["backend"] == "vllm"
    assert show["details"]["quantization_level"] == "4bit"
    assert show["model_info"] == {"vllm.context_length": 16384}


def test_vllm_translates_thinking_switch():
    extra = VLLMClient("http://vllm")._adapt_extra({"reasoning_effort": "none", "top_p": 0.9})
    assert extra == {"top_p": 0.9, "chat_template_kwargs": {"enable_thinking": False}}
