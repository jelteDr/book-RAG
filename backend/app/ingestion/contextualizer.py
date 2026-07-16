"""Contextual Ingestion: LLM-generierter Kurz-Kontext pro Chunk (Exp 5: +0.182 MRR).

Ein Chunk mitten aus Kapitel 12 weiß nicht, dass „he" Jonathan Harker ist. Vor dem
Embedden generiert das Chat-LLM deshalb 1–2 Sätze Kontext (Figuren benannt, Pronomen
aufgelöst, Ort/Geschehen). Der Kontext geht NUR ins Embedding ein — der anzeigbare
Chunk-Text bleibt unverändert; zur Transparenz wird er als `context` in der Payload
mitgespeichert.
"""

from __future__ import annotations

from collections.abc import Callable

from app.clients.ollama_client import OllamaClient

_CTX_PROMPT = (
    'Here is an excerpt from the novel "{title}" ({chapter}).\n\n'
    "---\n{text}\n---\n\n"
    "Write 1-2 short English sentences situating this excerpt within the novel: "
    "name the characters involved (resolve pronouns), the location and what is "
    "happening. Answer with ONLY those sentences."
)


async def generate_context(
    ollama: OllamaClient, model: str, *, title: str | None, chapter: str | None, text: str
) -> str:
    """Erzeugt den Kontext für einen Chunk; bei Fehlern leer (Chunk bleibt nutzbar)."""
    prompt = _CTX_PROMPT.format(
        title=title or "?",
        chapter=chapter or "beginning of the book",
        text=text[:2400],
    )
    try:
        ctx = await ollama.complete(
            [{"role": "user", "content": prompt}], model, temperature=0.0, max_tokens=90
        )
        return " ".join(ctx.split())
    except Exception:
        return ""


def embedding_text(context: str, chunk_text: str) -> str:
    """Der Text, der tatsächlich eingebettet wird (Kontext-Präfix, falls vorhanden)."""
    return f"{context}\n\n{chunk_text}" if context else chunk_text


async def generate_contexts(
    ollama: OllamaClient,
    model: str,
    *,
    title: str | None,
    chunks: list,  # Objekte/Payloads mit .chapter/.text bzw. ["chapter"]/["text"]
    on_progress: Callable[[int, int], None] | None = None,
) -> list[str]:
    """Kontexte für eine Chunk-Liste (sequentiell — Ollama serialisiert Generierung ohnehin)."""
    contexts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        chapter = chunk["chapter"] if isinstance(chunk, dict) else chunk.chapter
        text = chunk["text"] if isinstance(chunk, dict) else chunk.text
        contexts.append(
            await generate_context(ollama, model, title=title, chapter=chapter, text=text)
        )
        if on_progress and (i % 25 == 0 or i == len(chunks)):
            on_progress(i, len(chunks))
    return contexts
