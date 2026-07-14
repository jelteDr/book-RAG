"""Zentrale Konfiguration, aus Umgebungsvariablen / .env geladen."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Feldnamen werden case-insensitiv aus den Env-Vars gefüllt
    # (z. B. OLLAMA_BASE_URL -> ollama_base_url).
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ollama (Chat + Embeddings), OpenAI-kompatibler /v1-Endpunkt.
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "qwen2.5:7b-instruct-q4_k_m"
    embed_model: str = "bge-m3"  # FEST – Wechsel würde den Index inkonsistent machen.

    # Qdrant.
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "book_chunks"

    # Backend / Persistenz.
    backend_port: int = 8000
    sqlite_path: str = "./data/book_rag.db"

    # Retrieval.
    top_k: int = 8


settings = Settings()
