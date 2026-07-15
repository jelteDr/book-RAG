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

    # Backend / Persistenz (PostgreSQL via SQLModel/asyncpg).
    backend_port: int = 8000
    database_url: str = "postgresql+asyncpg://bookrag:bookrag@localhost:5432/bookrag"

    # Retrieval.
    top_k: int = 8

    # Cross-Encoder-Reranker (optional, opt-in; braucht die reranker-Dependency-Gruppe).
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_candidates: int = 30  # Dense holt so viele Kandidaten, Reranker sortiert auf top_k

    # Faithfulness-Check (optional, opt-in; markiert ungestützte Zitate via NLI).
    # Braucht ebenfalls die reranker-Dependency-Gruppe (torch/transformers).
    faithfulness_check_enabled: bool = False
    faithfulness_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    faithfulness_threshold: float = 0.5


settings = Settings()
