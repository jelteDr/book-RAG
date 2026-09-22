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

    # Small-to-Big (optional, opt-in): beste Treffer werden vor dem Prompt-Bau um
    # Nachbar-Chunks erweitert (kein Re-Embedding, ein Qdrant-Lookup pro Treffer).
    # Prompt-Kosten: jeder erweiterte Treffer wird bis zu (2*window+1)-mal so lang.
    small_to_big_enabled: bool = False
    s2b_window: int = 1  # Nachbar-Chunks je Richtung
    s2b_top_n: int = 3  # nur die besten N Treffer erweitern (Prompt-Budget)

    # Contextual Ingestion (optional, opt-in; Exp 5: +0.182 MRR).
    # Beim Ingest generiert das Chat-LLM pro Chunk 1-2 Sätze Kontext (Pronomen
    # aufgelöst, Ort/Geschehen), die NUR ins Embedding eingehen. Kostet ~4-6 s
    # pro Chunk — ein Buch-Upload dauert damit deutlich länger.
    contextual_ingest_enabled: bool = False

    # Cross-Encoder-Reranker (optional, opt-in; braucht die reranker-Dependency-Gruppe).
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_candidates: int = 30  # Dense holt so viele Kandidaten, Reranker sortiert auf top_k

    # Graph-RAG (optional, opt-in; Exp 8). Braucht einen gebauten Graphen unter
    # `<graph_dir>/<group>/` (make graph + make graph-build). Dense bleibt Träger:
    # score = cos + alpha * graph_signal — bei alpha=0 identisch zur Baseline.
    graph_rag_enabled: bool = False
    graph_rag_mode: str = "local"  # local | global (Community-Berichte, Exp 9) | auto (Router)
    graph_local_mode: str = "link"  # link (Entity-Linking) | expand (Nachbar-Expansion)
    graph_dir: str = "../data/graph"  # im Container: /data/graph (Volume ./data)
    graph_alpha: float = 0.03  # Gewicht des Graph-Signals in der Fusion
    graph_top_m: int = 5  # so viele Entities werden pro Frage verlinkt
    graph_min_sim: float = 0.45  # Mindest-Cosine Frage<->Entity fürs Linking
    graph_global_m: int = 6  # so viele Community-Berichte werden pro globaler Frage vorausgewählt
    graph_global_map: bool = False  # Map-Step (ein LLM-Aufruf je Bericht) vor der Antwort
    graph_global_chunks_per_community: int = 2  # Originalpassagen je Bericht als Zitatquellen

    # Faithfulness-Check (optional, opt-in; markiert ungestützte Zitate via NLI).
    # Braucht ebenfalls die reranker-Dependency-Gruppe (torch/transformers).
    faithfulness_check_enabled: bool = False
    faithfulness_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    faithfulness_threshold: float = 0.5


settings = Settings()
