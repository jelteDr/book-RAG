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

    # vllm-metal (optional, opt-in): zweiter Inferenz-Server auf der Apple-GPU (MLX).
    # Chat/Completion für die dort bedienten Modelle laufen über vLLM, alles andere und
    # ALLE Embeddings bleiben bei Ollama (Index-Konsistenz). Default-Modell umstellen:
    # CHAT_MODEL auf den vLLM-Modellnamen setzen (z. B. mlx-community/Qwen2.5-7B-Instruct-4bit).
    vllm_enabled: bool = False
    vllm_base_url: str = "http://localhost:8010"

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

    # Corrective RAG (optional, opt-in; Exp 11): LangGraph-Schleife über Chat-Request
    # `mode=crag`. Grader bewertet die Treffer, bei zu wenig Relevantem wird die Anfrage
    # umgeschrieben und neu gesucht; mit aktivem Faithfulness-Check wird eine Antwort mit
    # zu vielen unbelegten Zitaten einmal streng neu generiert. Kostet 1-3 LLM-Aufrufe extra.
    crag_enabled: bool = False
    crag_max_retries: int = 1  # zusätzliche Suchläufe nach Umschreiben
    crag_min_relevant: int = 2  # so viele relevante Passagen reichen für die Antwort
    crag_max_regen: int = 1  # Neu-Generierungen nach dem Faithfulness-Check
    crag_unsupported_max: float = 0.5  # Anteil unbelegter Zitate, ab dem neu generiert wird
    crag_grader_model: str | None = None  # None = Chat-Modell der Anfrage
    # filter (Exp 11): nur grader-relevante Passagen in den Prompt (gemessen: negativ).
    # trigger (Exp 11b): Grader löst nur den Retry aus, Trefferlisten werden gemischt statt gefiltert.
    # Default trigger: in Exp 11b neutral, filter in Exp 11 schädlich.
    crag_strategy: str = "trigger"


settings = Settings()
