# Makefile für book-RAG — vereinfacht Start/Stop/Cleanup und Dev-Aufgaben.
# `make` oder `make help` zeigt alle Targets.

FRONTEND_PORT ?= 4200
BACKEND_PORT  ?= 8001

.DEFAULT_GOAL := help
.PHONY: help up up-ml down restart clean setup models ollama-host ollama-ctx ingest eval logs ps \
	graph graph-build eval-graph graph-communities eval-global

help: ## Diese Übersicht anzeigen
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

up: ## Alles starten (Qdrant + Backend + Frontend) via docker compose
	docker compose up -d --build
	@echo ""
	@echo "Frontend:  http://localhost:$(FRONTEND_PORT)"
	@echo "Backend:   http://localhost:$(BACKEND_PORT)/health"
	@echo "Falls /health 'ollama:false' zeigt: einmalig 'make ollama-host' + Ollama-App neu starten."

up-ml: ## Wie 'up', aber mit Reranker + Faithfulness (grosses Image: torch!)
	INSTALL_ML_EXTRAS=true RERANKER_ENABLED=true FAITHFULNESS_CHECK_ENABLED=true \
		docker compose up -d --build
	@echo ""
	@echo "ML-Extras aktiv. Status: curl http://localhost:$(BACKEND_PORT)/health"
	@echo "Achtung: Reranker + NLI + Ollama gleichzeitig ist auf 24 GB RAM knapp."

down: ## Alle Container stoppen (Volumes/Daten bleiben erhalten)
	docker compose down

restart: down up ## Neu starten

clean: ## ALLES entfernen: Container, Volumes, gebaute Images UND lokale Caches
	docker compose down -v --rmi local --remove-orphans || true
	rm -rf backend/.venv backend/.ruff_cache backend/.pytest_cache
	rm -rf frontend/node_modules frontend/dist frontend/.angular
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "Sauber. (data/ und Ollama-Modelle bleiben unangetastet.)"

setup: ## Lokale Backend-Umgebung (uv, Python 3.12) für Ingest/Eval
	cd backend && uv sync

models: ## Benötigte Ollama-Modelle ziehen (Embeddings + Chat)
	ollama pull bge-m3
	ollama pull qwen2.5:7b-instruct-q4_k_m

ollama-host: ## Einmalig (macOS): Host-Ollama für Container erreichbar machen
	launchctl setenv OLLAMA_HOST 0.0.0.0:11434
	@echo "Gesetzt. Bitte die Ollama-App EINMAL beenden und neu öffnen, dann 'make restart'."

ollama-ctx: ## Einmalig (macOS): Kontextfenster erhöhen — sonst schneidet Ollama RAG-Prompts bei 4096 Token ab
	launchctl setenv OLLAMA_CONTEXT_LENGTH 16384
	@echo "Gesetzt. Bitte die Ollama-App EINMAL beenden und neu öffnen (lädt Modelle mit num_ctx=16384)."

# Gruppe für Demo-Ingest + Eval (überschreibbar: make eval GROUP=...)
GROUP ?= Horror

ingest: ## Demo-Buch Dracula ingesten (Gruppe $(GROUP))
	cd backend && uv run python -m app.ingestion.ingest_cli \
		../data/Dracula/dracula.txt --book-id dracula --group "$(GROUP)" \
		--title "Dracula" --author "Bram Stoker" --commit

eval: ## Retrieval-Evaluation gegen das Gold-Set (Recall@k, MRR)
	cd backend && uv run python ../eval/retrieval_eval.py \
		--gold ../eval/gold_dracula.jsonl --group "$(GROUP)" --k 8

# Graph-RAG (Exp 8): Extraktion (~9 s/Chunk, Dracula ≈ 1,5 h, resumierbar) -> Graph-Build -> Eval.
graph: ## Graph-RAG: Entities/Relationen je Chunk extrahieren (Gruppe $(GROUP); Nachtjob)
	cd backend && uv run python -m app.graph.extract_cli --group "$(GROUP)"

graph-build: ## Graph-RAG: Graph + Entity-Embeddings aus der Extraktion bauen
	cd backend && uv run python -m app.graph.build_cli --group "$(GROUP)"

eval-graph: ## Exp 8: Graph-Arme vs. dense auf dem Span-Gold-Set (paired)
	cd backend && uv run --with matplotlib python ../eval/graph_experiment.py \
		--gold ../eval/gold_v2.jsonl --group "$(GROUP)" --k 8

graph-communities: ## Exp 9: Louvain-Communities + LLM-Berichte + Embeddings (Gruppe $(GROUP))
	cd backend && uv run python -m app.graph.communities_cli --group "$(GROUP)" --summarize --embed

eval-global: ## Exp 9: globaler Graph-Pfad vs. dense auf den thematischen Gold-Fragen (Judge + NLI)
	cd backend && uv run --with rouge-score --with transformers --with torch --with sentencepiece \
		--with protobuf --with matplotlib python ../eval/global_graph_experiment.py \
		--gold ../eval/gold_global.jsonl --group "$(GROUP)" --m 6 --control 3

logs: ## Container-Logs folgen
	docker compose logs -f

ps: ## Container-Status
	docker compose ps
