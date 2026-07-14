# Makefile für book-RAG — vereinfacht Start/Stop/Cleanup und Dev-Aufgaben.
# `make` oder `make help` zeigt alle Targets.

FRONTEND_PORT ?= 4200
BACKEND_PORT  ?= 8001

.DEFAULT_GOAL := help
.PHONY: help up down restart clean setup models ollama-host ingest eval logs ps

help: ## Diese Übersicht anzeigen
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

up: ## Alles starten (Qdrant + Backend + Frontend) via docker compose
	docker compose up -d --build
	@echo ""
	@echo "Frontend:  http://localhost:$(FRONTEND_PORT)"
	@echo "Backend:   http://localhost:$(BACKEND_PORT)/health"
	@echo "Falls /health 'ollama:false' zeigt: einmalig 'make ollama-host' + Ollama-App neu starten."

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

ingest: ## Demo-Buch Dracula ingesten (Gruppe horror-classics)
	cd backend && uv run python -m app.ingestion.ingest_cli \
		../data/Dracula/dracula.txt --book-id dracula --group horror-classics \
		--title "Dracula" --author "Bram Stoker" --commit

eval: ## Retrieval-Evaluation gegen das Gold-Set (Recall@k, MRR)
	cd backend && uv run python ../eval/retrieval_eval.py \
		--gold ../eval/gold_dracula.jsonl --group horror-classics --k 8

logs: ## Container-Logs folgen
	docker compose logs -f

ps: ## Container-Status
	docker compose ps
