# book-RAG — Lokale RAG-Anwendung für Buchtexte mit Zitierung & Monitoring

Eine lokal lauffähige **Retrieval-Augmented-Generation**-Anwendung: Buchtexte
werden aufbereitet und indexiert, und ein lokales LLM (via **Ollama**) beantwortet
Fragen dazu — mit **Zitier-Markern [n]**, die jede Aussage auf konkrete
Quell-Passagen zurückführen (Grounding & Erklärbarkeit). Dazu eine methodisch
saubere Evaluation (Retrieval-Qualität, Faithfulness) und ein Serving-Benchmark.

> **Portfolio-Projekt** (Ziel: Data Scientist / ML Engineer). Es verbindet
> Engineering (Ingestion-Pipeline, Container, Streaming-API) mit angewandter
> Data Science (kontrollierte Evaluation, Retrieval-Metriken, NLI-Faithfulness).

## Architektur (Kurzüberblick)

```
Angular (minimal) ──REST + SSE──► FastAPI (Python 3.12)
                                     │
                        ┌────────────┼────────────┐
                        ▼            ▼             ▼
                     Ollama       Qdrant         SQLite
                  (Chat + bge-m3) (Vektoren)   (Metadaten/Log)
```

- **Ollama** — lokales LLM (`qwen2.5:7b-instruct`) + Embeddings (`bge-m3`, multilingual)
- **FastAPI** — Retrieval, Prompt-Bau, SSE-Streaming (Time-To-First-Token messbar)
- **Qdrant** — Vektor-Suche mit Metadaten-Filter
- **SQLite** — Metadaten + Monitoring-Log
- **Notebooks** — Evaluation (Recall@k, MRR/nDCG), Faithfulness (NLI), Benchmark

Details: [`docs/PLAN.md`](docs/PLAN.md).

## Voraussetzungen (macOS, Apple Silicon)

- Docker, Ollama, `uv` (Python-3.12-Toolchain)
- **Host-Ollama für Container erreichbar machen** (sonst scheitert `docker compose up`):
  ```bash
  launchctl setenv OLLAMA_HOST 0.0.0.0:11434   # danach Ollama neu starten
  ollama pull qwen2.5:7b-instruct-q4_K_M
  ollama pull bge-m3
  ```

## Schnellstart (dev-Profil: Ollama nativ auf dem Host)

```bash
cp .env.example .env
docker compose up -d qdrant          # Vektor-DB
# Backend lokal (uv) oder via Container:
uv run --project backend uvicorn app.main:app --reload --port 8000
curl localhost:8000/health
curl localhost:8000/models           # listet lokale Ollama-Modelle
```

## Git-Workflow (Git Flow)

- `main` — stabiler Stand
- `develop` — Integrationsbranch (aktueller Arbeitsstand)
- `feature/<name>` — einzelne Arbeitspakete, von `develop` abgezweigt, per
  `git merge --no-ff` zurück nach `develop`

## Token / Push (einmalig, Nutzer)

Das Remote ist bereits angelegt: `https://github.com/jelteDr/book-RAG.git`.
Der Personal Access Token wird **nur beim ersten Push** eingegeben und von der
macOS-Keychain gespeichert (Anti-Pattern: Token **nie** in die Remote-URL, `.env`
oder einen Commit):

```bash
git push -u origin main
git push -u origin develop
# Username: jelteDr   |   Passwort: <Personal Access Token>  (Scope: repo)
```

## ⚠️ Sicherheit & Urheberrecht

Das Repo ist **öffentlich**. Daher gilt hart:
- **Buchtexte, Datenbanken, Vektorstore, Modelle, `.env` werden nie committet**
  (siehe `.gitignore`). Sie liegen nur lokal bzw. in Docker-Volumes.
- Öffentlich reproduzierbar ist die Demo nur mit einem **Public-Domain-Werk**
  (Project Gutenberg). Urheberrechtlich geschützte Texte (z. B. Game of Thrones)
  bleiben ausschließlich lokal.
- Pre-Commit-Hooks (`gitleaks`, `nbstripout`) als Sicherheitsnetz:
  ```bash
  uv tool install pre-commit   # oder: pipx install pre-commit
  pre-commit install
  ```

## Status

In Entwicklung. Meilensteine (M0–M5) in [`docs/PLAN.md`](docs/PLAN.md);
**M2 markiert den vorzeigbaren Projektstand** (M3–M5 optional).

## Lizenz

[MIT](LICENSE). Demo-Texte von [Project Gutenberg](https://www.gutenberg.org)
(gemeinfrei) mit entsprechender Attribution.
