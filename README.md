# book-RAG — Lokale RAG-Anwendung für Buchtexte mit Zitierung & Evaluation

Eine vollständig **lokal lauffähige Retrieval-Augmented-Generation-Anwendung**: Buchtexte
werden aufbereitet und indexiert, ein lokales LLM (via **Ollama**) beantwortet Fragen dazu —
**auf Deutsch, mit anklickbaren Zitier-Markern `[n]`**, die jede Aussage auf die konkrete
Quell-Passage zurückführen (Grounding & Erklärbarkeit). Kern des Projekts ist eine
**methodisch saubere, ehrliche Evaluation** der Retrieval- und Antwortqualität.

## Demo

Deutsche Frage → englischer Quelltext (cross-lingual) → gestreamte Antwort mit `[n]`-Zitaten;
Klick auf einen Marker zeigt die exakte Quell-Passage.

**Ablauf:** Frage eintippen → Antwort wird tokenweise gestreamt (TTFT sichtbar) → jede Aussage
trägt einen `[n]`-Marker → Klick auf einen Marker/Chip öffnet den zugehörigen Quell-Chunk mit
Buch, Kapitel und Ähnlichkeits-Score. Lokal reproduzierbar via `make up` (siehe Schnellstart).

## Architektur

```
Angular (minimal, Signals) ──REST + SSE──► FastAPI (Python 3.12)
 Chat · Dokumente · Modelle · Dashboard            │  RagService (Service-Schicht)
                                          ┌─────────┼──────────┐
                                          ▼         ▼          ▼
                                       Ollama     Qdrant    PostgreSQL
                                    Chat + bge-m3  Vektoren  (Metadaten · Chats · query_log)
```

- **Ollama** — lokales LLM (`qwen2.5:7b-instruct`) + Embeddings (`bge-m3`, multilingual)
- **FastAPI** — dünne Routen + `RagService` (retrieve → Prompt → Streaming → Zitat-Mapping)
- **Qdrant** — Vektor-Suche (Cosine) mit Metadaten-Filter (Gruppen/Bücher/Sammelgruppen)
- **PostgreSQL** — Gruppen/Bücher, Modell-Registry, gespeicherte Unterhaltungen, Anfrage-Log
- **Angular** — schlanke SPA (Standalone-Components, Signals, `@if/@for`), nginx-`/api`-Proxy für SSE;
  Views: Chat, Dokumente (Upload/Gruppen), Modelle (Metadaten), Dashboard (Live-Telemetrie)
- **Notebooks/Skripte** — die komplette Evaluations-Suite (`eval/`, `notebooks/`)

## Features

- Robuste **Ingestion-Pipeline**: Encoding-Sniffing, Gutenberg-Boilerplate-Strip, deterministische
  **Textbereinigung** (De-Wrapping, Zeichen-Normalisierung) mit **Dry-Run-Prüf-Gate**, strukturbewusstes
  Chunking (Kapitel + Offsets auf bereinigtem Text), Input-Validierung vor dem Embedden.
- **Upload-UI mit Gruppen**: Texte im Browser hochladen (Dry-Run-Report → Commit), in Gruppen
  (Genre/Franchise) organisieren; Slugs entstehen automatisch aus dem Namen.
- **Sammelgruppen**: mehrere Gruppen unter einem Namen bündeln (z. B. „Fantasy" = GoT + Harry
  Potter) — **ohne Re-Ingestion**; der Slug expandiert beim Retrieval zu einem
  Qdrant-`MatchAny`-Filter über die Mitglieds-Gruppen.
- **Grounding via Zitat-Contract**: nummerierte Quellen im Prompt, `[n]` wird **serverseitig** auf
  Chunk-IDs gemappt (halluzinierte Marker werden verworfen); Quell-Passagen werden mit der
  Unterhaltung **persistiert** und bleiben beim Wiederöffnen lesbar.
- **Gespeicherte Chats mit Verlauf**: Unterhaltungen in Postgres, Sidebar nach Gruppen gebündelt;
  **History-aware Retrieval** (Folgefragen werden per LLM zu eigenständigen Suchanfragen kondensiert).
- **Cross-linguales Retrieval** (deutsche Fragen, englische Texte) via `bge-m3`.
- **Dashboard**: Live-Telemetrie aus dem `query_log` — Anfragen, Ø TTFT, Ø TPS pro Modell,
  inkl. Vergleichs-Barplots; Modell-Metadaten (Parameter, Quantisierung, Kontext) aus Ollama.
- **Optionale Qualitäts-Services** (opt-in): Cross-Encoder-**Reranker** (`bge-reranker-v2-m3`)
  und **NLI-Faithfulness-Check**, der ungestützte Zitate im Frontend mit ⚠ markiert.
- **Contextual Ingestion** (opt-in, `CONTEXTUAL_INGEST_ENABLED`): Beim Ingest generiert das
  LLM pro Chunk 1–2 Sätze Kontext (Pronomen aufgelöst), die nur ins Embedding eingehen —
  **stärkster gemessener Retrieval-Hebel** (Exp 5: MRR +0.182). Bestehende Bücher lassen
  sich ohne Originaldatei umstellen (`app.ingestion.reembed_cli`).
- **Serving-Metriken** pro Anfrage: TTFT, TPS, Tokens.

## Schnellstart (macOS, Apple Silicon)

Voraussetzungen: Docker, Ollama, [`uv`](https://docs.astral.sh/uv/).

```bash
cd book-RAG
make ollama-host          # EINMALIG (macOS): OLLAMA_HOST=0.0.0.0 -> danach Ollama-App neu starten
make models               # EINMALIG: bge-m3 + qwen2.5:7b ziehen
make up                   # Qdrant + Backend + Frontend starten
make ingest               # Demo-Buch (Public Domain) ingesten
# -> Frontend: http://localhost:4200   Backend: http://localhost:8001/health
```
`make down` stoppt alles, `make clean` entfernt auch Volumes/Images/Caches, `make help` listet alle Targets.

> **macOS-Hinweis:** Docker-Container erreichen das Host-Ollama nur, wenn es auf `0.0.0.0` lauscht
> (`make ollama-host` + Ollama-App-Neustart). Das ist der einzige manuelle Einmalschritt.

## Evaluation (der Kern)

Alle Experimente sind reproduzierbar (`eval/`), gegen ein handgelabeltes Gold-Set
(`eval/gold_dracula.jsonl`, **n=21**, Kapitel-Labels im Quelltext verankert). Volle Ergebnisse
und Methodik-Caveats: [`eval/RESULTS.md`](eval/RESULTS.md). Ein **Gold-Set v2** mit
passagen-genauen Span-Labels entsteht gerade in
[`notebooks/gold_set_v2.ipynb`](notebooks/gold_set_v2.ipynb) (Labeling-Helfer auf der echten
Pipeline-Suche, interaktive PCA-Karte des Embedding-Raums, Wortwolken).

> **Ehrlichkeits-Hinweis:** n=21 ist ein Dev-Set ohne statistische Signifikanz — die Ergebnisse
> sind Trends, keine belastbaren Rankings. „Recall@k" ist hier faktisch **Hit-Rate@k**.

### Was ich untersucht habe (und was die Daten *wirklich* sagen)

**1. Deutsche vs. englische Frage** — Hypothese: der cross-linguale Gap ist der Flaschenhals.
→ **Teilweise widerlegt:** `bge-m3` ist echt multilingual (DE rankt den Top-Treffer sogar besser);
Query-Übersetzung ist **kein** Fix.

![DE vs EN](results/lang_experiment.png)

**2. Dense vs. BM25 vs. Hybrid (RRF)** — *adversariell verifiziert* durch ein 4-Skeptiker-Panel,
das meinen ersten (zu optimistischen) Entwurf korrigierte (mislabeled Metrik + latenter Bug).
→ **Naives RRF-Hybrid bringt keinen belastbaren Netto-Vorteil** (gepaarte Bilanz 4:4); BM25 solo
am schwächsten, aber cross-lingual gehandicapt.

![Dense vs BM25 vs Hybrid](results/hybrid_experiment.png)

**3. Cross-Encoder-Reranker** (`bge-reranker-v2-m3`) → verbessert **Precision@1 (+0.05) und MRR
(+0.035)** (bringt einen Treffer nach vorn) — die kapitel-basierte Metrik unterbewertet den
passagen-basierten Reranker allerdings systematisch.

![Dense vs Reranker](results/reranker_experiment.png)

### Antwortqualität (Metrik-Suite, `eval/answer_eval.py`)

Pro Frage wird die volle RAG-Pipeline ausgeführt und gemessen: **ROUGE-L**, **Antwort-Token-F1**,
**Faithfulness** (Entailment via `mDeBERTa-xnli`, bewusst statt PPL), **Recall@k/MRR**, **TTFT/TPS**.

| Kategorie | Metrik | Wert (mean, n=21) |
|---|---|---|
| Antwort | ROUGE-L | 0.124 |
| Antwort | Antwort-Token-F1 | 0.153 |
| Antwort | **Faithfulness (NLI)** | **0.667** |
| Retrieval | Recall@8 | 0.770 |
| Retrieval | MRR | 0.556 |
| Serving | TTFT (median) | ~7.4 s\* |
| Serving | TPS (median) | 11.8 tok/s |
| Serving | e2e (median) | ~19 s |

<sub>\* TTFT im Batch-Lauf durch Speicherdruck/Modell-Reloads erhöht; interaktiv/warm ~0,2–4 s.</sub>

**Kernbeobachtung:** ROUGE-L/F1 sind niedrig, obwohl die Antworten korrekt sind — sie messen nur
n-Gramm-**Oberflächenüberlappung** und bestrafen Paraphrasen. **Faithfulness (NLI)** ist für ein
generatives RAG-System das aussagekräftigere Qualitätsmaß (misst *Stützung*, nicht Wortgleichheit).

## Wichtige Design-Entscheidungen

- **Ollama statt vLLM:** läuft nativ (Metal) auf Apple Silicon; ein Prozess für Chat + Embeddings.
- **`bge-m3` (Embeddings):** multilingual → deutsche Fragen gegen englische Texte, ohne `torch`-Abhängigkeit im Backend.
- **Service-Schicht (`RagService`):** Geschäftslogik gekapselt, Routen dünn (OOP + KISS).
- **Faithfulness statt PPL:** Perplexität misst Fluenz, nicht Korrektheit/Grounding.
- **Sofort-public Repo:** strikte History-Hygiene (nur Public-Domain-Demotext; geschützte Texte bleiben lokal).

**Optional: Cross-Encoder-Reranker** (`bge-reranker-v2-m3`) **und NLI-Faithfulness-Check**
(`mDeBERTa-xnli`). Standardmäßig aus (hält das Image schlank); lokal via `uv sync --group reranker`
+ `RERANKER_ENABLED=true` / `FAITHFULNESS_CHECK_ENABLED=true`. Dann holt Dense
`RERANK_CANDIDATES` (30) Kandidaten, der Reranker sortiert sie neu und gibt die Top-`TOP_K` ans LLM;
der Faithfulness-Check markiert Zitate, die die Quelle laut NLI nicht stützt (⚠ im Frontend).
Beide Modelle laden lazy und laufen im Thread. Status unter `GET /health`.

## Projektstruktur

```
backend/    FastAPI (routes/ · services/ · rag/ · ingestion/ · clients/ · db/)
frontend/   Angular (chat/ · groups/ · models/ · dashboard/ · services), nginx-Proxy
eval/       Gold-Set + Experimente (retrieval/lang/hybrid/reranker/answer) + RESULTS.md
notebooks/  DS-Notebooks (Gold-Set v2: Span-Labels, PCA-Karte, Wortwolken)
results/    Charts & Roh-Ergebnisse
benchmark/  Serving-Benchmark (Ollama vs. llama-server)
docs/       Projektplan
```

## Git-Workflow & Sicherheit

Git Flow: `main` (stabil) ← `develop` (Integration) ← `feature/*`. Commits auf Deutsch.
**Nie committet:** Buchtexte, Datenbanken, Vektorstore, Modelle, Secrets (`.gitignore` + Pre-Commit-Hooks
`gitleaks`/`nbstripout`). Urheberrechtlich geschützte Texte (z. B. Game of Thrones) bleiben ausschließlich lokal.

## Grenzen & Roadmap

- Eval-Set klein (n=21, kapitel-basiert) → **Gold-Set v2 mit Span-Labels + n≥30–50** ist in
  Kuration (`notebooks/gold_set_v2.ipynb`); danach Eval-Skripte auf Span-Overlap-Metrik umstellen.
- **Contextual Retrieval** (LLM-generierter Chunk-Kontext beim Ingest) als nächstes Experiment.
- **Sparse-Hybrid** mit bge-m3-eigenen Sparse-Gewichten (adressiert das cross-linguale
  BM25-Handicap aus Exp 2).
- Bewusst minimal (lokales Projekt): kein Auth/CORS/Rate-Limiting.
- ~~Reranker in die `/chat`-Pipeline integrieren~~ ✅ umgesetzt (opt-in, s. o.).
- ~~Ingestion-UI, Gruppen-Verwaltung, Modellwechsel-UI~~ ✅ umgesetzt.

## Lizenz

[MIT](LICENSE). Demo-Texte von [Project Gutenberg](https://www.gutenberg.org) (gemeinfrei).
