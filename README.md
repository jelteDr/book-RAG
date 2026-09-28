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
  — Apple-nahes Designsystem auf Tailwind-Tokens (hell/dunkel, responsiv; siehe [`frontend/DESIGN.md`](frontend/DESIGN.md))
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
make ollama-ctx           # EINMALIG: Kontextfenster 16k — sonst schneidet Ollama RAG-Prompts bei 4096 Token ab!
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

| Kategorie | Metrik | k=8 (Default) | k=6 | k=4 |
|---|---|---|---|---|
| Antwort | ROUGE-L | 0.172 | 0.185 | 0.173 |
| Antwort | Antwort-Token-F1 | 0.197 | 0.217 | 0.207 |
| Antwort | **Faithfulness (NLI)** | **0.448** | **0.505** | **0.564** |
| Ehrlichkeit | Refusal-Rate (unanswerable) / False-Refusal | 0.75 / 0.00 | 0.75 / 0.00 | 0.75 / 0.00 |
| Retrieval | Hit-Rate@k (span) | **0.812** | 0.781 | 0.688 |
| Retrieval | MRR (span) | **0.474** | 0.469 | 0.451 |
| Serving | e2e (median) | 16 s | 10 s | 8 s |

<sub>Gold v2, n=36 (32 beantwortbar), span-basiert, `num_ctx` 16384 — Exp 10 in `eval/RESULTS.md`.
Eine früher hier berichtete Faithfulness von 0.667 war ein Artefakt: Ollama kürzte die Prompts
still auf 2050 Token, das Modell sah nur die letzten ~2 Passagen (s. Exp 7/10).</sub>

**Trade-off, den die Suite sichtbar macht:** Mehr Passagen verbessern das Retrieval nur bis
k=8, senken aber die Stützung der Antwort monoton — ein 7B-Modell verliert mit jeder weiteren
Passage Verankerung. Empfehlung: `TOP_K=6` (Exp 10).

**Kernbeobachtung:** ROUGE-L/F1 sind niedrig, obwohl die Antworten korrekt sind — sie messen nur
n-Gramm-**Oberflächenüberlappung** und bestrafen Paraphrasen. **Faithfulness (NLI)** ist für ein
generatives RAG-System das aussagekräftigere Qualitätsmaß (misst *Stützung*, nicht Wortgleichheit).

## Design-Entscheidungen

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

**Optional: Small-to-Big** (`SMALL_TO_BIG_ENABLED=true`): Die Suche bleibt auf den kleinen,
präzisen Chunks; die besten `S2B_TOP_N` (3) Treffer werden vor dem Prompt-Bau um ± `S2B_WINDOW` (1)
Nachbar-Chunks erweitert — per deterministischer Punkt-ID direkt aus Qdrant, Overlap exakt über
char-Offsets zusammengefügt, kein Re-Embedding. Exp 7: Die antwort-tragende Passage steht damit
deutlich öfter in Quelle [1] (MRR 0.474 → 0.540, kein Item schlechter), Kosten +36 % Prompt-Länge
(→ vorher `make ollama-ctx`, sonst schneidet der Runner den Prompt ab).

## Projektstruktur

```
backend/    FastAPI (routes/ · services/ · rag/ · ingestion/ · clients/ · db/)
frontend/   Angular (chat/ · groups/ · models/ · dashboard/ · shell/ · ui/ · services), nginx-Proxy
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

- ~~Gold-Set v2 mit Span-Labels~~ ✅ kuratiert (36 Items); alle Eval-Skripte messen span-basiert.
  Ausbau auf n≥50 bleibt sinnvoll (Exp 8 zeigt: 25 von 32 Items bewegen sich gar nicht).
- ~~Contextual Retrieval~~ ✅ gemessen + produktiv (Exp 5, stärkster Hebel: span-MRR +0.054).
- ~~Sparse-Hybrid (bge-m3)~~ ✅ gemessen — ehrliches Negativ-Ergebnis (Exp 6), nicht in der Pipeline.
- ~~Small-to-Big~~ ✅ gemessen (Exp 7, MRR 0.474→0.540), opt-in `SMALL_TO_BIG_ENABLED`.
- **Graph-RAG** (`backend/app/graph/`, opt-in `GRAPH_RAG_ENABLED`): Entity-Graph aus einer
  LLM-Extraktion je Chunk. **Exp 8 (lokal, Entity-Linking / Nachbar-Expansion): kein messbarer
  Effekt** auf der Span-Metrik (paired 4:3:25). **Exp 9 (global, Louvain-Communities +
  LLM-Berichte für thematische Fragen, eigenes Gold-Set + LLM-Judge + NLI): negativ** —
  Berichte aus einer 7B-Extraktion sind Entity-Listen statt Themen, die Antworten werden
  länger und schlechter zitiert, nicht besser; ein 7B degeneriert bei 9k-Token-Prompts.
  Beides bleibt opt-in im Code, das Flag bleibt aus. Hypothesen vorab, Zahlen und Interpretation
  in `eval/RESULTS.md`.
- Bewusst minimal (lokales Projekt): kein Auth/CORS/Rate-Limiting.
- ~~Reranker in die `/chat`-Pipeline integrieren~~ ✅ umgesetzt (opt-in, s. o.).
- ~~Ingestion-UI, Gruppen-Verwaltung, Modellwechsel-UI~~ ✅ umgesetzt.

## Lizenz

[MIT](LICENSE). Demo-Texte von [Project Gutenberg](https://www.gutenberg.org).
