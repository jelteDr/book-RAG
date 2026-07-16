# Plan: Lokale RAG-Anwendung für Buchtexte mit Zitierung & Monitoring (`book-rag`)

> **Status-Update (Juli 2026):** Dieser Plan ist das ursprüngliche Planungsdokument und wird
> bewusst nicht laufend umgeschrieben. Abweichungen der Umsetzung:
> - **Metadaten-Store ist PostgreSQL statt SQLite** (SQLModel/asyncpg; Gruppen, Bücher,
>   Modell-Registry, gespeicherte Unterhaltungen, query_log).
> - **M3 und M4 sind umgesetzt** (Upload-UI mit Dry-Run-Gate, Gruppen + Qdrant-Filter,
>   DELETE/PATCH, Modell-Auswahl, NLI-Faithfulness opt-in inline) — darüber hinaus:
>   Sammelgruppen (Collections), Dashboard, Chat-Persistenz mit Verlauf, Auto-Slug.
> - Aktueller Stand & Roadmap: siehe README („Grenzen & Roadmap") und `eval/RESULTS.md`.

## Kontext

Portfolio-Projekt für den Wechsel Richtung **Data Scientist / ML Engineer**. Die
Bachelorarbeit deckt die analytisch-statistische Seite ab (Transfer Learning /
Knowledge Distillation über 50k Bilder); dieses Projekt zeigt die
**Engineering- + angewandte-DS-Seite**: eine lokal lauffähige RAG-Anwendung mit
Ingestion-Pipeline, erklärbaren Antworten (Zitier-Marker) und **methodisch
sauberer Evaluation**.

Man lädt Buchtexte hoch (später Game of Thrones via Kaggle), stellt per Chat
Fragen, und das LLM antwortet mit **Zitier-Markern [n]**, die auf konkrete
Quell-Passagen mappen. Ein Serving-Benchmark (Ollama vs. llama-server) bleibt als
klar getrennter Bonus-Anhang erhalten.

Das Repo **existiert bereits** öffentlich auf GitHub: `jelteDr/book-RAG`
(`https://github.com/jelteDr/book-RAG.git`). Claude bereitet Git/Remote/Hooks vor;
den **Token hinterlegt der Nutzer selbst** (Claude fasst ihn nie an — Credential-Policy).

### Bestätigte Entscheidungen
- **Sichtbarkeit:** GitHub-Repo **sofort public** → strikte History-Hygiene ab Commit 1.
- **Frontend:** **Angular minimal** (Chat + Zitat-Chips + Modell-Dropdown). Monitoring/
  Modell-Vergleich im **Notebook**, nicht als Angular-Dashboard (spart Zeit, stärkt DS-Signal).
- **Scope-Schnittkante:** **Ende M2 = vollständiges, vorzeigbares Projekt**
  (1 sauberes Buch + Chat + korrekte [n]-Zitate + Demo). **M3–M5 sind optionale Stretch-Goals.**
- **Qualitäts-Metriken:** **alles lokal**; echte Faithfulness via **multilingualem
  NLI** (mDeBERTa-xnli o. ä.), kontrollierter Offline-Benchmark (temp=0, fixer Fragen-Satz).
- **LLM-Runtime:** Ollama (App) + llama-server (Benchmark, Bonus-Anhang). Beide OpenAI-kompatibel.
- **Embeddings:** `bge-m3` (multilingual, DE-Frage/EN-Text) über Ollama (kein torch).
- **Chat-Modell:** Default `qwen2.5:7b-instruct-q4_K_M`; MVP-Start `llama3.2:3b`.
- **Vektor-DB:** Qdrant (Docker). **Metadaten/Log:** SQLite.
- **Backend:** FastAPI, **Python 3.12** via uv. **Doku + Code-Kommentare: Deutsch.**
- **Projektort:** `~/LLM/book-RAG/` (self-contained Git-Repo; Remote existiert bereits).

## ⚠️ Sofort-public: Sicherheits- & Urheberrechts-Regeln (hart)

1. **Nie committen:** Buchtexte, SQLite-DB, Qdrant-Storage, Cleaning-Report-Text-Samples,
   Notebook-Zell-Outputs, `.env`, Modelle. Buchtext liegt **auch** in DB + Vektorstore
   → diese gehören ausschließlich in Docker-Volumes, nie ins Repo.
2. **Reihenfolge:** (1) `book-rag/` anlegen → (2) **vollständiges `.gitignore` als Commit 1**
   → (3) Pre-Commit-Hooks → (4) erst dann Code. Kein `git add` vor Schritt 2/3.
3. **Netze:** `gitleaks` (fängt PATs `ghp_`/`github_pat_`, Keys) + `nbstripout`
   (strippt Notebook-Outputs) als Pre-Commit-Hooks; **GitHub Push-Protection/Secret-Scanning aktivieren.**
4. **Öffentliches Demo-Set:** nur **ein Public-Domain-Gutenberg-Werk** (mit LICENSE + Attribution).
   GoT/Kaggle-Text nur lokal.

## GitHub-Setup (Token-sicher — Claude bereitet vor, Nutzer hinterlegt Token)

Repo existiert bereits public: `https://github.com/jelteDr/book-RAG.git`.

Claude führt aus (keine Credentials): `git init` in `book-RAG/`;
`git config credential.helper osxkeychain`; **repo-lokale Commit-Identität**
`user.name="jelteDr"`, `user.email="jeltedreetz@icloud.com"` (nicht die Firmen-Mail);
`git remote add origin https://github.com/jelteDr/book-RAG.git`; `git branch -M main`.

**Commit-Konventionen:** Commit-Messages auf **Deutsch**; **kein** Claude-Co-Author-Trailer
(ausdrücklicher Nutzerwunsch).

**Branching (Git Flow):** `main` = stabil; `develop` = Integrationsbranch (gilt als aktueller
Arbeitsstand); jedes Feature/Arbeitspaket auf `feature/<name>` **von `develop` abgezweigt** und
nach Fertigstellung mit `git merge --no-ff` zurück in `develop`. Releases: `develop` → `main`.

**Erster Commit (sicher, statt README-only):** `.gitignore` + `LICENSE` + `README.md` +
`.pre-commit-config.yaml` zusammen als Commit 1 — so ist der Ignore-Schutz aktiv, bevor Code/Daten kommen.

Manuelle Nutzerschritte:
- Erster `git push -u origin main`: Username `jelteDr`, **PAT als Passwort** → osxkeychain speichert ihn.
  (Falls das Remote schon einen initialen README-Commit hat: vorher `git pull --rebase origin main`.)
- **PAT-Scope:** `repo` (classic) bzw. `Contents: write` (fine-grained).
- **Anti-Patterns (nie):** Token in Remote-URL (`https://user:TOKEN@…` → Klartext in `.git/config`),
  Token in `.env` oder in einem Commit.

## Architektur

```
 [Container] Angular (minimal) ──REST + SSE──► [Container] FastAPI (Python 3.12)
   - Chat + [n]-Chips + Modell-Dropdown              │
   (Monitoring/Vergleich → Notebook)          ┌──────┼───────────┐
                                              ▼      ▼           ▼
                                          Ollama   Qdrant      SQLite
                                          (Host/    [Container] [Volume]
                                          Container) Vektoren+  Metadaten,
                                          chat+bge-m3 Payload    Log
   Notebooks (DS-Kern): Cleaning/Chunking-Tuning · Retrieval-Eval (Gold-Set, Recall@k)
                        · Faithfulness (NLI) · Modell-Benchmark (temp=0)
   Bonus-Anhang: llama-server Serving-Benchmark (TTFT/TPS vs. Concurrency)
```

### Tech-Stack
| Komponente | Wahl | Begründung |
|---|---|---|
| LLM-Runtime | Ollama, Chat via **OpenAI-compat `/v1`** | Streaming + `usage`-Tokens; `/api/tags` nur fürs Modell-Listing |
| Embeddings | Ollama `bge-m3` | multilingual (DE→EN), kein torch |
| Backend | FastAPI + httpx (async) | SSE-Streaming (TTFT), Pydantic |
| Vektor-DB | Qdrant (Docker), **Cosine** | Payload-Filter für Gruppen (Stretch) |
| Metadaten/Log | SQLite (Volume) | Zero-Ops |
| Frontend | **Angular minimal** hinter nginx | dein Angular-Wunsch, schlank gehalten |
| Faithfulness | lokales **NLI** (mDeBERTa-xnli) | echtes Entailment statt Cosine |
| Benchmark (Bonus) | llama-server `--metrics` | Serving-Kurven, getrennt vom RAG-Narrativ |

## Containerisierung (docker-compose, verifizierte Fixes)

- **Netzwerk-Fix (kritisch):** Host-Ollama lauscht nur auf `127.0.0.1` → Container erreicht es
  nicht. Lösung: Host-Ollama mit **`OLLAMA_HOST=0.0.0.0:11434`** starten (launchctl/App-Setting);
  im `backend`-Service **`extra_hosts: ["host.docker.internal:host-gateway"]`** (nötig für Linux,
  schadet macOS nicht). `dev` ist kein reines „clone & up" — Host-Prereqs (Ollama + Modelle) dokumentieren.
- **Profile:** `dev` (Ollama auf Host, Metal) Standard; `full` via **`docker-compose.full.yml`-Override**,
  das `OLLAMA_BASE_URL=http://ollama:11434` setzt (Profile ändern **keine** env-Vars!) + Named Volume
  `ollama_models` + Modell-Provisioning (`ollama pull …`) vor Backend-Start. `full` ist Doku-Fußnote
  (auf Mac CPU-only).
- **Robustheit:** Healthchecks je Service + `depends_on: condition: service_healthy`; Qdrant-Probe
  via TCP/`readyz` (kein curl im Image); **Images fest pinnen** (`qdrant/qdrant:vX`, `ollama/ollama:X`),
  kein `latest`.
- **Frontend↔Backend:** Prod-Build hinter **nginx mit `/api`-Reverse-Proxy** (löst CORS/Origin/fixe URL;
  `proxy_buffering off` für SSE). Alternativ `ng serve` + FastAPI-CORSMiddleware.
- **Ports** (in `.env.example` + README): frontend, backend 8000, qdrant 6333/6334, ollama 11434,
  llama-server 8080. Im `full`-Profil Port-Kollision Host-/Container-Ollama vermeiden.

## Repo-Struktur

```
book-RAG/
├── docker-compose.yml / docker-compose.full.yml   # dev / full-Override
├── .env.example                 # OLLAMA_BASE_URL, Modelle, Ports
├── .gitignore                   # vollständig (s. u.), Commit 1
├── .pre-commit-config.yaml      # gitleaks + nbstripout
├── LICENSE                      # MIT (+ Gutenberg-Attribution im README)
├── README.md                    # DE, "mini-thesis" + Demo-GIF/Screenshots + GitHub-Setup
├── docs/PLAN.md                 # dieser Plan
├── data/                        # GITIGNORED (Buchtexte, nur lokal)
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml           # uv, Python 3.12
│   └── app/
│       ├── main.py              # FastAPI + SSE
│       ├── routes/              # chat.py, ingest.py, models.py, (groups.py, monitoring.py = Stretch)
│       ├── ingestion/           # parser, cleaner(+report), chunker, metadata
│       ├── rag/                 # retriever, prompt_builder, citations, quality
│       ├── clients/             # ollama_client.py, qdrant_client.py
│       └── db/                  # sqlite schema/models
├── frontend/                    # Angular minimal (Chat + Chips + Modell-Dropdown) + nginx + Dockerfile
├── notebooks/                   # 01_cleaning_chunking, 02_retrieval_eval, 03_faithfulness, 04_benchmark
├── eval/                        # Gold-Eval-Set (Fragen→chunk_ids/Gold-Antworten)
├── benchmark/                   # aus ~/LLM/src übernommen (parametrisiert)
└── scripts/                     # llama-server install/serve/download (Bonus)
```

**`.gitignore` (vollständig):** `data/`, `models/`, `*.gguf`, `*.bin`, `*.db`, `*.sqlite*`,
`qdrant_storage/`, `.env`, `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.ruff_cache/`,
`.ipynb_checkpoints/`, `node_modules/`, `dist/`, `.angular/`, `.DS_Store`.

## Datenfluss

**Ingestion:** Upload → Encoding-Sniffing (`charset-normalizer`) → Metadaten
(Gutenberg-Header-Parse, Boilerplate strippen) → **Textbereinigung + Dry-Run-Prüf-Gate**
(s. u.) → strukturbewusstes Chunking → Embeddings (`bge-m3`, L2-normalisiert) → **Qdrant-Upsert
(Payload = anzeigbarer Chunk-Text = Source-of-Truth); SQLite nur Metadaten/Log** →
Ingestion-Status pro Buch (`pending`→`committed` erst nach vollständig verifiziertem Upsert;
bei Fehler Cleanup der Teil-Vektoren via `delete by book_id`). Idempotent (Datei-Hash).

**Query:** Frage → Embed → Qdrant-Search (top_k≈8, Score) → Prompt mit nummerierten Blöcken
`[1..k]` → Ollama Chat `stream=True` → SSE (TTFT) → Citation-Mapping → Log. **Fehlerpfade:**
Ollama/Qdrant nicht erreichbar → sauberer 5xx + Frontend-Meldung; httpx Timeout/Retry;
leeres Retrieval → definierte „nichts gefunden"-Antwort.

## Textbereinigung & Prüf-Gate (`ingestion/cleaner.py`)

Deterministische Pipeline: (1) Unicode-NFC, Steuer-/Zero-Width-Zeichen raus; (2) Zeichen-Norm
(Smart-Quotes/Dashes/Ligaturen); (3) **De-Wrapping** (harte Umbrüche mergen, De-Hyphenation,
echte Absätze erhalten); (4) Artefakte raus (Seitenzahlen, Kopf-/Fußzeilen per Häufigkeit,
Boilerplate). **Prüf-Gate:** Dry-Run erzeugt **Cleaning-Report** (entfernte Zeichen/Zeilen,
Vorher/Nachher-Sample **gekürzt, nur in gitignore-Pfad**), kein Upsert; erst nach Bestätigung
Commit. **Guardrails:** Zeichen-Whitelist; Abbruch bei zu hohem verworfenem Anteil.
**Wichtig:** `char_start/end` werden auf dem **bereinigten** Text berechnet (De-Wrapping
verschiebt Offsets → sonst zeigt der Chip die falsche Stelle).

## Zitier-Marker & Grounding (Widerspruch aufgelöst)

Ehrlich als **„Grounding via Prompt-Contract"** (keine mechanistische Attribution — Logprobs/
Attention über Ollama nicht verfügbar).
- **Streaming beibehalten:** Prosa mit **inline `[n]`-Markern streamen** (TTFT bleibt messbar).
  **Kein `format:json`** (würde JSON-Syntax streamen → Puffern bis Ende → TTFT≈e2e, Streaming tot).
- **Mapping serverseitig:** `[n]` ist nur Index in die bekannte Kontext-Reihenfolge; Modell rät
  **nie** IDs; halluzinierte `n>k` verwerfen.
- **UX:** `[n]` als klickbarer Chip → Chunk-Text + Score; retrievte-nicht-zitierte Chunks separat.

## Retrieval-Korrektheit (neu, verifizierbar)

- `bge-m3` über Ollama = **Dense-only** (kein Sparse/ColBERT) → schwächer bei Eigennamen (GoT!).
- Zwingend: Embeddings **L2-normalisiert**, Qdrant-Collection auf **Cosine**, **kein**
  Query-Instruction-Prefix (bge-m3 braucht keinen).
- Chunk-Größe an **bge-m3/XLM-R-Tokenizer** messen (≠ qwen2.5/tiktoken).
- Optional (Stretch): Hybrid (Qdrant Sparse/BM25 + RRF) als **gemessener** Trade-off.

## Monitoring & Qualität (methodisch sauber, lokal)

- **Live-Log (SQLite):** pro Query `question, answer, model, retrieved/cited_chunk_ids, ttft_ms,
  e2e_ms, tps, prompt/completion_tokens`. **In M0/M1 verifizieren, dass `/v1` streaming die
  `usage`-Tokens liefert** — sonst Fallback lokale Tokenizer-Zählung (sonst bricht TPS/Token-Logging).
  Live-Zahlen sind **deskriptive Produktions-Telemetrie**, ausdrücklich **kein** Modell-Benchmark.
- **Faithfulness (lokal, valide):** multilinguales **NLI** (mDeBERTa-xnli) prüft, ob zitierter
  Chunk die Aussage **stützt** (Entailment) — nicht Cosine. Cosine nur als „semantische Überlappung"
  gelabelt. Zusätzlich **Citation-Präzision** (stützt der Chunk die Aussage?) statt nur -Coverage.
- **Gold-Eval-Set (DS-Kern, `eval/`):** 20–50 handgelabelte Fragen → relevante `chunk_ids`/Gold-Antworten.
  Notebook berichtet **Recall@k, MRR/nDCG**. **Ein kontrolliertes Before/After-Experiment** mit Chart
  (z. B. Recall/Faithfulness **mit vs. ohne Cleaning**, oder Chunk-Size-Sweep). Das ist der stärkste
  Interview-Beleg („X gemessen, um Y % verbessert").
- **Modell-Benchmark (Notebook, fair):** fixer Fragen-Satz, **temperature=0 + fixer Seed**,
  **Cold- vs. Warm-TTFT getrennt**, cross-tokenizer Token-Kosten nur mit Vorsicht vergleichen.

## Toolchain & Setup

- **uv installieren** (Homebrew/offizielles Skript) + `uv python install 3.12` (System-Default ist
  3.14 → genau das Wheel-Problem, das wir umgehen).
- **Node/Angular:** lokales Node ist v26 (von Angular nicht unterstützt) → im Frontend-Container
  **Node-LTS pinnen** (z. B. `node:22-slim`), lokales Node irrelevant; Package-Manager festlegen.
- **LICENSE** (MIT) + Gutenberg-Attribution im README (ohne LICENSE = „all rights reserved").

## Backend-Endpunkte & Lifecycle
- Kern (M1–M2): `POST /chat` (SSE), `POST /ingest` (Dry-Run/Commit), `GET /models`, `GET /health`.
- Stretch (M3+): `DELETE /books/{id}` + `/groups/{id}` (Qdrant+SQLite atomar; nötig für Re-Ingestion/
  Copyright-Cleanup), `PATCH` Metadaten mit Payload-Sync, Gruppen-Zusammenfassungen.

## Wiederverwendung des bestehenden Scaffolds
- `~/LLM/src/benchmark.py` → `book-RAG/benchmark/` übernehmen; `BASE_URL`/`model` als **CLI-Args**;
  Token-Zählung auf `usage.completion_tokens` (statt `'"content"'`-Heuristik). **Nur Quellcode
  kopieren — kein `.venv/`, `models/`, `__pycache__`.**
- `~/LLM/scripts/*.sh` übernehmen (llama-server-Bonus).

## Milestones (M2 = fertiges Projekt; M3–M5 optional)

- **M0 — Setup & Sicherheit:** `book-RAG/` anlegen, vollständiges `.gitignore` (Commit 1),
  Pre-Commit-Hooks (gitleaks/nbstripout), LICENSE, Git-Identität + Remote (Token bleibt Nutzersache),
  uv + Python 3.12, Ollama-Prereqs (`OLLAMA_HOST=0.0.0.0`, `llama3.2:3b` + `bge-m3` ziehen), Qdrant via
  compose, Benchmark-Scaffold parametrisieren, Smoke-Test gegen Ollama `/v1` (usage-Tokens verifizieren).
- **M1 — Vertical Slice:** **sauberes Gutenberg-Sample** (nicht roh!) ingesten
  (parse→clean→chunk→embed→Qdrant), `POST /chat` mit Retrieval + SSE, **Angular-Minimal-Chat**;
  TTFT in der UI sichtbar. Ziel: eine Frage Ende-zu-Ende.
- **M2 — Zitate + Demo (= SHIPPABLE):** inline `[n]`-Streaming + serverseitiges Mapping, klickbare
  Chips + Score; „nicht in den Quellen"-Fall; **Gold-Eval-Set + Recall@k + 1 Before/After-Chart**;
  **2–3-min Screencast/GIF + Screenshots ins README**; README „mini-thesis" fertig. → **Vollständiges Portfolio-Stück.**
- **M3 (optional) — Ingestion-UI & Gruppen:** ✅ umgesetzt — Upload-UI, Cleaning-Gate im UI,
  Gruppen + Qdrant-Filter, DELETE/PATCH (Kaggle-Adapter entfiel: Upload deckt es ab).
- **M4 (optional) — Modellwechsel & Faithfulness:** ✅ umgesetzt — Modell-Auswahl im Chat,
  NLI-Faithfulness inline (opt-in); Modell-Benchmark-Notebook offen.
- **M5 (optional) — Bonus:** llama-server Serving-Benchmark-Kurven, `full`-Compose-Profil.

## Risiken & Mitigationen
1. **Sofort-public + Copyright** → `.gitignore` Commit 1, gitleaks/nbstripout, Push-Protection; nie DB/Vektorstore/Buchtext committen.
2. **Token-Leak** → Keychain, nie Token in URL/.env/Commit; PAT-Scope `repo`.
3. **Python 3.14 Wheels** → uv + 3.12; Embeddings über Ollama (kein torch).
4. **dev-Profil Ollama unreachable** → `OLLAMA_HOST=0.0.0.0` + `extra_hosts host-gateway`.
5. **Streaming↔JSON-Konflikt** → inline `[n]`-Prosa streamen, kein `format:json`.
6. **DS-Metrik-Validität** → NLI statt Cosine; kontrollierter Offline-Benchmark (temp=0, Gold-Set).
7. **Falsche Zitat-Position** → Offsets auf bereinigtem Text.
8. **Scope-Tod bei M3** → M2 als definierte Fertig-Grenze; M3–M5 optional.
9. **usage-Tokens fehlen** → früh verifizieren, Fallback lokale Zählung.
10. **Node/Angular-Version** → Node-LTS im Container pinnen.

## Verifikation (End-to-End)
- **M0:** `benchmark/smoke_test.py --base-url http://localhost:11434/v1` liefert Antwort **mit
  `usage`-Tokens**; Qdrant `readyz` grün; `git log` zeigt `.gitignore` als Commit 1; `pre-commit run --all-files`
  grün; **Test-Commit mit Fake-PAT wird von gitleaks blockiert**.
- **M1:** Gutenberg-Sample ingesten, im Angular-Chat faktische Frage → korrekte gestreamte Antwort, TTFT sichtbar.
- **M2:** Antwort enthält `[n]`-Chips; Klick zeigt korrekten Chunk + Score; erfundene Frage → „nicht in den Quellen".
  Notebook zeigt Recall@k + 1 Before/After-Chart. README hat Demo-GIF. Repo public, History enthält **keinen** Buchtext (`git log -p | grep`-Stichprobe).
- **M3–M5 (optional):** Upload+Cleaning-Report im UI; Modellwechsel im Dropdown; NLI-Faithfulness geloggt;
  Benchmark-Kurve Ollama vs. llama-server.
