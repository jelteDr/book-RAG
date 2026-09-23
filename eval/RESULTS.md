# Evaluations-Ergebnisse

Reproduzierbar via `make eval` (Baseline) bzw. den Experiment-Skripten in `eval/`.
Korpus: Dracula (581 Chunks), Embeddings `bge-m3`, Qdrant Cosine. Relevanz
**kapitel-basiert** (Gold-Set `eval/gold_dracula.jsonl`, inzwischen **n=21** Fragen;
die frühen Experimente unten liefen noch gegen n=10 — jeweils am Experiment vermerkt).

> **Metrik-Hinweis (Ehrlichkeit):** Was hier „Recall@k" heißt, ist faktisch
> **Hit-Rate@k / Success@k** — der Wert ist 1.0, sobald *irgendein* relevantes
> Kapitel in Top-k liegt (nicht der Deckungsgrad über alle Gold-Kapitel). Bei
> Multi-Kapitel-Fragen überschätzt das die Güte. Der Vergleich zwischen Methoden
> bleibt fair (für alle identisch berechnet).
>
> **Statistik-Hinweis:** n=10 ist ein exploratives **Dev-Set ohne Signifikanz**.
> Jede Frage bewegt Hit-Rate um 0.10; MRR-Standardfehler ~0.13. Aussagen sind
> Trends, keine belastbaren Rankings. Für belastbare Schlüsse: Gold-Set auf
> n≥30–50 vergrößern + Bootstrap-CI / paired Test.

## Baseline — dense (bge-m3), deutsche Fragen, k=8

| Metrik | Wert |
|---|---|
| Hit-Rate@1 | 0.40 |
| Hit-Rate@3 | 0.60 |
| Hit-Rate@5 | 0.80 |
| Hit-Rate@8 | 0.80 |
| MRR | 0.535 |

Misses (kein relevantes Kapitel in Top-8): **d01** (Harkers Reisegrund),
**d09** (Vampir-Abwehrmittel).

## Experiment 1 — Frage DE vs. EN (`eval/lang_experiment.py`)

**Hypothese:** Die mäßige Baseline liegt am cross-lingualen Gap (DE-Frage → EN-Text).

| Metrik | DE | EN | Δ |
|---|---|---|---|
| Hit-Rate@1 | 0.40 | 0.20 | −0.20 |
| Hit-Rate@3 | 0.60 | 0.70 | +0.10 |
| Hit-Rate@5 | 0.80 | 0.80 | 0.00 |
| Hit-Rate@8 | 0.80 | **1.00** | +0.20 |
| MRR | 0.517 | 0.450 | −0.067 |

![DE vs EN](../results/lang_experiment.png)

**Ergebnis — Hypothese teilweise widerlegt:** `bge-m3` ist echt multilingual; DE
rankt den Top-Treffer sogar besser (Hit-Rate@1, MRR). EN hilft nur im Tail (die
Misses d01/d09 tauchen auf, aber erst auf Rang 8). → **Query-Übersetzung ist kein Fix.**

## Experiment 2 — Dense vs. BM25 vs. Hybrid/RRF (`eval/hybrid_experiment.py`)

**Idee:** BM25 (lexikalisch) matcht Eigennamen direkt; per Reciprocal Rank Fusion
(RRF, k=60, CAND=30) mit Dense kombiniert. Adversariell verifiziert (Panel aus 4
Skeptikern, Zahlen exakt reproduziert).

| Metrik | dense | bm25 | hybrid |
|---|---|---|---|
| Hit-Rate@1 | 0.40 | 0.20 | 0.30 |
| Hit-Rate@3 | 0.60 | 0.30 | 0.60 |
| Hit-Rate@5 | 0.80 | 0.50 | 0.60 |
| Hit-Rate@8 | 0.80 | 0.60 | 0.80 |
| MRR | 0.535 | 0.331 | 0.467 |

![dense vs bm25 vs hybrid](../results/hybrid_experiment.png)

**Ehrliches Ergebnis (nach Verifikation):**
- **Naives RRF-Hybrid bringt bei n=10 KEINEN messbaren Netto-Vorteil** gegenüber
  Dense. Die gepaarte Reciprocal-Rank-Bilanz Dense↔Hybrid ist **4:4** (praktisch
  ein Münzwurf). Hybrid *rettet* einzelne Dense-Misses (d01: 10→6, d09: 12→1),
  *verschlechtert* aber einzelne Dense-Top-1-Treffer (d02: 4→16, d08: 1→9) —
  jeweils ~2 Fragen, nicht generalisierbar.
- **BM25 solo liegt durchgängig am niedrigsten** — aber im **Cross-Lingual-Handicap**
  (deutsche Frage gegen englischen Korpus). Das ist *kein* fairer BM25-Ceiling;
  BM25 gewinnt nur dort, wo seltene Eigennamen fallen (d03 „Demeter/Whitby", d06 „Westenra").
- „Dense ist am besten" wäre **überzogen** (bei n=10 nicht von Hybrid unterscheidbar).

**Struktureller Hinweis:** RRF fusioniert auf **Chunk**-Ebene, die Metrik ist
**Kapitel**-basiert — Dense und BM25 treffen dasselbe relevante Kapitel oft über
*verschiedene* Chunks, daher greift die RRF-Verstärkung kaum. Die „Verwässerung"
ist teils ein struktureller Effekt der Chunk-Fusion, nicht nur der Gleichgewichtung.

## Experiment 3 — Cross-Encoder-Reranker (`eval/reranker_experiment.py`)

Gold-Set auf **n=21** erweitert (Labels im Quelltext verankert). Dense holt 30
Kandidaten, `bge-reranker-v2-m3` (multilingual) sortiert neu.

| Metrik | dense | + Reranker | Δ |
|---|---|---|---|
| Hit-Rate@1 | 0.43 | **0.48** | +0.05 |
| Hit-Rate@3 | 0.67 | 0.67 | 0.00 |
| Hit-Rate@5 | 0.76 | 0.71 | −0.05 |
| Hit-Rate@8 | 0.81 | 0.76 | −0.05 |
| MRR | 0.567 | **0.602** | +0.035 |

![dense vs reranker](../results/reranker_experiment.png)

**Ergebnis:** Der Reranker verbessert **Precision@1 und MRR** (bringt einen Treffer
nach vorn) — genau seine erwartete Stärke — verschlechtert aber die **Tail-Recall**
(@5/@8). Gepaarte Erst-Treffer-Bilanz 4:7 (dense), aber der Reranker gewinnt deutlicher,
wo er gewinnt (d21 15→2, d10 3→1, d06 4→1).

**Caveat (wichtig):** Die Metrik ist **kapitel-basiert**, der Cross-Encoder arbeitet
**passagen-basiert**. Er stuft „richtiges Kapitel, aber thematisch daneben"-Chunks
korrekt herab — die Kapitel-Metrik zählt das als Verlust (z. B. d02 4→26). Der Reranker
wird dadurch systematisch **unterbewertet**; für RAG zählt „bester Chunk zuerst"
(Precision@1/MRR), und das verbessert er. Faire Bewertung braucht **passagen-basierte
Gold-Labels**.

## Metrik-Suite — Antwortqualität + Serving (`eval/answer_eval.py`)

Pro Gold-Frage (n=21) wird die volle RAG-Pipeline ausgeführt (Modell `qwen2.5:7b`, k=8):

| Kategorie | Metrik | Wert (mean) |
|---|---|---|
| Antwort | ROUGE-L | 0.124 |
| Antwort | Antwort-Token-F1 | 0.153 |
| Antwort | **Faithfulness (NLI, mDeBERTa-xnli)** | **0.667** |
| Retrieval | Recall@8 | 0.770 |
| Retrieval | MRR | 0.556 |
| Serving | TTFT (median) | ~7.4 s\* |
| Serving | TPS (median) | 11.8 tok/s |
| Serving | e2e (median) | ~19 s |

<sub>\* TTFT im Batch-Lauf durch Speicherdruck/Modell-Reloads (24 GB) erhöht; interaktiv/warm ~0,2–4 s.</sub>

**Interpretation:**
- **ROUGE-L/F1 sind niedrig (~0.12–0.15), obwohl die Antworten korrekt sind** — sie messen nur
  n-Gramm-Überlappung mit einer kurzen Referenzantwort und bestrafen Paraphrasen. Für generatives
  RAG sind sie deshalb nur schwache Signale (nützlich als *relativer* Vergleich zwischen Modellen).
- **Faithfulness (NLI) = 0.667** ist das aussagekräftigere Maß: 2/3 der Antwort-Aussagen werden von
  den abgerufenen Passagen *gestützt* (Entailment). Die Lücke zeigt Raum nach oben (Retrieval-Gaps
  oder ungestützte Modell-Zusätze) — bewusst statt PPL gewählt, das nur Fluenz misst.
- Roh-Ergebnisse pro Frage: `results/answer_eval.json`.

## Exp 4 — Overlap-Dedup im Retrieval (`eval/dedup_experiment.py`)

**Frage:** Der Chunker arbeitet mit ~200 Zeichen Overlap — wie oft belegen dadurch
*dieselben* Passagen mehrere Top-k-Plätze, und was bringt Deduplizierung?

**Setup:** paired auf denselben 21 Fragen (Gruppe `Horror`, k=8, kapitel-basierte
Metrik wie Exp 1–3). Arme: `baseline` (Suche wie bisher) vs. `dedup` (k+8 Kandidaten
holen, überlappende Zeichenbereiche desselben Buchs deduplizieren, auf k kürzen).

| Messgröße | baseline | dedup |
|---|---|---|
| Fragen mit ≥1 Duplikat in Top-8 | 16/21 (Ø 1,33) | 0 |
| Ø verschiedene Kapitel in Top-8 | 4,71 | **5,52** |
| Hit-Rate@8 | 0,810 | **0,857** |
| MRR | 0,556 | **0,565** |

Gepaarte MRR-Bilanz: **2 besser / 0 schlechter / 19 gleich** — ein seltener Fall
ohne Downside: Duplikate raus = mehr *verschiedene* Information im Prompt, und in
2 Fällen rückt dadurch ein relevantes Kapitel neu in die Top-8. Der Haupteffekt
(mehr nutzbarer Kontext fürs LLM) liegt außerhalb dieser Retrieval-Metrik und
sollte sich in der Antwortqualität zeigen. **Konsequenz: Dedup ist in
`app/rag/retriever.py` produktiv aktiv** (Puffer `DEDUP_EXTRA=8`).

## Exp 5 — Contextual Retrieval (`eval/contextual_experiment.py`)

**Idee** (nach Anthropics „Contextual Retrieval"): Ein Chunk mitten aus Kapitel 12
weiß nicht, dass „he" Jonathan Harker ist. Beim Ingest generiert `qwen2.5:7b`
(temp=0) deshalb 1–2 Sätze Kontext pro Chunk (Figuren benannt, Pronomen aufgelöst,
Ort/Geschehen), die **nur ins Embedding** eingehen — der anzeigbare Text bleibt
unverändert. Einmalkosten: 581 LLM-Aufrufe (~1 h lokal), Kontexte gecheckpointet
in `results/contextual_contexts.jsonl`, Embeddings in separater Collection.

**Paired** auf denselben 21 Fragen (k=8, kapitel-basiert, dense vs. dense+Kontext,
beide ohne Dedup — sauberer Einzelvergleich):

| Metrik | baseline | + Kontext | Δ |
|---|---|---|---|
| Hit-Rate@8 | 0.810 | **0.952** | +0.14 |
| MRR | 0.556 | **0.738** | **+0.182** |

Gepaarte MRR-Bilanz: **10 besser / 2 schlechter / 9 gleich** (Vorzeichentest
p≈0.04 — als einziges Experiment bisher auch bei n=21 nominell signifikant).
**Beide hartnäckigen Alt-Misses werden gerettet:** d01 (Harkers Reisegrund,
MISS→Rang 2) und d13 (MISS→Rang 1); d14 springt von Rang 6 auf 1. Kosten:
2 Regressionen (d05 1→3, d20 2→8) — vermutlich Kontext-Halluzinationen einzelner
Chunks (der LLM-Kontext ist nicht fehlerfrei, siehe Checkpoint-Datei).

**Einordnung:** Der mit Abstand stärkste bisher gemessene Einzelhebel (+0.18 MRR;
zum Vergleich Reranker +0.035, Dedup +0.009). **Empfohlener nächster Schritt:**
Kontext-Generierung als opt-in-Flag in die Ingestion-Pipeline (`CONTEXTUAL_INGEST_ENABLED`)
und Re-Ingest; vorher mit Gold-Set v2 (Span-Metrik) gegenprüfen.

## Gold-Set v2 — Span-Labels + Gegenprüfung (`eval/gold_v2.jsonl`)

**Das Set:** 36 Items (26 fact / 3 paraphrase / 3 multi / 4 unanswerable). Labels sind
**Text-Spans** (`book_id` + `char_start`/`char_end` im bereinigten Text) statt Kapitel —
jede Passage wurde beim Kuratieren gelesen und gegen die Soll-Antwort verifiziert
(Workflow aus `notebooks/gold_set_v2.ipynb`). Die 21 v1-Items wurden übernommen und
mit Spans nachgerüstet; ihre Kapitel-Labels bleiben als Vergleichs-Feld erhalten.
Die Datei enthält **keinen Buchtext** (nur eigene Formulierungen + Offsets) und ist
deshalb committbar. Bemerkenswert: Bei mehreren Items (z. B. d09) liegen die echten
Antwort-Passagen **außerhalb** der alten Kapitel-Labels.

**Befund 1 — die Kapitel-Metrik überschätzt massiv.** Dieselben Retrieval-Läufe
(Live-Index: contextual + dedup, n=32 beantwortbare Items, k=8), zwei Maßstäbe:

| Metrik | kapitel-basiert | span-basiert |
|---|---|---|
| Hit-Rate@1 | 0.50 | **0.31** |
| Hit-Rate@8 | 0.88 | **0.81** |
| MRR | 0.636 | **0.474** |

Ø MRR-Differenz **+0.162** — „richtiges Kapitel, falsche Passage" zählte bisher als
Treffer. Alle früheren Absolutwerte sind entsprechend zu lesen.

**Befund 2 — Gegenprüfung Exp 5 (Contextual) mit fairer Metrik.** Paired plain vs.
contextual (rohe dense-Suche ohne Dedup, Span-Metrik, n=32):

| Metrik | plain | + Kontext | Δ |
|---|---|---|---|
| Hit-Rate@8 | 0.688 | **0.812** | +0.12 |
| MRR | 0.408 | **0.462** | +0.054 |

Gepaarte Bilanz 12 besser / 8 schlechter / 12 gleich. **Der Contextual-Gewinn ist
real, aber deutlich kleiner als die Kapitel-Metrik suggerierte** (+0.054 statt
+0.182 MRR; bei n=32 nicht signifikant). Belastbar ist vor allem der Hit-Rate@8-Gewinn:
4 zusätzliche Fragen bekommen eine antwort-tragende Passage in die Top-8. Ein Teil des
Exp-5-Effekts war also „landet öfter irgendwo im richtigen Kapitel" — genau die Sorte
Verzerrung, für deren Aufdeckung das v2-Set gebaut wurde.

## Exp 6 — Sparse-Hybrid mit bge-m3-Sparse-Gewichten (`eval/sparse_hybrid_experiment.py`)

**Idee:** Exp 2 zeigte, dass klassisches BM25 cross-lingual (DE-Frage/EN-Text)
gehandicapt ist. bge-m3 liefert neben Dense auch **gelernte lexikalische Gewichte**
über sein multilinguales Subword-Vokabular — die Hoffnung: lexikalische Präzision
ohne den harten Sprachbruch. Gemessen gegen den Live-Index (Horror = contextual),
die eigentliche Frage: bringt Sparse ZUSÄTZLICH zu Contextual-Dense noch etwas?

**Setup:** Gold-Set v2 (n=32, span-basiert, k=8), Arme paired auf denselben Fragen:
dense (Live-Pfad), sparse (FlagEmbedding-Lexical-Weights, CPU), RRF-Fusion (k=60,
je 50 Kandidaten), sparse_en (englische Frage; n=21 Items mit `question_en`).

| Arm | Hit@1 | Hit@8 | MRR |
|---|---|---|---|
| dense (contextual) | **0.31** | **0.81** | **0.462** |
| sparse | 0.12 | 0.47 | 0.218 |
| hybrid RRF | 0.31 | 0.59 | 0.416 |
| sparse_en (n=21) | 0.19 | 0.52 | 0.332 |

Paired RRF vs. dense: 7 besser / **12 schlechter** / 13 gleich.

**Befund — ehrliches Negativ-Ergebnis:** Der Hybrid ist netto eine
**Verschlechterung**: RRF gewichtet Chunks hoch, die in beiden Listen auftauchen,
und drückt damit dense-only-Treffer aus den Top-8 (Hit@8 0.81 → 0.59). Der
sparse_en-Arm zeigt zudem: bge-m3-Sparse ist auf diesen Daten nicht nur
cross-lingual gehandicapt, sondern generell schwach (0.332 auch monolingual) —
Contextual-Dense ist schlicht der stärkere Kanal. Konsistent mit Exp 2 (naives
RRF ohne Netto-Vorteil). **Entscheidung: kein Sparse-Hybrid in der Pipeline;**
denkbares Follow-up wäre eine dense-dominierte gewichtete Fusion oder
bge-m3-ColBERT-Multivektoren, Priorität aber niedrig.

Hinweis Reproduktion: Der Sparse-Gewichte-Cache
(`results/sparse_weights_<group>.jsonl`) ist ein Buchtext-Derivat und bleibt
gitignored; das Skript baut ihn bei Bedarf neu (~10 min CPU für 581 Chunks).

## Exp 7 — Small-to-Big (`eval/small_to_big_experiment.py`)

**Idee:** Die Suche bleibt auf den kleinen, präzisen Chunks; erst NACH dem Retrieval
werden die besten `top_n` Treffer um ihre Nachbar-Chunks (± `window`) erweitert —
über die deterministischen Punkt-IDs direkt aus Qdrant, die Overlap-Zeichen werden
per char-Offset exakt zusammengefügt (`app/rag/expander.py`). Kein Re-Embedding.
Opt-in via `SMALL_TO_BIG_ENABLED` (+ `S2B_WINDOW`, `S2B_TOP_N`).

**Messung:** Small-to-Big ändert nicht, WAS gefunden wird, sondern was das Modell
davon LIEST. Gemessen wird daher die Span-Abdeckung des Prompt-Kontexts (Gold-Set v2,
n=32, k=8, Live-Index Horror, paired — beide Arme nutzen dieselben Suchergebnisse):

| Arm | Kontext-Hit@8 | MRR (erster abdeckender Treffer) | Ø Prompt-Kontext |
|---|---|---|---|
| plain | 0.81 | 0.474 | 18.433 Zeichen |
| expanded (window=1, top_n=3) | 0.81 | **0.540** | 25.036 Zeichen (+36 %) |
| expanded (window=2, top_n=3) | 0.81 | **0.566** | 31.065 Zeichen (+69 %) |

Paired: window=1 → 4 besser / 0 schlechter / 28 gleich; window=2 → 6/0/26.
**Kein einziges Item verschlechtert sich** — erwartbar, denn Erweiterung kann
Abdeckung nur hinzufügen; der zweite Dedup-Durchlauf verhindert, dass erweiterte
Fenster andere Treffer duplizieren. Hit@8 bleibt gleich, weil komplette MISSes
(Passage nicht in den Top-8-Nachbarschaften) nicht zu retten sind.

**Interpretation:** Die antwort-tragende Passage steht deutlich öfter in Quelle [1]
(plain 11/32 → window=1 15/32) — das Modell liest die richtige Stelle also früher
und im Zusammenhang. Kosten: +36 % Prompt (window=1) bzw. +69 % (window=2).
Empfehlung: window=1/top_n=3 aktivieren, window=2 nur wenn das Kontextfenster-Problem
(s. u.) behoben ist.

**⚠ Nebenbefund — Ollama schneidet unsere Prompts ab:** Das Server-Log
(`~/.ollama/logs/server.log`, 2026-07-16) zeigt mehrfach
`truncating input prompt limit=2050 prompt=4168..6037 keep=4`: Der Runner lief ohne
`num_ctx`-Konfiguration (Default 4096, zeitweise 2 Slots à ~2048) — bei unseren
~4,6k-Token-Prompts fehlten dem Modell System-Prompt und die ersten Quellen
komplett. Das dürfte einen Teil der niedrigen Faithfulness (0.667) und der
Zitat-Ausfälle erklären. Fix: `OLLAMA_CONTEXT_LENGTH=16384` (make-Target
`ollama-ctx`), danach Metrik-Suite neu laufen lassen.

## Exp 8 — Graph-RAG lokal (`eval/graph_experiment.py`)

**Idee:** Je Chunk extrahiert das Chat-LLM (qwen2.5:7b, temp 0, JSON-Modus) bis zu 8
Entities (PERSON/PLACE/ORGANISATION/OBJECT/EVENT) und Relationen; daraus entsteht ein
Entity-Graph (`backend/app/graph/`, Knoten = normalisierte Entities mit Aliasen, Kanten =
Ko-Erwähnung, `idf = ln(N/n_mentions)` dämpft Hubs wie Dracula). Beim Retrieval wird die
**deutsche** Frage per bge-m3 gegen die **englischen** Entity-Embeddings verlinkt
(kein String-Match), 1-Hop-Nachbarn gedämpft aktiviert, und jeder Chunk bekommt ein
Graph-Signal `g(c) = Σ a(e)·idf(e)`. Fusion **dense-erhaltend**:
`s = cos(q,c) + α·g̃(c)` — bei α=0 exakt die Baseline (Sanity-Check im Skript). Bewusst
kein RRF (Exp 2/6: Rang-Fusion verdrängte dense-Treffer).

Arme: `dense` (Baseline, contextual + dedup), `link` (Entity-Linking), `expand`
(dense-first, Seeds aus Top-3, Ko-Erwähnung ≥ 2), `graph_only` (Diagnose ohne dense).

**Hypothese (vorab festgelegt, vor dem Messlauf committet):**
- H1: Entity-Linking holt Passagen in die Top-8, die die gefragten Figuren/Orte nur
  beiläufig erwähnen (dense rankt sie tief) → Hit@8 ↑, MRR mindestens gleich.
- H2: Der Effekt konzentriert sich auf `multi`/`paraphrase` (n09 Demeter, n10 Erdkisten,
  n11 Minas Hilfe: Passagen über mehrere Kapitel, wenig lexikalische Überlappung).
- **Erfolg:** paired besser ≥ 2·schlechter **und** ΔMRR ≥ +0.03 **und** Hit@8 nicht
  schlechter. **Negativ:** schlechter ≥ besser oder Hit@8 sinkt. Dazwischen: „kein
  messbarer Effekt bei n=32". Hyperparameter a priori: α=0.03, m=5, τ=0.45; die
  Sensitivität (α∈{0.02,0.05}, τ∈{0.40,0.50}) ist **post hoc** und dient nur der
  Einordnung, nicht der Auswahl (kein Dev-Split bei n=32).

**Setup:** Gold v2, n=32 beantwortbar, k=8, Gruppe Horror (Live-Index contextual+dedup),
paired. Extraktion: 581 Aufrufe, 0 % JSON-Fehler nach Umstellung auf kompaktes JSON
(max_tokens 600; Pilot v1 mit 450 schnitt 40 % ab), Ø 6,2 Entities + 4,2 Relationen je
Chunk. Graph: 446 Knoten (148 PERSON, 181 PLACE), 606 Kanten; 303 kleingeschriebene
Gattungsbegriffe verworfen, 16 Aliase gemergt, 20 ambig belassen („Mina" ≠ „Mina Harker" —
bewusst, weil auch „Mina Murray" existiert). Hubs: Dracula (494 Chunks), Harker (338),
Van Helsing (243).

| Arm | Hit@1 | Hit@3 | Hit@5 | Hit@8 | MRR | Cov@8 |
|---|---|---|---|---|---|---|
| dense (Baseline) | 0.31 | 0.59 | 0.78 | **0.81** | 0.474 | 0.66 |
| link (α=0.03) | 0.31 | 0.56 | 0.81 | 0.81 | 0.480 | 0.66 |
| expand (α=0.03) | 0.34 | 0.59 | 0.72 | 0.81 | 0.481 | 0.67 |
| graph_only (Diagnose) | 0.12 | 0.28 | 0.38 | 0.47 | 0.234 | 0.35 |

Paired vs. dense (Rang des ersten Span-Treffers): **link 4 besser / 3 schlechter / 25 gleich**
(Vorzeichentest p = 1.0, ΔMRR +0.006); **expand 3 / 4 / 25** (ΔMRR +0.008). Je Fragetyp
(Hit@8 / MRR): fact n=26 0.92/0.532 → link 0.92/0.542, expand 0.92/0.541; multi n=3
0.33/0.111 → link 0.33/0.083, expand 0.33/0.111; paraphrase n=3 unverändert 0.33/0.333.
Sensitivität `link` (post hoc): α=0.02 MRR 0.483 (4/1/27); **α=0.05 Hit@8 0.84, MRR 0.452
(4/9/19)**; τ=0.40 0.480, τ=0.50 0.467.

![Graph-RAG lokal](../results/graph_experiment.png)

**Befund: kein messbarer Effekt** (vorab definierter Mittelfall — weder Erfolg noch negativ).
Beide Arme bewegen bei 25 von 32 Fragen nichts, der Rest ist 4:3 bzw. 3:4; ΔMRR liegt weit
unter der Schwelle +0.03, Hit@8 bleibt exakt gleich. H1 (Hit@8 ↑) ist widerlegt, H2
(Effekt bei multi/paraphrase) bei n=3 nicht erkennbar.

**Interpretation:**
1. **Das Graph-Signal ist real, aber schwach — und komplementär.** `graph_only` allein
   erreicht MRR 0.234 / Hit@8 0.47 (mehr als bge-m3-Sparse in Exp 6: 0.218 / 0.47) und
   findet zwei dense-MISSes (d01 @4, n09 @2 — das Demeter/Czarina-Catherine-Multi-Item). Die
   additive Fusion kann sie aber nicht heben: ihr Cosine liegt zu weit unter dem Top-8-Cut.
   Mit α=0.05 kommen sie rein (Hit@8 0.84), dafür kippt die Präzision (MRR 0.452, 4:9) — der
   Graph tauscht Precision gegen Recall, netto kein Gewinn.
2. **Entity-Linking DE→EN funktioniert für Eigennamen** (Demeter 0.66, Mr. Swales 0.72,
   Bloofer lady 0.56, Renfield 0.60), **ist aber verrauscht bei Orten**: Paddington, Bath,
   Whitby Castle tauchen mit 0.5–0.6 bei Fragen auf, die nichts damit zu tun haben —
   181 PLACE-Knoten aus einer 7B-Extraktion sind schlicht zu viel Beifang, und idf dämpft nur
   Hubs, nicht Rauschen.
3. **Die contextual-dense-Baseline lässt wenig Luft:** 26 von 32 Items sind Faktfragen, bei
   denen der Kontext-Präfix (Exp 5) das Pronomen-/Kontextproblem schon löst — genau das, was
   ein Entity-Graph sonst beisteuern würde.

**Kosten:** Extraktion 581 LLM-Aufrufe ≈ 2,9 h (Pilot 8,8 s/Chunk, Nachtjob unter Systemlast
17,8 s), Graph-Build + Embeddings ≈ 1 min, Retrieval-Latenz +14 ms je Frage (dense 35 ms →
link 49 ms, expand 48 ms; Graph lädt in 10 ms). Speicher: graph.json 0,4 MB + Vektoren 1,7 MB.

**Entscheidung:** `GRAPH_RAG_ENABLED` bleibt aus; der lokale Graph-Arm ist kein Hebel für
Passagen-Retrieval. Nicht gemessen, aber naheliegend als Folge-Idee: das Graph-Signal nicht
additiv, sondern als **Kandidaten-Generator für den Reranker** nutzen (dense Top-30 ∪ graph
Top-10 → Cross-Encoder) — Recall-Gewinn ohne den Precision-Tausch. Die eigentliche
Graph-RAG-Stärke (thematische Fragen) misst Exp 9.

## Exp 9 — Graph-RAG global: Community-Berichte (`eval/global_graph_experiment.py`)

**Idee:** Der lokale Pfad (Exp 8) findet Passagen. Die eigentliche Stärke von Graph-RAG
(Microsoft-Stil) liegt bei **thematischen** Fragen über den ganzen Roman („Wie entwickelt
sich Minas Rolle?"), für die keine einzelne Passage reicht. Dafür: Louvain-Communities auf
dem Entity-Graphen (Seed 42, deterministisch, Pruning `n_mentions ≥ 2`, `min_size 3`),
je Community ein englischer LLM-Bericht (title/summary/findings, JSON, temp 0), Berichte
mit bge-m3 eingebettet. Globaler Pfad: Frage → Top-m Berichte → Quellen = je Bericht `[i]`
plus 2 repräsentative Originalpassagen `[i+1]`, `[i+2]` (Zitate bleiben im Buchtext
verankert; Frontend-Chips funktionieren unverändert, Chip = „Zusammenfassung: <Titel>").
Arm `global_map` zusätzlich mit Map-Step (je Bericht ein Aufruf: relevante Punkte 0–100).

**Warum ein eigenes Gold-Set:** Auf der Span-Metrik kann ein globaler Pfad strukturell nicht
gewinnen (er liefert Berichte, keine Passagen). `eval/gold_global.jsonl`: 10 thematische
Dracula-Fragen (DE) mit Rubrik = 3–5 Kernpunkte, **vor** der Berichtserstellung aus
Buchwissen geschrieben (Leakage-Regel, Datum im `note`-Feld; vom Nutzer kuratiert). Dazu
3 Kontroll-Items aus gold_v2 (Faktfragen, Rubrik = Gold-Antwort), bei denen global
erwartungsgemäß **verlieren** sollte — das begründet das Routing statt eines Globalschalters.

**Metriken:** Rubrik-Abdeckung per LLM-Judge (qwen2.5:7b, ein Aufruf je Kernpunkt, JA/NEIN,
blind, gemischte Reihenfolge) mit Kalibrierung im selben Lauf (Verweigerung → soll ≈ 0 %,
Rubrik als Antwort → soll ≈ 100 %); NLI-Faithfulness `faith_chunks` nur gegen Buch-Chunks
(für alle Arme gleich definiert) und `faith_sources` inkl. Berichte; TTFT/e2e/Länge/
LLM-Aufrufe/Zitatquote; paired Coverage `global_* vs dense16` (dense16 = k=16 als starke
Baseline für breite Fragen).

**Hypothese (vorab):** Community-Berichte liefern bei thematischen Fragen mehr Kernpunkte
als 8 oder 16 Einzelpassagen, bei moderat niedrigerer Verankerung im Buchtext.
- **Positiv:** Coverage(global) ≥ Coverage(dense16) + 0.15 **und** paired ≥ 6/10 besser
  **und** `faith_chunks` ≥ dense16 − 0.15.
- **Negativ:** Coverage ≤ dense16 **oder** `faith_chunks` < 0.35 (Halluzinationsverdacht)
  **oder** Judge-Kalibrierung außerhalb [≤ 20 %, ≥ 80 %] (→ nicht interpretierbar, wird
  so berichtet).
- Dazwischen „gemischt": Produktdefault bleibt `local`.
- Erwartung Kontroll-Items: global schlechter als dense8.

**Setup:** Gruppe Horror (Live-Index), m=6, 2 Passagen je Bericht, temp 0, n=10 global
+ 3 Kontrolle; Louvain **Auflösung 2.0** (vor der Berichtserstellung festgelegt: 1.0 ergab nur
7 Communities bei Modularität 0.16 mit einem 55-Knoten-Blob um Dracula/Harker/London; 2.0
ergibt 14 thematisch lesbare Communities — Whitby/Demeter, Sewards Anstalt/Renfield, Lucys
Krankenzimmer, Verfolgung Varna/Galatz, Transsilvanien-Reise, Carfax; 146 Knoten nach
Pruning, Modularität 0.11); Judge = Antwortmodell (Bias benannt: gleicher Judge für alle Arme,
Hand-Stichprobe ≥ 3 Fragen × alle Arme als Spalte `manual_coverage`).

Messlauf 2026-09-23 (nach Ollama-Neustart mit `num_ctx` 16384 — ein erster Lauf mit 4096
schnitt **alle** 52 Prompts still auf 2050 Token ab und wurde verworfen; die Eval-Skripte prüfen
das Kontextfenster seitdem vorab). Judge-Kalibrierung: Verweigerung → 2 %, Rubrik als Antwort
→ 94 % (im Band, Ergebnis interpretierbar). Rubriken = **unkuratierter Entwurf** aus Buchwissen.

| Arm | Coverage (n=10) | Cov. Kontrolle (n=3) | faith_chunks | faith_sources | Zitatquote | Ø Aufrufe | Ø e2e |
|---|---|---|---|---|---|---|---|
| dense8 | 0.14 | **0.67** | **0.68** | 0.68 | **0.81** | 1 | **42 s** |
| dense16 | **0.16** | 0.67 | 0.47 | 0.47 | 0.78 | 1 | 58 s |
| global_direct | 0.04 | 0.00 (verweigert) | 0.56 | 0.86 | 0.47 | 1 | 81 s |
| global_map | 0.14 | 0.33 | 0.53 | 0.81 | 0.48 | 7 | 71 s |

Paired Coverage vs. dense16: **global_direct 1 besser / 4 schlechter / 5 gleich**,
**global_map 1 / 2 / 7**. Je Frage liegen alle vier Arme zwischen 0.0 und 0.6; bei g01, g09,
g10 trifft kein Arm einen einzigen Kernpunkt.

![Graph-RAG global](../results/global_graph_experiment.png)

**Befund: NEGATIV** (Vorab-Kriterium „Coverage ≤ dense16" erfüllt, für beide globalen Arme).
Community-Berichte machen die Antworten auf thematische Fragen nicht besser, nur länger,
langsamer und schlechter zitiert.

**Interpretation:**
1. **Bodeneffekt bei allen Armen.** Coverage 0.04–0.16 heißt: die Rubriken (Kernpunkte aus
   dem Wissen über den ganzen Roman) liegen weit über dem, was qwen2.5:7b aus 8–16 Passagen
   **oder** aus 14 Berichten erzeugt. Beispiel g03 (Dokumente als Erzählmittel): dense16 trifft
   3/5, global_direct 0/5 — obwohl beide inhaltlich Ähnliches sagen; der strikte JA/NEIN-Judge
   („ausdrücklich") hat bei so allgemeinen Antworten wenig Auflösung. Bei n=10 sind damit auch
   die Differenzen zwischen den dense-Armen nicht belastbar. Der Vergleich ist trotzdem
   entscheidbar: global gewinnt an keiner Stelle systematisch.
2. **Die Berichte sind das Problem, nicht der Suchpfad.** Der Graph ist ein Ko-Okkurrenz-Graph
   aus einer 7B-Extraktion; die Louvain-Communities sind Entity-Cluster, keine Themen. 5 von 14
   Berichten heißen wörtlich „Entities and Relationships in Dracula", der Inhalt ist eine
   Wer-mit-wem-Aufzählung. `faith_sources` 0.86 vs. `faith_chunks` 0.56 zeigt, worauf sich die
   globalen Antworten stützen: auf die Berichte, nicht auf den Buchtext. Die Halluzinations-
   prüfung wandert damit in die Berichtsstufe — und die ist ungeprüft (Smoke-Test: ein Bericht
   ließ Dr. Seward nach Whitby reisen).
3. **7B-Degeneration bei langem, gemischtsprachigem Kontext.** g01/global_direct kippt bei
   ~8,9k Token (englische Berichte, deutsche Frage) mitten im Satz ins Chinesische und erfindet
   die Figur „Márius Trótski"; 1 von 26 globalen Antworten, 0 von 26 dense. Der große Kontext
   ist für dieses Modell selbst ein Risiko, nicht nur ein Kostenfaktor.
4. **Nebenbefund dense16 vs. dense8:** +0.02 Coverage, aber `faith_chunks` 0.68 → 0.47 und
   e2e +16 s. Mehr Passagen verschlechtern die Verankerung deutlich — k=8 bleibt richtig, auch
   für breite Fragen.
5. **Kontroll-Items wie vorhergesagt:** global_direct verweigert Faktfragen ehrlich (Coverage 0,
   keine Halluzination), dense 0.67. global_map löst als **einziger** Arm n09 (Demeter →
   Czarina Catherine, das Multi-Hop-Item, das auch in Exp 8 nur `graph_only` fand) — anekdotisch,
   aber genau der Fragetyp, für den Communities gedacht sind.

**Kosten:** Berichte 14 LLM-Aufrufe ≈ 15 min; Antwort e2e 81 s (direct) bzw. 71 s (map, 7
Aufrufe) statt 42 s; Zitatquote fällt von 0.81 auf 0.47. Eval-Lauf gesamt ≈ 2,5 h (50 min
Antworten, 10 min Judge, ~1,5 h NLI auf CPU).

**Entscheidung:** Kein globaler Pfad im Produkt; `GRAPH_RAG_MODE` bleibt `local`, das Flag
bleibt aus. Der Code bleibt als opt-in (`mode=global` im Chat-Request) für Demos erhalten.
Was es für einen zweiten Versuch bräuchte, in dieser Reihenfolge: (a) Berichte, die Themen
statt Entity-Listen beschreiben — d. h. eine Extraktion mit Relations-Beschreibungen und ein
größeres Modell dafür; (b) Berichte per NLI gegen die repräsentativen Passagen prüfen, bevor
sie Quellen werden; (c) ein größeres Antwortmodell oder kürzere Kontexte (kein 9k-Token-Prompt
für ein 7B); (d) kuratierte Rubriken mit Kernpunkten, die aus dem Text belegbar sind, und ein
Judge mit Skala statt JA/NEIN.

## Exp 10 — k-Sweep: wie viele Passagen verträgt das 7B-Modell? (`eval/answer_eval.py --k`)

**Anlass:** Nebenbefund aus Exp 9 — dense16 senkte `faith_chunks` gegenüber dense8 von 0.68
auf 0.47. **Vorab-Erwartung:** Faithfulness fällt mit k monoton, Hit@k steigt, Sweet Spot 6–8.

**Setup:** Metrik-Suite auf Gold v2 (n=36, davon 32 beantwortbar, 4 unanswerable), span-basiert,
qwen2.5:7b, temp 0, **num_ctx 16384** (erste Suite seit dem Kontext-Fix), NLI auf MPS.
k ∈ {4, 6, 8, 12}; alles andere identisch (contextual + dedup, kein S2B, kein Reranker).

| k | Faithfulness | ROUGE-L | Token-F1 | Refusal (unansw.) | False-Refusal | Hit@k | MRR | TTFT med. | e2e med. |
|---|---|---|---|---|---|---|---|---|---|
| 4 | **0.564** | 0.173 | 0.207 | 0.75 | 0.00 | 0.688 | 0.451 | 5.2 s | **8.2 s** |
| 6 | 0.505 | 0.185 | 0.217 | 0.75 | 0.00 | 0.781 | 0.469 | 4.6 s | 9.6 s |
| 8 (Default) | 0.448 | 0.172 | 0.197 | 0.75 | 0.00 | **0.812** | **0.474** | 12.8 s | 16.1 s |
| 12 | 0.444 | 0.179 | 0.236 | 0.75 | 0.00 | 0.812 | 0.474 | 25.8 s | 32.3 s |

Paired Faithfulness je Frage vs. k=8 (n=32): k=4 **13 besser / 5 schlechter / 14 gleich**
(Vorzeichentest p ≈ 0.10), k=6 10 / 6 / 16 (p ≈ 0.45), k=12 13 / 9 / 10 (p ≈ 0.52).

![k-Sweep](../results/ksweep_experiment.png)

**Befund:** Die Erwartung bestätigt sich als **Trend**, nicht als signifikanter Effekt:
Faithfulness fällt monoton 0.56 → 0.51 → 0.45 → 0.44, Hit@k sättigt bei 8 (0.81), ab k=8
kostet jede weitere Passage nur noch Latenz (e2e 16 s → 32 s bei k=12) ohne Retrieval-Gewinn.
ROUGE-L/F1 sind flach (Paraphrasen-Problem, s. Metrik-Suite oben), Refusal-Verhalten ist von
k unabhängig (3/4 unanswerable verweigert, 0 False-Refusals bei allen k). Bei n=32 und den
Streuungen (k=12 vs. 8 ist 13:9 trotz niedrigerem Mittel) ist keiner der Paarvergleiche
signifikant; die Monotonie über vier Stufen ist das belastbarere Argument.

**Einordnung der alten Zahl:** Die bisher berichtete Faithfulness 0.667 (n=21, kapitel-basiert)
entstand mit `num_ctx` 4096: Ollama kürzte die Prompts still auf 2050 Token, das Modell sah
effektiv nur die letzten ~2 Passagen — genau der k-Bereich, in dem die Stützung am höchsten
ist. Die Zahl war also kein Qualitätsmaß der Pipeline, sondern ein Artefakt der Kürzung.
Referenz ab jetzt: **k=8 → 0.448** (korrekter Kontext, n=32, span-basiert).

**Empfehlung:** `TOP_K=6` als Default — Faithfulness +0.057, e2e −40 %, dafür Hit@k −0.03
(eine Frage von 32) und MRR −0.005. k=4 gewinnt weitere +0.06 Stützung, verliert aber vier
Fragen im Retrieval. Wer Small-to-Big (Exp 7) aktiviert, sollte k eher senken als erhöhen:
die Erweiterung liefert Kontext, ohne die Passagenzahl zu erhöhen. Default im Repo bleibt bis
zur Entscheidung 8; Umstellung = eine Env-Zeile (`TOP_K=6`).

**Kosten:** vier Suiten ≈ 55 min gesamt (NLI auf MPS statt CPU: Minuten statt einer Stunde je Lauf).

## Nächste Schritte

1. ~~Gold-Set v2 mit Span-Labels~~ ✅ kuratiert (36 Items, s. o.); Eval läuft auf
   Span-Metrik. Ausbau auf n≥50 und `question_en` für die neuen Items bleibt sinnvoll.
2. ~~Exp 5 — Contextual Retrieval~~ ✅ gemessen (+0.182 MRR, s. o.) — offen ist die
   **Produktivierung**: `CONTEXTUAL_INGEST_ENABLED`-Flag in der Ingestion-Pipeline + Re-Ingest.
3. ~~Exp 6 — Sparse-Hybrid~~ ✅ gemessen (s. o.) — ehrliches Negativ-Ergebnis,
   kein Sparse-Hybrid in der Pipeline.
4. Bootstrap-CI / paired Test, sobald n≥30.
5. ~~Reranker in die `/chat`-Pipeline integrieren~~ ✅ umgesetzt (opt-in via
   `RERANKER_ENABLED`, s. README).
6. ~~Exp 8 — Graph-RAG lokal~~ ✅ gemessen: kein messbarer Effekt (4:3:25 / 3:4:25), Flag
   bleibt aus; offen: Graph-Signal als Reranker-Kandidatenquelle.
7. ~~Exp 9 — Graph-RAG global~~ ✅ gemessen: negativ (Coverage 0.04/0.14 vs. dense16 0.16,
   Berichte zu flach, 7B degeneriert bei 9k Token); kein globaler Pfad im Produkt.
8. ~~k-Sweep~~ ✅ gemessen (Exp 10): Faithfulness fällt monoton mit k, Hit@k sättigt bei 8;
   Empfehlung TOP_K=6. Offen: Entscheidung über den Default; Bootstrap-CI (Punkt 4) würde
   die k-Vergleiche erst belastbar machen.
