# Evaluations-Ergebnisse

Reproduzierbar via `make eval` (Baseline) bzw. den Experiment-Skripten in `eval/`.
Korpus: Dracula (581 Chunks), Embeddings `bge-m3`, Qdrant Cosine. Relevanz
kapitel-basiert (Gold-Set: `eval/gold_dracula.jsonl`, 10 Fragen).

## Baseline (deutsche Fragen, k=8)

| Metrik | Wert |
|---|---|
| Recall@1 | 0.40 |
| Recall@3 | 0.60 |
| Recall@5 | 0.80 |
| Recall@8 | 0.80 |
| MRR | 0.517 |

Misses (kein relevantes Kapitel in Top-8): **d01** (Harkers Reisegrund) und
**d09** (Vampir-Abwehrmittel).

## Experiment 1 — Frage DE vs. EN (`eval/lang_experiment.py`)

**Hypothese:** Die mäßige Baseline liegt am cross-lingualen Gap (DE-Frage →
EN-Text). **Test:** dieselben Fragen auf Deutsch vs. Englisch retrieven (nur die
Query wird neu eingebettet, die Chunks bleiben unverändert).

| Metrik | DE | EN | Δ |
|---|---|---|---|
| Recall@1 | 0.40 | 0.20 | −0.20 |
| Recall@3 | 0.60 | 0.70 | +0.10 |
| Recall@5 | 0.80 | 0.80 | 0.00 |
| Recall@8 | 0.80 | **1.00** | +0.20 |
| MRR | 0.517 | 0.450 | −0.067 |

![DE vs EN](../results/lang_experiment.png)

**Ergebnis — Hypothese teilweise widerlegt:**
- `bge-m3` ist echt multilingual: deutsche Fragen ranken den **Top-Treffer sogar
  besser** (Recall@1 0.40 vs 0.20, MRR 0.517 vs 0.450). Query-Übersetzung ist also
  **kein** pauschaler Fix — sie verschlechtert die Präzision an Position 1.
- Englisch bringt nur im **Tail** Vorteile: die beiden harten Misses d01/d09
  tauchen auf Englisch auf (Recall@8 1.00 vs 0.80), aber erst auf Rang 8.

**Schlussfolgerung / nächster Schritt:** Der Flaschenhals ist nicht primär die
Sprache, sondern **breite, über viele Kapitel verteilte Fragen + Eigennamen**
("Demeter", "Harker"), bei denen reines Dense-Retrieval schwächelt. Deshalb ist
das nächste Experiment **Hybrid-Retrieval (Dense + BM25/Sparse, RRF)** bzw. ein
Cross-Encoder-Reranker — das adressiert Eigennamen und Tail-Treffer direkt.
