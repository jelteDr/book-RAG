# Notebooks (DS-Kern)

Analyse & Evaluation (Outputs werden von `nbstripout` vor dem Commit entfernt):

- `01_cleaning_chunking.ipynb` — Cleaning-Regeln & Chunk-Größen datengetrieben justieren (Vorher/Nachher).
- `02_retrieval_eval.ipynb` — Gold-Eval-Set: Recall@k, MRR/nDCG; ein kontrolliertes Before/After-Experiment.
- `03_faithfulness.ipynb` — Faithfulness via multilingualem NLI (mDeBERTa-xnli), Citation-Präzision.
- `04_benchmark.ipynb` — Serving-Benchmark-Auswertung (TTFT/TPS), Ollama vs. llama-server.

Umgebung: Python 3.12 (uv). Werden ab M2/M4 gefüllt.
