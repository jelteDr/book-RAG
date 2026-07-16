# Notebooks (DS-Kern)

Analyse & Evaluation (Outputs werden von `nbstripout` vor dem Commit entfernt):

- `gold_set_v2.ipynb` — Gold-Set v2 kuratieren: passagen-genaue **Span-Labels** statt Kapitel
  (n≥30–50, Fragetypen-Mix inkl. unanswerable); interaktive PCA des Embedding-Raums (Hover,
  v1-Fragen als Overlay), Wortwolken; Labeling-Helfer auf der echten Pipeline-Suche,
  Smoke-Eval (span-basierte Hit-Rate@k/MRR), Export `eval/gold_v2.jsonl`.
- `01_cleaning_chunking.ipynb` — Cleaning-Regeln & Chunk-Größen datengetrieben justieren (Vorher/Nachher).
- `02_retrieval_eval.ipynb` — Gold-Eval-Set: Recall@k, MRR/nDCG; ein kontrolliertes Before/After-Experiment.
- `03_faithfulness.ipynb` — Faithfulness via multilingualem NLI (mDeBERTa-xnli), Citation-Präzision.
- `04_benchmark.ipynb` — Serving-Benchmark-Auswertung (TTFT/TPS), Ollama vs. llama-server.

Umgebung: Python 3.12 (uv). Werden ab M2/M4 gefüllt.
