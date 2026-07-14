# Gold-Eval-Set

Handgelabeltes Evaluations-Set für reproduzierbare Retrieval-/Antwort-Qualität
(20–50 Fragen). Format (JSONL), wird in M2 gefüllt:

```json
{"id": "q001", "question": "…", "relevant_chunk_ids": ["…"], "gold_answer": "…"}
```

Verwendung in `notebooks/02_retrieval_eval.ipynb` (Recall@k, MRR/nDCG) und
`notebooks/03_faithfulness.ipynb`. Enthält **keinen** urheberrechtlich
geschützten Volltext — nur Fragen, Chunk-IDs und kurze Referenzantworten.
