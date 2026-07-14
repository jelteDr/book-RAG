"""RAG-Kern (M1/M2): retriever, prompt_builder, citations, quality.

- retriever:      Embed der Frage -> Qdrant-Suche (Cosine, Filter), top_k.
- prompt_builder: nummerierte Kontext-Blöcke [1..k] + Zitier-Instruktion.
- citations:      [n] serverseitig auf Chunk-IDs mappen (halluzinierte n>k verwerfen).
- quality:        Faithfulness via NLI, Citation-Präzision (lokal).
"""
