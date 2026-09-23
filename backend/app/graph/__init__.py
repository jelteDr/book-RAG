"""Graph-RAG (Exp 8/9): Entity-Graph aus den Chunks, Graph-gestütztes Retrieval.

Ablauf:
  1. `extract_cli`  — LLM extrahiert je Chunk Entities + Relationen (JSON, Checkpoint).
  2. `build_cli`    — baut daraus einen networkx-Graphen + Entity-Embeddings (data/graph/<group>/).
  3. `service`      — GraphService lädt den Graphen lazy und liefert den Retrieval-Arm
                      (`retriever.graph_retrieve`), der im RagService opt-in hängt.

Alle Artefakte liegen unter `data/graph/` (Buchtext-Derivate, gitignored, im
Container über das ./data-Volume sichtbar).
"""
