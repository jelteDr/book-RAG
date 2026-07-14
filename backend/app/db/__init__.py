"""Persistenz (SQLite): Metadaten (Bücher, Gruppen, Ingestion-Status) + Monitoring-Log.

Source-of-Truth für anzeigbaren Chunk-Text ist die Qdrant-Payload; SQLite hält
Metadaten und das Query-Log (Frage, Antwort, Modell, ttft/e2e, tps, Tokens,
retrieved/cited chunk_ids, Qualitäts-Scores). Schema-Implementierung in M1.
"""
