"""Ingestion-Pipeline (M1/M3): parser, cleaner, chunker, metadata.

- parser:   Rohtext + Encoding-Sniffing, Gutenberg-Header/Boilerplate.
- cleaner:  deterministische Bereinigung + Dry-Run-Prüf-Gate (Cleaning-Report).
- chunker:  strukturbewusstes Chunking (Kapitel -> rekursiv), Offsets auf
            BEREINIGTEM Text.
"""
