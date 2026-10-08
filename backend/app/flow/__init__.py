"""Corrective RAG (Exp 11) als LangGraph-Zustandsgraph — opt-in über `mode=crag`.

LangGraph steuert hier nur den Ablauf (Knoten, bedingte Kanten, Schleifen). Die Knoten
selbst sind dünne Wrapper um die vorhandenen Bausteine (retriever, prompt_builder,
citations, FaithfulnessService) — der gemessene Retrieval-Kern bleibt unverändert.
"""
