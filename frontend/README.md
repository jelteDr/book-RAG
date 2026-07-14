# Frontend (Angular, minimal)

Wird in **M1** angelegt: schlanke Angular-App mit Chat-Ansicht, Zitat-Chips
(`[n]` → Quell-Chunk) und Modell-Dropdown. Kein Monitoring-Dashboard im
Frontend — Auswertung/Modell-Vergleich laufen in den Notebooks.

- Build im Container mit **gepinntem Node-LTS** (z. B. `node:22-slim`); das lokale
  Node (v26) wird nicht verwendet.
- Auslieferung als statischer Build hinter **nginx** mit `/api`-Reverse-Proxy auf
  das Backend (löst CORS/Origin; `proxy_buffering off` für SSE-Streaming).
