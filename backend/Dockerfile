# Backend-Image auf Basis von Python 3.12 (umgeht das Wheel-Problem von 3.14).
# uv verwaltet Abhängigkeiten reproduzierbar.
FROM python:3.12-slim

# uv aus dem offiziellen Image kopieren (kein Netz-Install nötig).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Zuerst nur die Dependency-Dateien -> besseres Layer-Caching. uv.lock + --frozen: das Image
# bekommt exakt die gelockten Versionen (ohne Lock zog ein Rebuild sqlmodel 0.0.48, das naive
# datetimes ablehnt -> Chats wurden still nicht mehr gespeichert).
COPY pyproject.toml uv.lock ./

# Optional: ML-Extras (torch + sentence-transformers) fuer Reranker/Faithfulness.
# Standard false -> Image bleibt schlank (~200 MB statt mehreren GB).
# Aktivieren: docker compose build --build-arg INSTALL_ML_EXTRAS=true (bzw. via .env).
ARG INSTALL_ML_EXTRAS=false
RUN if [ "$INSTALL_ML_EXTRAS" = "true" ]; then \
        uv sync --frozen --no-install-project --no-dev --group reranker; \
    else \
        uv sync --frozen --no-install-project --no-dev; \
    fi

# Anschließend der Anwendungscode.
COPY app ./app

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
