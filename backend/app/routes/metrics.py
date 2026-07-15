"""Metrik-Route: aggregiert das query_log pro Modell (Serving-Telemetrie)."""

from fastapi import APIRouter
from sqlalchemy import func
from sqlmodel import select

from app.db.models import QueryLog
from app.db.session import session_factory

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics() -> dict:
    async with session_factory() as session:
        stmt = (
            select(
                QueryLog.model,
                func.count().label("queries"),
                func.avg(QueryLog.ttft_ms).label("avg_ttft_ms"),
                func.avg(QueryLog.tps).label("avg_tps"),
                func.avg(QueryLog.completion_tokens).label("avg_completion_tokens"),
            )
            .group_by(QueryLog.model)
        )
        rows = (await session.exec(stmt)).all()

    def _round(value: float | None) -> float | None:
        return round(float(value), 1) if value is not None else None  # func.avg liefert Decimal

    return {
        "per_model": [
            {
                "model": r.model,
                "queries": r.queries,
                "avg_ttft_ms": _round(r.avg_ttft_ms),
                "avg_tps": _round(r.avg_tps),
                "avg_completion_tokens": _round(r.avg_completion_tokens),
            }
            for r in rows
        ]
    }
