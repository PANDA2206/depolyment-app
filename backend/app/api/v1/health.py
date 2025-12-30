"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ...core.config import get_settings
from ...schemas import HealthStatus

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=HealthStatus)
def liveness() -> HealthStatus:
    settings = get_settings()
    return HealthStatus(status="ok", version="0.1.0", environment=settings.environment)


@router.get("/ready", response_model=HealthStatus)
def readiness() -> HealthStatus:
    settings = get_settings()
    return HealthStatus(status="ready", version="0.1.0", environment=settings.environment)
