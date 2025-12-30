"""Recommendation API surface."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ...schemas import RecommendationRequest, RecommendationResponse

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


def _get_engine(request: Request, strategy: str | None = None):  # noqa: ANN001
    available = request.app.state.recommendation_engines
    if strategy and strategy in available:
        return available[strategy]
    default_strategy = request.app.state.default_strategy
    return available[default_strategy]


@router.post("/", response_model=RecommendationResponse)
def post_recommendations(
    payload: RecommendationRequest,
    request: Request,
    strategy: str | None = None,
) -> RecommendationResponse:
    engine = _get_engine(request, strategy)
    if engine is None:
        raise HTTPException(status_code=500, detail="Recommendation engine unavailable")
    return engine.recommend(payload)
