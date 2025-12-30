"""FastAPI application entry point."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, FastAPI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from .api.v1 import catalog, health, recommendations
from .core.config import Settings, get_settings
from .schemas import RecommendationRequest, RecommendationResponse
from .services import catalog as catalog_service
from .services.graph import RecommendationEngine, RuleBasedEngine
from .services.vectorstore import bootstrap_vector_store

logger = structlog.get_logger(__name__)


class MockChatModel(BaseChatModel):
    model_name: str = "mock-chat"

    def _generate(  # type: ignore[override]
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
    ) -> ChatResult:
        _ = stop, run_manager
        joined = " | ".join(msg.content if isinstance(msg.content, str) else "" for msg in messages)
        content = f"Mock recommendation insight based on: {joined[:200]}"
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[[generation]])


def _build_llm(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "azure":
        return AzureChatOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint or "",
            azure_deployment=settings.azure_openai_deployment or "",
            temperature=0.2,
        )
    if settings.llm_provider == "openai":
        return ChatOpenAI(api_key=settings.openai_api_key, model=settings.openai_model, temperature=0.2)
    return MockChatModel()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="FashionFlow API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    api_v1 = APIRouter(prefix="/api/v1")
    api_v1.include_router(catalog.router)
    api_v1.include_router(recommendations.router)

    app.include_router(api_v1)
    app.include_router(health.router)

    @app.on_event("startup")
    async def startup_event() -> None:
        products = catalog_service.list_products()
        vector_store = bootstrap_vector_store(settings, products)
        llm = _build_llm(settings)
        multi_agent_engine = RecommendationEngine(llm, vector_store, products)
        rule_engine = RuleBasedEngine(products)

        app.state.recommendation_engines = {
            "multi_agent": multi_agent_engine,
            "rule_based": rule_engine,
        }
        app.state.default_strategy = settings.agent_strategy
        logger.info("application.startup", strategy=settings.agent_strategy, env=settings.environment)

    return app


app = create_app()


def offline_recommend(
    payload: RecommendationRequest,
    strategy: str = "multi_agent",
) -> RecommendationResponse:
    engine = app.state.recommendation_engines.get(strategy)
    if not engine:
        raise ValueError(f"Strategy {strategy} not available")
    return engine.recommend(payload)
