"""FastAPI application entry point."""

from __future__ import annotations

from typing import Any

import httpx
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


class HuggingFaceInferenceChatModel(BaseChatModel):
    """Thin wrapper around the Hugging Face Inference API."""

    model_name: str

    def __init__(
        self,
        api_token: str,
        model_name: str,
        temperature: float = 0.2,
        max_new_tokens: int = 256,
        base_url: str | None = None,
    ) -> None:
        self.api_token = api_token
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        base = (base_url or "https://api-inference.huggingface.co").rstrip("/")
        self.endpoint = f"{base}/models/{model_name}"
        self._client = httpx.Client(timeout=60.0)

    @property
    def _llm_type(self) -> str:
        return "huggingface-inference"

    def _format_prompt(self, messages: list[BaseMessage]) -> str:
        segments = []
        for message in messages:
            role = getattr(message, "type", "user")
            content = message.content if isinstance(message.content, str) else str(message.content)
            segments.append(f"{role.upper()}: {content}")
        return "\n".join(segments)

    def _generate(  # type: ignore[override]
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
    ) -> ChatResult:
        _ = stop, run_manager
        prompt = self._format_prompt(messages)
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                "return_full_text": False,
            },
        }
        headers = {"Authorization": f"Bearer {self.api_token}"}
        try:
            response = self._client.post(
                self.endpoint,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            content = self._extract_text(data)
        except httpx.HTTPError as exc:  # pragma: no cover - network edge
            logger.warning("llm.huggingface_error", error=str(exc))
            content = "Open-source model unavailable; using fallback rationale."
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[[generation]])

    def _extract_text(self, data: Any) -> str:
        if isinstance(data, list) and data:
            candidate = data[0]
            if isinstance(candidate, dict):
                generated = candidate.get("generated_text")
                summary = candidate.get("summary_text")
                if generated or summary:
                    return generated or summary or ""
                return str(candidate)
        if isinstance(data, dict):
            generated = data.get("generated_text")
            if generated:
                return generated
            return str(data)
        return "Could not parse Hugging Face response."


def _build_llm(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "azure" and settings.azure_openai_api_key:
        return AzureChatOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint or "",
            azure_deployment=settings.azure_openai_deployment or "",
            temperature=0.2,
        )
    if settings.llm_provider == "openai" and settings.openai_api_key:
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=0.2,
        )
    if settings.huggingface_api_token:
        return HuggingFaceInferenceChatModel(
            api_token=settings.huggingface_api_token,
            model_name=settings.huggingface_model,
            temperature=0.2,
            max_new_tokens=settings.huggingface_max_new_tokens,
            base_url=settings.huggingface_base_url,
        )
    logger.warning(
        "llm.fallback_mock",
        reason="missing credentials",
        provider=settings.llm_provider,
    )
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
        logger.info(
            "application.startup",
            strategy=settings.agent_strategy,
            env=settings.environment,
        )

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
