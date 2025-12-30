"""Recommendation orchestration using LangGraph."""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph

from ..schemas import Product, RecommendationItem, RecommendationRequest, RecommendationResponse
from .agents import AgentContext, MerchandiserAgent, StylistAgent, TrendScoutAgent


class RecommendationState(TypedDict, total=False):
    user_profile: RecommendationRequest
    vector_hits: list[dict[str, Any]]
    shortlist: list[Product]
    recommendations: list[dict[str, Any]]
    trend_insights: str
    ops_notes: list[str]
    budget_ceiling: float | None


class RecommendationEngine:
    def __init__(self, llm: BaseChatModel, vector_store, catalog: list[Product]):  # noqa: ANN001
        self.llm = llm
        self.vector_store = vector_store
        self.catalog_index = {product.sku: product for product in catalog}
        self.ctx = AgentContext(llm=llm)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(RecommendationState)
        graph.add_node("vector_search", self._vector_search_node)
        graph.add_node("trend", TrendScoutAgent(self.ctx))
        graph.add_node("stylist", StylistAgent(self.ctx))
        graph.add_node("merch", MerchandiserAgent(self.ctx))

        graph.set_entry_point("vector_search")
        graph.add_edge("vector_search", "trend")
        graph.add_edge("trend", "stylist")
        graph.add_edge("stylist", "merch")
        graph.add_edge("merch", END)
        return graph.compile()

    def _vector_search_node(self, state: RecommendationState) -> RecommendationState:
        profile = state["user_profile"]
        query = " ".join(
            profile.style_goals
            + profile.preferred_colors
            + profile.context.get("keywords", [])
        ) or "elevated essentials"
        docs = self.vector_store.similarity_search(query, k=4)
        hits = []
        shortlisted: list[Product] = []
        for doc in docs:
            sku = doc.metadata.get("sku")
            product = self.catalog_index.get(sku)
            if product:
                shortlisted.append(product)
            hits.append({"sku": sku, "score": doc.metadata.get("score")})
        if not shortlisted:
            shortlisted = list(self.catalog_index.values())[:3]
        state["vector_hits"] = hits
        state["shortlist"] = shortlisted
        state["budget_ceiling"] = profile.budget
        return state

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        initial_state: RecommendationState = {
            "user_profile": request,
            "vector_hits": [],
            "shortlist": [],
            "budget_ceiling": request.budget,
        }
        final_state = self.graph.invoke(initial_state)
        recommendations = final_state.get("recommendations", [])
        items = [RecommendationItem(**rec) for rec in recommendations]
        return RecommendationResponse(
            user_id=request.user_id,
            strategy="multi_agent",
            generated_at=datetime.utcnow(),
            items=items,
            notes="; ".join(final_state.get("ops_notes", [])),
        )


class RuleBasedEngine:
    def __init__(self, catalog: list[Product]):
        self.catalog = catalog

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        filtered = [
            product
            for product in self.catalog
            if not request.preferred_colors or set(request.preferred_colors) & set(product.colors)
        ]
        shortlist = (filtered or self.catalog)[:3]
        items = [
            RecommendationItem(
                sku=product.sku,
                name=product.name,
                rationale="Matched by rule-based filter",
                price=product.price,
                currency=product.currency,
                confidence=0.5,
                complementary_items=request.cart_items,
            )
            for product in shortlist
        ]
        return RecommendationResponse(
            user_id=request.user_id,
            strategy="rule_based",
            generated_at=datetime.utcnow(),
            items=items,
            notes="Rule-based fallback",
        )
