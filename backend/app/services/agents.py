"""LangGraph agent definitions for the recommendation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from ..schemas import Product, RecommendationRequest


@dataclass
class AgentContext:
    llm: BaseChatModel


@dataclass
class TrendScoutAgent:
    ctx: AgentContext

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        profile: RecommendationRequest = state["user_profile"]
        vector_hits = state["vector_hits"]
        prompt = (
            "You are TrendScout, a fashion analyst. Summarize current trends relevant to the user.\n"
            f"User goals: {profile.style_goals}. Preferred colors: {profile.preferred_colors}.\n"
            f"Vector hits: {vector_hits}."
        )
        summary = self.ctx.llm.invoke(prompt).content if self._can_call_llm() else "Neutral palette tailoring"
        state["trend_insights"] = summary
        return state

    def _can_call_llm(self) -> bool:
        return hasattr(self.ctx.llm, "invoke")


@dataclass
class StylistAgent:
    ctx: AgentContext

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        profile: RecommendationRequest = state["user_profile"]
        shortlisted: list[Product] = state["shortlist"]
        insights = state.get("trend_insights", "")
        prompt = (
            "You are a celebrity stylist. Craft outfit rationales for the shortlisted products.\n"
            f"User data: {profile.model_dump()}. Trend insights: {insights}."
        )
        rationale = self.ctx.llm.invoke(prompt).content if self._can_call_llm() else "Perfect for understated glam."

        enriched = []
        for product in shortlisted:
            enriched.append(
                {
                    "sku": product.sku,
                    "name": product.name,
                    "price": product.price,
                    "currency": product.currency,
                    "rationale": rationale,
                    "confidence": 0.74,
                    "complementary_items": profile.cart_items,
                }
            )
        state["recommendations"] = enriched
        return state

    def _can_call_llm(self) -> bool:
        return hasattr(self.ctx.llm, "invoke")


@dataclass
class MerchandiserAgent:
    ctx: AgentContext

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        recs = state.get("recommendations", [])
        notes = []
        for rec in recs:
            if rec["price"] > (state.get("budget_ceiling") or 9999):
                rec["confidence"] = 0.45
                notes.append(f"Adjusted price sensitivity for {rec['sku']}")
        state["ops_notes"] = notes or ["Inventory validated"]
        return state
