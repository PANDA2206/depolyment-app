"""Pydantic schemas shared across API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Product(BaseModel):
    sku: str
    name: str
    brand: str
    price: float
    currency: str = "USD"
    colors: list[str] = Field(default_factory=list)
    sizes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    inventory: int = 0
    image_url: str | None = None
    description: str | None = None


class CatalogResponse(BaseModel):
    products: list[Product]


class RecommendationRequest(BaseModel):
    user_id: str
    style_goals: list[str] = Field(default_factory=list)
    preferred_colors: list[str] = Field(default_factory=list)
    disliked_colors: list[str] = Field(default_factory=list)
    budget: float | None = None
    event_date: datetime | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    cart_items: list[str] = Field(default_factory=list)


class RecommendationItem(BaseModel):
    sku: str
    name: str
    rationale: str
    price: float
    currency: str
    confidence: float = Field(ge=0, le=1)
    complementary_items: list[str] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    user_id: str
    strategy: str
    generated_at: datetime
    items: list[RecommendationItem]
    notes: str | None = None


class HealthStatus(BaseModel):
    status: str
    version: str
    environment: str
