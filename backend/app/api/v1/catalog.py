"""Catalog endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ...schemas import CatalogResponse
from ...services import catalog as catalog_service

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("/", response_model=CatalogResponse)
def list_catalog() -> CatalogResponse:
    products = catalog_service.list_products()
    return CatalogResponse(products=products)
