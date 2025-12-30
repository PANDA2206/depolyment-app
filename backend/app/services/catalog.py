"""Catalog data utilities."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
import json
from pathlib import Path

from ..schemas import Product


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "products.json"


@lru_cache(maxsize=1)
def _load_products(path: Path = DATA_PATH) -> list[Product]:
    with path.open() as handle:
        raw = json.load(handle)
    return [Product(**item) for item in raw]


def list_products() -> list[Product]:
    return _load_products()


def filter_products(
    tags: set[str] | None = None,
    colors: set[str] | None = None,
) -> Iterable[Product]:
    for product in _load_products():
        if tags and not (tags & set(product.tags)):
            continue
        if colors and not (colors & set(product.colors)):
            continue
        yield product
