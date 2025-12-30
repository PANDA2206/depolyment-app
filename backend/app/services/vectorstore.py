"""Vector store bootstrapping for catalog intelligence."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from ..core.config import Settings
from ..schemas import Product


class ConstantEmbeddings(Embeddings):
    """Deterministic fallback when no provider credentials exist."""

    def __init__(self, dimension: int = 1536) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        scale = min(len(text) / 100.0, 1.0)
        return [scale] * self.dimension


def _build_embeddings(settings: Settings) -> Embeddings:
    if settings.llm_provider == "azure":
        return AzureOpenAIEmbeddings(
            azure_deployment=settings.azure_openai_deployment or "",
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
        )
    if settings.llm_provider == "openai":
        return OpenAIEmbeddings(api_key=settings.openai_api_key)
    if settings.llm_provider == "huggingface":
        return HuggingFaceEmbeddings(model_name=settings.huggingface_embedding_model)
    return ConstantEmbeddings()


def bootstrap_vector_store(settings: Settings, products: Sequence[Product]) -> Chroma:
    persist_path = settings.vectorstore_path
    persist_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = _build_embeddings(settings)

    store = Chroma(
        collection_name=settings.vectorstore_collection,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
    )

    current = store._collection.count()  # type: ignore[attr-defined]
    if current == 0:
        documents = []
        for product in products:
            tags = " ".join(product.tags)
            content = f"{product.name} {product.description or ''} Tags: {tags}"
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "sku": product.sku,
                        "brand": product.brand,
                        "price": product.price,
                        "colors": product.colors,
                        "tags": product.tags,
                    },
                )
            )
        store.add_documents(documents)
        store.persist()

    return store
