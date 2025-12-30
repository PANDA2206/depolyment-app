#!/usr/bin/env python3
"""Minimal Retrieval-Augmented Generation (RAG) script.

This script expects the following third-party packages to be installed:
- langchain
- langchain-community
- langchain-openai
- chromadb

Usage example:
    python rag_script.py --docs ./data --question "What is RAG?"

Set the OPENAI_API_KEY environment variable before running if you plan to use
OpenAI models for embeddings and generation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings


def _iter_documents(doc_dir: Path) -> Iterable:
    """Load every supported text document under doc_dir recursively."""

    loader = DirectoryLoader(
        doc_dir.as_posix(),
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )
    return loader.load()


def build_vector_store(doc_dir: Path, persist_dir: Path | None = None) -> Chroma:
    """Create (or load) a vector store from the given directory."""

    documents = _iter_documents(doc_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = splitter.split_documents(documents)

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    vectordb = Chroma.from_documents(
        split_docs,
        embedding=embeddings,
        persist_directory=persist_dir.as_posix() if persist_dir else None,
    )

    if persist_dir:
        vectordb.persist()

    return vectordb


def build_chain(vectordb: Chroma) -> RetrievalQA:
    """Wire up a RetrievalQA chain with an OpenAI chat model."""

    llm = AzureChatOpenAI(
        temperature=0.2,
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
    )


def run_rag(question: str, doc_dir: Path, persist_dir: Path | None = None) -> None:
    """Execute the full pipeline and print the answer plus sources."""

    if not doc_dir.exists():
        raise FileNotFoundError(f"Document directory not found: {doc_dir}")

    vectordb = build_vector_store(doc_dir, persist_dir)
    chain = build_chain(vectordb)
    response = chain.invoke({"query": question})

    print("\nAnswer:\n" + response["result"])  # noqa: T201

    sources = response.get("source_documents", [])
    if not sources:
        print("\nNo sources returned.")  # noqa: T201
        return

    print("\nSources:")  # noqa: T201
    for idx, doc in enumerate(sources, start=1):
        metadata = doc.metadata
        source_path = metadata.get("source", "unknown")
        print(f"  {idx}. {source_path}")  # noqa: T201


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple RAG pipeline.")
    parser.add_argument(
        "--docs",
        type=Path,
        required=True,
        help="Directory containing .txt files for ingestion.",
    )
    parser.add_argument(
        "--persist",
        type=Path,
        default=None,
        help="Optional directory to persist the Chroma index.",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to pass through the pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    args = parse_args()
    run_rag(args.question, args.docs, args.persist)


if __name__ == "__main__":
    main()
