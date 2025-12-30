from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .assets import list_active_metrics, load_seed_terms
from .config import settings


def embeddings() -> OpenAIEmbeddings:
    if settings.embed_base_url and "dashscope.aliyuncs.com" in settings.embed_base_url:
        try:
            from langchain_community.embeddings import DashScopeEmbeddings
        except ImportError as exc:
            raise ImportError(
                "DashScope embeddings require the `dashscope` package. "
                "Install it with `uv add dashscope` or `pip install dashscope`."
            ) from exc
        return DashScopeEmbeddings(
            model=settings.embed_model,
            dashscope_api_key=settings.embed_api_key,
        )

    return OpenAIEmbeddings(
        model=settings.embed_model,
        api_key=settings.embed_api_key,
        base_url=settings.embed_base_url,
    )


def chroma(collection: str) -> Chroma:
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings(),
        persist_directory=settings.chroma_dir,
    )


@dataclass
class RetrievalHit:
    id: str
    score: float
    kind: str
    payload: dict[str, Any]


def build_docs() -> list[Document]:
    metrics = list_active_metrics()
    terms = load_seed_terms()
    docs: list[Document] = []

    for m in metrics:
        text = (
            f"metric_key: {m.metric_key}\n"
            f"metric_name_zh: {m.metric_name_zh}\n"
            f"description: {m.description}\n"
            f"fact_table: {m.fact_table}\n"
            f"time_column: {m.time_column}\n"
            f"measure_expr: {m.measure_expr}\n"
            f"default_filters: {m.default_filters}\n"
            f"allowed_dims: {','.join(m.allowed_dims)}\n"
        )
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "kind": "metric",
                    "metric_key": m.metric_key,
                    "metric_name_zh": m.metric_name_zh,
                },
            )
        )

    for t in terms:
        text = f"term: {t.term}\ncanonical: {t.canonical}\ndefinition: {t.definition}\n"
        docs.append(
            Document(
                page_content=text,
                metadata={"kind": "term", "term": t.term, "canonical": t.canonical},
            )
        )
    return docs


def rebuild_vector_index() -> None:
    db = chroma("semantic_assets")
    # Start fresh
    try:
        db.delete_collection()
    except Exception:
        pass
    db = chroma("semantic_assets")
    db.add_documents(build_docs())
    db.persist()


def retrieve_semantic_assets(query: str, top_k: int = 6) -> list[RetrievalHit]:
    db = chroma("semantic_assets")
    hits = db.similarity_search_with_relevance_scores(query, k=top_k)
    out: list[RetrievalHit] = []
    for i, (doc, score) in enumerate(hits):
        md = dict(doc.metadata)
        kind = md.get("kind", "unknown")
        out.append(
            RetrievalHit(
                id=f"hit_{i}",
                score=float(score),
                kind=kind,
                payload={"text": doc.page_content, "meta": md},
            )
        )
    return out
