"""Small exact vector index for reproducible retrieval-policy experiments.

This is intentionally independent of Pinecone APIs. It is a real searchable
index over persisted corpus records, not a mock response fixture. Exact cosine
search keeps the experiment deterministic and inspectable for small labeled
corpora.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from labeled_corpus import CorpusDocument


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("vector_dimension_mismatch")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("zero_vector")
    return dot / (left_norm * right_norm)


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    score: float
    namespace: str
    freshness: float
    quality: float
    metadata: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "score": self.score,
            "namespace": self.namespace,
            "freshness": self.freshness,
            "quality": self.quality,
            "metadata": dict(sorted(self.metadata.items())),
        }


class DisposableVectorIndex:
    def __init__(self, documents: Iterable[CorpusDocument]) -> None:
        rows = list(documents)
        if not rows:
            raise ValueError("index_documents_empty")
        dimension = len(rows[0].vector)
        ids: set[str] = set()
        for document in rows:
            if len(document.vector) != dimension:
                raise ValueError("index_dimension_mismatch")
            if document.document_id in ids:
                raise ValueError("duplicate_document_id")
            ids.add(document.document_id)
        self._documents = tuple(rows)
        self.dimension = dimension

    def search(
        self,
        query_vector: tuple[float, ...],
        *,
        top_k: int,
        namespace: str | None = None,
        filter_equals: Mapping[str, str] | None = None,
        reranker: str = "none",
    ) -> tuple[SearchHit, ...]:
        if len(query_vector) != self.dimension:
            raise ValueError("query_dimension_mismatch")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k_invalid")
        if reranker not in {"none", "quality", "freshness", "quality_freshness"}:
            raise ValueError("reranker_invalid")
        filters = dict(filter_equals or {})
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in filters.items()):
            raise ValueError("filter_invalid")

        scored: list[tuple[float, CorpusDocument]] = []
        for document in self._documents:
            if namespace is not None and document.namespace != namespace:
                continue
            if any(document.metadata.get(key) != value for key, value in filters.items()):
                continue
            base = _cosine(query_vector, document.vector)
            if reranker == "quality":
                score = 0.85 * base + 0.15 * document.quality
            elif reranker == "freshness":
                score = 0.85 * base + 0.15 * document.freshness
            elif reranker == "quality_freshness":
                score = 0.75 * base + 0.15 * document.quality + 0.10 * document.freshness
            else:
                score = base
            scored.append((score, document))
        scored.sort(key=lambda row: (-row[0], -row[1].quality, -row[1].freshness, row[1].document_id))
        return tuple(
            SearchHit(
                document_id=document.document_id,
                score=round(score, 12),
                namespace=document.namespace,
                freshness=document.freshness,
                quality=document.quality,
                metadata=document.metadata,
            )
            for score, document in scored[:top_k]
        )
