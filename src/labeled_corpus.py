"""Schema-validated labeled retrieval corpus ingestion."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def _finite(value: float, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label}_not_numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label}_not_finite")
    return value


def _unit_interval(value: float, label: str) -> float:
    value = _finite(value, label)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label}_out_of_range")
    return value


def _vector(raw: Any, label: str) -> tuple[float, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{label}_vector_missing")
    values = tuple(_finite(item, f"{label}_vector") for item in raw)
    if not any(value != 0.0 for value in values):
        raise ValueError(f"{label}_zero_vector")
    return values


def _digest(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CorpusDocument:
    document_id: str
    vector: tuple[float, ...]
    namespace: str
    metadata: Mapping[str, str]
    freshness: float
    quality: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "vector": list(self.vector),
            "namespace": self.namespace,
            "metadata": dict(sorted(self.metadata.items())),
            "freshness": self.freshness,
            "quality": self.quality,
        }


@dataclass(frozen=True)
class CorpusQuery:
    query_id: str
    vector: tuple[float, ...]
    relevant_document_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "vector": list(self.vector),
            "relevant_document_ids": list(self.relevant_document_ids),
        }


@dataclass(frozen=True)
class LabeledCorpus:
    corpus_id: str
    documents: tuple[CorpusDocument, ...]
    queries: tuple[CorpusQuery, ...]
    dimension: int
    provenance: Mapping[str, str]
    corpus_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "glaciereq.labeled-retrieval-corpus.v1",
            "corpus_id": self.corpus_id,
            "dimension": self.dimension,
            "documents": [item.as_dict() for item in self.documents],
            "queries": [item.as_dict() for item in self.queries],
            "provenance": dict(sorted(self.provenance.items())),
            "corpus_digest": self.corpus_digest,
        }


def load_labeled_corpus(path: str | Path) -> LabeledCorpus:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("corpus_must_be_object")
    corpus_id = payload.get("corpus_id")
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise ValueError("corpus_id_missing")
    documents_raw = payload.get("documents")
    queries_raw = payload.get("queries")
    if not isinstance(documents_raw, list) or not documents_raw:
        raise ValueError("documents_missing")
    if not isinstance(queries_raw, list) or not queries_raw:
        raise ValueError("queries_missing")

    documents: list[CorpusDocument] = []
    document_ids: set[str] = set()
    dimension: int | None = None
    for row in documents_raw:
        if not isinstance(row, dict):
            raise ValueError("document_invalid")
        document_id = row.get("document_id")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id_missing")
        if document_id in document_ids:
            raise ValueError("duplicate_document_id")
        document_ids.add(document_id)
        vector = _vector(row.get("vector"), "document")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ValueError("document_dimension_mismatch")
        namespace = row.get("namespace")
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("document_namespace_missing")
        metadata_raw = row.get("metadata") or {}
        if not isinstance(metadata_raw, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata_raw.items()):
            raise ValueError("document_metadata_invalid")
        documents.append(
            CorpusDocument(
                document_id=document_id,
                vector=vector,
                namespace=namespace,
                metadata=dict(metadata_raw),
                freshness=_unit_interval(row.get("freshness", 0.0), "freshness"),
                quality=_unit_interval(row.get("quality", 0.0), "quality"),
            )
        )

    queries: list[CorpusQuery] = []
    query_ids: set[str] = set()
    for row in queries_raw:
        if not isinstance(row, dict):
            raise ValueError("query_invalid")
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("query_id_missing")
        if query_id in query_ids:
            raise ValueError("duplicate_query_id")
        query_ids.add(query_id)
        vector = _vector(row.get("vector"), "query")
        if len(vector) != dimension:
            raise ValueError("query_dimension_mismatch")
        relevant = row.get("relevant_document_ids")
        if not isinstance(relevant, list) or not relevant or not all(isinstance(item, str) and item for item in relevant):
            raise ValueError("query_relevance_missing")
        unknown = sorted(set(relevant) - document_ids)
        if unknown:
            raise ValueError("query_references_unknown_document:" + ",".join(unknown))
        queries.append(CorpusQuery(query_id=query_id, vector=vector, relevant_document_ids=tuple(relevant)))

    provenance_raw = payload.get("provenance") or {}
    if not isinstance(provenance_raw, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in provenance_raw.items()):
        raise ValueError("provenance_invalid")
    canonical = {
        "corpus_id": corpus_id.strip(),
        "dimension": dimension,
        "documents": [item.as_dict() for item in sorted(documents, key=lambda item: item.document_id)],
        "queries": [item.as_dict() for item in sorted(queries, key=lambda item: item.query_id)],
        "provenance": dict(sorted(provenance_raw.items())),
    }
    return LabeledCorpus(
        corpus_id=corpus_id.strip(),
        documents=tuple(documents),
        queries=tuple(queries),
        dimension=int(dimension or 0),
        provenance=dict(provenance_raw),
        corpus_digest=_digest(canonical),
    )
