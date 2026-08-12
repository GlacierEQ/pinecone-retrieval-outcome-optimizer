"""Run retrieval policies against a labeled corpus and mint experiment receipts."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from disposable_vector_index import DisposableVectorIndex
from labeled_corpus import LabeledCorpus
from outcome_replay_optimizer import (
    LabeledOutcome,
    OutcomeReplayOptimizer,
    RetrievalPolicy,
)


def _digest(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentPolicy:
    policy_id: str
    namespace: str | None = None
    filter_equals: Mapping[str, str] | None = None
    reranker: str = "none"
    top_k: int = 5
    freshness_window_hours: float = 24.0

    def validate(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("policy_id_missing")
        if self.namespace is not None and (not isinstance(self.namespace, str) or not self.namespace.strip()):
            raise ValueError("policy_namespace_invalid")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k <= 0:
            raise ValueError("policy_top_k_invalid")
        if not isinstance(self.freshness_window_hours, (int, float)) or self.freshness_window_hours <= 0:
            raise ValueError("policy_freshness_window_invalid")
        filters = dict(self.filter_equals or {})
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in filters.items()):
            raise ValueError("policy_filter_invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "namespace": self.namespace,
            "filter_equals": dict(sorted((self.filter_equals or {}).items())),
            "reranker": self.reranker,
            "top_k": self.top_k,
            "freshness_window_hours": self.freshness_window_hours,
        }

    def as_replay_policy(self) -> RetrievalPolicy:
        filters = json.dumps(dict(sorted((self.filter_equals or {}).items())), separators=(",", ":"))
        return RetrievalPolicy(
            policy_id=self.policy_id,
            namespace_strategy=self.namespace or "all",
            filter_profile=filters,
            reranker=self.reranker,
            top_k=self.top_k,
            freshness_window_hours=float(self.freshness_window_hours),
        )


def _query_metrics(relevant: set[str], hits) -> tuple[float, float, float]:
    first_rank: int | None = None
    for index, hit in enumerate(hits, start=1):
        if hit.document_id in relevant:
            first_rank = index
            break
    task_success = 1.0 if first_rank is not None else 0.0
    answer_quality = 0.0 if first_rank is None else 1.0 / first_rank
    freshness = 0.0 if not hits else sum(hit.freshness for hit in hits) / len(hits)
    return task_success, answer_quality, freshness


def run_retrieval_experiment(
    corpus: LabeledCorpus,
    policies: Iterable[ExperimentPolicy],
    *,
    incumbent_policy_id: str | None = None,
    min_samples_per_policy: int = 3,
    min_promotion_margin: float = 0.02,
) -> dict[str, Any]:
    policy_list = list(policies)
    if len(policy_list) < 2:
        raise ValueError("at_least_two_policies_required")
    ids: set[str] = set()
    for policy in policy_list:
        policy.validate()
        if policy.policy_id in ids:
            raise ValueError("duplicate_policy_id")
        ids.add(policy.policy_id)
    if incumbent_policy_id is not None and incumbent_policy_id not in ids:
        raise ValueError("incumbent_policy_unknown")

    index = DisposableVectorIndex(corpus.documents)
    outcomes: list[LabeledOutcome] = []
    traces: list[dict[str, Any]] = []
    for policy in policy_list:
        for query in corpus.queries:
            start = time.perf_counter_ns()
            hits = index.search(
                query.vector,
                top_k=policy.top_k,
                namespace=policy.namespace,
                filter_equals=policy.filter_equals,
                reranker=policy.reranker,
            )
            elapsed_ms = max(0.001, (time.perf_counter_ns() - start) / 1_000_000.0)
            success, quality, freshness = _query_metrics(set(query.relevant_document_ids), hits)
            evidence_id = _digest(
                {
                    "corpus_digest": corpus.corpus_digest,
                    "query_id": query.query_id,
                    "policy": policy.as_dict(),
                    "hits": [hit.as_dict() for hit in hits],
                }
            )
            outcome = LabeledOutcome(
                query_id=query.query_id,
                policy_id=policy.policy_id,
                task_success=success,
                answer_quality=quality,
                latency_ms=elapsed_ms,
                cost_units=round(len(hits) * 0.1, 12),
                freshness_satisfaction=freshness,
                evidence_id=evidence_id,
            )
            outcomes.append(outcome)
            traces.append(
                {
                    "query_id": query.query_id,
                    "policy_id": policy.policy_id,
                    "hits": [hit.as_dict() for hit in hits],
                    "task_success": success,
                    "answer_quality": quality,
                    "freshness_satisfaction": freshness,
                    "latency_ms": elapsed_ms,
                    "cost_units": outcome.cost_units,
                    "evidence_id": evidence_id,
                }
            )

    replay = OutcomeReplayOptimizer(
        min_samples_per_policy=min_samples_per_policy,
        min_promotion_margin=min_promotion_margin,
    ).replay(
        [policy.as_replay_policy() for policy in policy_list],
        outcomes,
        incumbent_policy_id=incumbent_policy_id,
    )
    core = {
        "schema": "glaciereq.retrieval-outcome-experiment.v1",
        "corpus_id": corpus.corpus_id,
        "corpus_digest": corpus.corpus_digest,
        "policies": [policy.as_dict() for policy in sorted(policy_list, key=lambda item: item.policy_id)],
        "incumbent_policy_id": incumbent_policy_id,
        "traces": traces,
        "replay": replay.as_dict(),
    }
    core["experiment_digest"] = _digest(core)
    return core
