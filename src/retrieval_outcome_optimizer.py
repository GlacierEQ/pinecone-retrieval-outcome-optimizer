"""Retrieval Outcome Optimizer — independent reference implementation.

Ranks retrieval candidates against explicit quality, relevance, freshness,
latency, and cost constraints. The optimizer is deterministic and fails closed
when a request cannot satisfy its declared evidence budget.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class Decision(str, Enum):
    ALLOW = "ALLOW"
    REFUSE = "REFUSE"


RELEVANCE_WEIGHT = 0.50
QUALITY_WEIGHT = 0.30
FRESHNESS_WEIGHT = 0.20


def _finite(value: float, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{label}_not_finite")
    return float(value)


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class RetrievalCandidate:
    candidate_id: str
    relevance: float
    quality: float
    freshness: float
    latency_ms: float
    cost_units: float


@dataclass(frozen=True)
class RetrievalConstraints:
    min_relevance: float = 0.60
    min_quality: float = 0.60
    min_freshness: float = 0.0
    max_latency_ms: float = 500.0
    max_total_cost_units: float = 10.0
    limit: int = 5


@dataclass(frozen=True)
class OptimizedCandidate:
    candidate_id: str
    score: float
    latency_ms: float
    cost_units: float


@dataclass(frozen=True)
class RetrievalOutcomeReceipt:
    decision: Decision
    reasons: tuple[str, ...]
    selected: tuple[OptimizedCandidate, ...]
    total_cost_units: float
    max_observed_latency_ms: float
    digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reasons": list(self.reasons),
            "selected": [c.__dict__ for c in self.selected],
            "total_cost_units": self.total_cost_units,
            "max_observed_latency_ms": self.max_observed_latency_ms,
            "digest": self.digest,
        }


class RetrievalOutcomeOptimizer:
    def _validate_constraints(self, c: RetrievalConstraints) -> None:
        for label, value in (
            ("min_relevance", c.min_relevance), ("min_quality", c.min_quality),
            ("min_freshness", c.min_freshness), ("max_latency_ms", c.max_latency_ms),
            ("max_total_cost_units", c.max_total_cost_units),
        ):
            _finite(value, label)
        if not 0 <= c.min_relevance <= 1 or not 0 <= c.min_quality <= 1 or not 0 <= c.min_freshness <= 1:
            raise ValueError("threshold_out_of_range")
        if c.max_latency_ms <= 0 or c.max_total_cost_units <= 0:
            raise ValueError("budget_non_positive")
        if not isinstance(c.limit, int) or isinstance(c.limit, bool) or c.limit <= 0:
            raise ValueError("limit_invalid")

    def _score(self, item: RetrievalCandidate) -> float:
        return (
            RELEVANCE_WEIGHT * item.relevance
            + QUALITY_WEIGHT * item.quality
            + FRESHNESS_WEIGHT * item.freshness
        )

    def optimize(
        self,
        candidates: Iterable[RetrievalCandidate],
        constraints: RetrievalConstraints = RetrievalConstraints(),
    ) -> RetrievalOutcomeReceipt:
        self._validate_constraints(constraints)
        eligible: list[RetrievalCandidate] = []
        rejected = 0
        for item in candidates:
            if not item.candidate_id.strip():
                raise ValueError("candidate_id_missing")
            relevance = _finite(item.relevance, "relevance")
            quality = _finite(item.quality, "quality")
            freshness = _finite(item.freshness, "freshness")
            latency = _finite(item.latency_ms, "latency_ms")
            cost = _finite(item.cost_units, "cost_units")
            if not all(0 <= v <= 1 for v in (relevance, quality, freshness)):
                raise ValueError("candidate_metric_out_of_range")
            if latency < 0 or cost < 0:
                raise ValueError("candidate_budget_negative")
            if (
                relevance < constraints.min_relevance
                or quality < constraints.min_quality
                or freshness < constraints.min_freshness
                or latency > constraints.max_latency_ms
            ):
                rejected += 1
                continue
            eligible.append(item)

        eligible.sort(key=lambda item: (-self._score(item), item.latency_ms, item.cost_units, item.candidate_id))
        selected: list[OptimizedCandidate] = []
        spent = 0.0
        for item in eligible:
            if len(selected) >= constraints.limit:
                break
            if spent + item.cost_units > constraints.max_total_cost_units:
                continue
            spent += item.cost_units
            selected.append(OptimizedCandidate(item.candidate_id, round(self._score(item), 12), item.latency_ms, item.cost_units))

        reasons: list[str] = []
        if not selected:
            reasons.append("no_candidate_satisfies_constraints")
        decision = Decision.REFUSE if reasons else Decision.ALLOW
        body = {
            "decision": decision.value,
            "selected": [c.__dict__ for c in selected],
            "constraints": constraints.__dict__,
            "rejected": rejected,
            "spent": spent,
        }
        return RetrievalOutcomeReceipt(
            decision=decision,
            reasons=tuple(reasons or ["outcome_budget_satisfied"]),
            selected=tuple(selected),
            total_cost_units=spent,
            max_observed_latency_ms=max((c.latency_ms for c in selected), default=0.0),
            digest=_digest(body),
        )


Mechanism = RetrievalOutcomeOptimizer
