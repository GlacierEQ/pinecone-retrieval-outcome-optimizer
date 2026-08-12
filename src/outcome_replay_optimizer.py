"""Labeled retrieval-outcome replay and policy selection.

This module closes the gap between optimizing proxy retrieval metrics and
optimizing downstream task success. It replays labeled outcomes produced by
multiple retrieval policies, computes a bounded utility with empirical-Bayes
shrinkage, and recommends a policy only when the evidence margin clears an
explicit confidence threshold.

No Pinecone API or proprietary behavior is assumed. Policies are plain,
portable retrieval configurations suitable for offline evaluation against any
labeled retrieval trace.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class ReplayDecision(str, Enum):
    PROMOTE = "PROMOTE"
    REFUSE = "REFUSE"


def _finite(value: float, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label}_not_numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label}_not_finite")
    return value


def _bounded(value: float, label: str) -> float:
    value = _finite(value, label)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label}_out_of_range")
    return value


def _digest(payload: Any) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RetrievalPolicy:
    """Portable retrieval configuration tested by replay."""

    policy_id: str
    namespace_strategy: str
    filter_profile: str
    reranker: str
    top_k: int
    freshness_window_hours: float

    def validate(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("policy_id_missing")
        for label, value in (
            ("namespace_strategy", self.namespace_strategy),
            ("filter_profile", self.filter_profile),
            ("reranker", self.reranker),
        ):
            if not value.strip():
                raise ValueError(f"{label}_missing")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k <= 0:
            raise ValueError("top_k_invalid")
        freshness = _finite(self.freshness_window_hours, "freshness_window_hours")
        if freshness <= 0:
            raise ValueError("freshness_window_non_positive")


@dataclass(frozen=True)
class LabeledOutcome:
    """Observed downstream outcome for one query under one policy."""

    query_id: str
    policy_id: str
    task_success: float
    answer_quality: float
    latency_ms: float
    cost_units: float
    freshness_satisfaction: float
    evidence_id: str

    def validate(self) -> None:
        if not self.query_id.strip():
            raise ValueError("query_id_missing")
        if not self.policy_id.strip():
            raise ValueError("policy_id_missing")
        if not self.evidence_id.strip():
            raise ValueError("evidence_id_missing")
        _bounded(self.task_success, "task_success")
        _bounded(self.answer_quality, "answer_quality")
        _bounded(self.freshness_satisfaction, "freshness_satisfaction")
        if _finite(self.latency_ms, "latency_ms") < 0:
            raise ValueError("latency_negative")
        if _finite(self.cost_units, "cost_units") < 0:
            raise ValueError("cost_negative")


@dataclass(frozen=True)
class ReplayWeights:
    task_success: float = 0.55
    answer_quality: float = 0.25
    freshness: float = 0.10
    latency_penalty: float = 0.05
    cost_penalty: float = 0.05

    def normalized(self) -> "ReplayWeights":
        values = {
            "task_success": _finite(self.task_success, "weight_task_success"),
            "answer_quality": _finite(self.answer_quality, "weight_answer_quality"),
            "freshness": _finite(self.freshness, "weight_freshness"),
            "latency_penalty": _finite(self.latency_penalty, "weight_latency_penalty"),
            "cost_penalty": _finite(self.cost_penalty, "weight_cost_penalty"),
        }
        if any(v < 0 for v in values.values()):
            raise ValueError("weight_negative")
        total = sum(values.values())
        if total <= 0:
            raise ValueError("weight_total_non_positive")
        return ReplayWeights(**{k: v / total for k, v in values.items()})


@dataclass(frozen=True)
class PolicyReplayScore:
    policy_id: str
    sample_count: int
    raw_utility: float
    shrunk_utility: float
    task_success_rate: float
    answer_quality_mean: float
    freshness_mean: float
    latency_mean_ms: float
    cost_mean_units: float


@dataclass(frozen=True)
class OutcomeReplayReceipt:
    decision: ReplayDecision
    selected_policy_id: str | None
    incumbent_policy_id: str | None
    margin: float
    reasons: tuple[str, ...]
    scores: tuple[PolicyReplayScore, ...]
    evidence_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "selected_policy_id": self.selected_policy_id,
            "incumbent_policy_id": self.incumbent_policy_id,
            "margin": self.margin,
            "reasons": list(self.reasons),
            "scores": [score.__dict__ for score in self.scores],
            "evidence_digest": self.evidence_digest,
        }


class OutcomeReplayOptimizer:
    """Select a retrieval policy from labeled downstream outcomes.

    Utility is normalized against observed latency/cost maxima and then shrunk
    toward a neutral prior. Shrinkage prevents a tiny lucky sample from beating
    a policy with substantial evidence. Promotion requires both minimum sample
    coverage and a configurable evidence margin over the incumbent/best rival.
    """

    def __init__(
        self,
        *,
        min_samples_per_policy: int = 3,
        min_promotion_margin: float = 0.02,
        prior_strength: float = 3.0,
        prior_utility: float = 0.50,
        weights: ReplayWeights = ReplayWeights(),
    ) -> None:
        if not isinstance(min_samples_per_policy, int) or isinstance(min_samples_per_policy, bool) or min_samples_per_policy < 1:
            raise ValueError("min_samples_per_policy_invalid")
        margin = _finite(min_promotion_margin, "min_promotion_margin")
        if margin < 0:
            raise ValueError("min_promotion_margin_negative")
        prior_strength = _finite(prior_strength, "prior_strength")
        if prior_strength < 0:
            raise ValueError("prior_strength_negative")
        self.min_samples_per_policy = min_samples_per_policy
        self.min_promotion_margin = margin
        self.prior_strength = prior_strength
        self.prior_utility = _bounded(prior_utility, "prior_utility")
        self.weights = weights.normalized()

    def _utility(self, outcome: LabeledOutcome, *, max_latency: float, max_cost: float) -> float:
        w = self.weights
        latency_ratio = outcome.latency_ms / max_latency if max_latency > 0 else 0.0
        cost_ratio = outcome.cost_units / max_cost if max_cost > 0 else 0.0
        utility = (
            w.task_success * outcome.task_success
            + w.answer_quality * outcome.answer_quality
            + w.freshness * outcome.freshness_satisfaction
            - w.latency_penalty * latency_ratio
            - w.cost_penalty * cost_ratio
        )
        return max(0.0, min(1.0, utility))

    def replay(
        self,
        policies: Iterable[RetrievalPolicy],
        outcomes: Iterable[LabeledOutcome],
        *,
        incumbent_policy_id: str | None = None,
    ) -> OutcomeReplayReceipt:
        policy_list = list(policies)
        outcome_list = list(outcomes)
        if len(policy_list) < 2:
            raise ValueError("at_least_two_policies_required")

        by_id: dict[str, RetrievalPolicy] = {}
        for policy in policy_list:
            policy.validate()
            if policy.policy_id in by_id:
                raise ValueError("duplicate_policy_id")
            by_id[policy.policy_id] = policy
        if incumbent_policy_id is not None and incumbent_policy_id not in by_id:
            raise ValueError("incumbent_policy_unknown")

        seen_observations: set[tuple[str, str, str]] = set()
        grouped: dict[str, list[LabeledOutcome]] = {policy_id: [] for policy_id in by_id}
        for outcome in outcome_list:
            outcome.validate()
            if outcome.policy_id not in by_id:
                raise ValueError("outcome_policy_unknown")
            identity = (outcome.query_id, outcome.policy_id, outcome.evidence_id)
            if identity in seen_observations:
                raise ValueError("duplicate_outcome_evidence")
            seen_observations.add(identity)
            grouped[outcome.policy_id].append(outcome)

        max_latency = max((row.latency_ms for row in outcome_list), default=0.0)
        max_cost = max((row.cost_units for row in outcome_list), default=0.0)
        scores: list[PolicyReplayScore] = []
        for policy_id in sorted(by_id):
            rows = grouped[policy_id]
            n = len(rows)
            if n:
                utilities = [self._utility(row, max_latency=max_latency, max_cost=max_cost) for row in rows]
                raw = sum(utilities) / n
                shrunk = (
                    sum(utilities) + self.prior_strength * self.prior_utility
                ) / (n + self.prior_strength)
                success = sum(row.task_success for row in rows) / n
                quality = sum(row.answer_quality for row in rows) / n
                freshness = sum(row.freshness_satisfaction for row in rows) / n
                latency = sum(row.latency_ms for row in rows) / n
                cost = sum(row.cost_units for row in rows) / n
            else:
                raw = 0.0
                shrunk = self.prior_utility
                success = quality = freshness = latency = cost = 0.0
            scores.append(
                PolicyReplayScore(
                    policy_id=policy_id,
                    sample_count=n,
                    raw_utility=round(raw, 12),
                    shrunk_utility=round(shrunk, 12),
                    task_success_rate=round(success, 12),
                    answer_quality_mean=round(quality, 12),
                    freshness_mean=round(freshness, 12),
                    latency_mean_ms=round(latency, 12),
                    cost_mean_units=round(cost, 12),
                )
            )

        ordered = sorted(scores, key=lambda s: (-s.shrunk_utility, -s.sample_count, s.policy_id))
        best = ordered[0]
        comparison = (
            next(score for score in scores if score.policy_id == incumbent_policy_id)
            if incumbent_policy_id is not None
            else ordered[1]
        )
        margin = best.shrunk_utility - comparison.shrunk_utility
        reasons: list[str] = []
        under_sampled = [s.policy_id for s in scores if s.sample_count < self.min_samples_per_policy]
        if under_sampled:
            reasons.append("insufficient_labeled_coverage:" + ",".join(sorted(under_sampled)))
        if margin < self.min_promotion_margin:
            reasons.append("promotion_margin_not_met")
        if incumbent_policy_id is not None and best.policy_id == incumbent_policy_id:
            reasons.append("incumbent_remains_best")

        decision = ReplayDecision.PROMOTE if not reasons else ReplayDecision.REFUSE
        selected = best.policy_id if decision is ReplayDecision.PROMOTE else None
        evidence_payload = {
            "policies": [policy.__dict__ for policy in sorted(policy_list, key=lambda p: p.policy_id)],
            "outcomes": [outcome.__dict__ for outcome in sorted(outcome_list, key=lambda o: (o.query_id, o.policy_id, o.evidence_id))],
            "scores": [score.__dict__ for score in ordered],
            "incumbent_policy_id": incumbent_policy_id,
        }
        return OutcomeReplayReceipt(
            decision=decision,
            selected_policy_id=selected,
            incumbent_policy_id=incumbent_policy_id,
            margin=round(margin, 12),
            reasons=tuple(reasons or ["labeled_outcome_margin_satisfied"]),
            scores=tuple(ordered),
            evidence_digest=_digest(evidence_payload),
        )
