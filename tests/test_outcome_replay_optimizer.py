from __future__ import annotations

import pytest

from outcome_replay_optimizer import (
    LabeledOutcome,
    OutcomeReplayOptimizer,
    ReplayDecision,
    RetrievalPolicy,
)


def policy(policy_id: str, *, reranker: str) -> RetrievalPolicy:
    return RetrievalPolicy(
        policy_id=policy_id,
        namespace_strategy="tenant+recency",
        filter_profile="language=en",
        reranker=reranker,
        top_k=8,
        freshness_window_hours=24.0,
    )


def outcome(
    query_id: str,
    policy_id: str,
    *,
    success: float,
    quality: float,
    latency: float,
    cost: float,
    freshness: float,
) -> LabeledOutcome:
    return LabeledOutcome(
        query_id=query_id,
        policy_id=policy_id,
        task_success=success,
        answer_quality=quality,
        latency_ms=latency,
        cost_units=cost,
        freshness_satisfaction=freshness,
        evidence_id=f"{query_id}:{policy_id}",
    )


def test_promotes_policy_that_improves_labeled_downstream_outcomes():
    policies = [policy("baseline", reranker="none"), policy("rerank-v2", reranker="cross-encoder")]
    rows = []
    for i in range(5):
        rows.append(outcome(f"q{i}", "baseline", success=0.45, quality=0.58, latency=80, cost=1.0, freshness=0.70))
        rows.append(outcome(f"q{i}", "rerank-v2", success=0.90, quality=0.92, latency=95, cost=1.2, freshness=0.90))

    receipt = OutcomeReplayOptimizer(min_promotion_margin=0.02).replay(
        policies,
        rows,
        incumbent_policy_id="baseline",
    )

    assert receipt.decision is ReplayDecision.PROMOTE
    assert receipt.selected_policy_id == "rerank-v2"
    assert receipt.margin > 0.02
    assert len(receipt.evidence_digest) == 64


def test_refuses_when_labeled_coverage_is_too_thin():
    policies = [policy("baseline", reranker="none"), policy("candidate", reranker="late-interaction")]
    rows = [
        outcome("q1", "baseline", success=0.5, quality=0.5, latency=50, cost=1.0, freshness=0.5),
        outcome("q1", "candidate", success=1.0, quality=1.0, latency=60, cost=1.1, freshness=1.0),
    ]

    receipt = OutcomeReplayOptimizer(min_samples_per_policy=3).replay(policies, rows)

    assert receipt.decision is ReplayDecision.REFUSE
    assert receipt.selected_policy_id is None
    assert any(reason.startswith("insufficient_labeled_coverage") for reason in receipt.reasons)


def test_refuses_when_improvement_margin_is_not_material():
    policies = [policy("a", reranker="none"), policy("b", reranker="none")]
    rows = []
    for i in range(4):
        rows.append(outcome(f"q{i}", "a", success=0.80, quality=0.80, latency=50, cost=1.0, freshness=0.80))
        rows.append(outcome(f"q{i}", "b", success=0.81, quality=0.80, latency=50, cost=1.0, freshness=0.80))

    receipt = OutcomeReplayOptimizer(min_promotion_margin=0.05).replay(policies, rows)

    assert receipt.decision is ReplayDecision.REFUSE
    assert "promotion_margin_not_met" in receipt.reasons


def test_rejects_duplicate_outcome_evidence():
    policies = [policy("a", reranker="none"), policy("b", reranker="cross-encoder")]
    repeated = outcome("q1", "a", success=0.8, quality=0.8, latency=50, cost=1, freshness=0.8)
    with pytest.raises(ValueError, match="duplicate_outcome_evidence"):
        OutcomeReplayOptimizer(min_samples_per_policy=1).replay(
            policies,
            [repeated, repeated, outcome("q1", "b", success=0.8, quality=0.8, latency=50, cost=1, freshness=0.8)],
        )


def test_rejects_unknown_policy_outcome_instead_of_silently_scoring_it():
    policies = [policy("a", reranker="none"), policy("b", reranker="cross-encoder")]
    rows = [outcome("q1", "unknown", success=1, quality=1, latency=1, cost=0, freshness=1)]
    with pytest.raises(ValueError, match="outcome_policy_unknown"):
        OutcomeReplayOptimizer(min_samples_per_policy=1).replay(policies, rows)
