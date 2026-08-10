from __future__ import annotations

import math

import pytest

from retrieval_outcome_optimizer import (
    Decision,
    RetrievalCandidate,
    RetrievalConstraints,
    RetrievalOutcomeOptimizer,
)


def candidate(candidate_id, relevance, quality, freshness, latency_ms, cost_units):
    return RetrievalCandidate(candidate_id, relevance, quality, freshness, latency_ms, cost_units)


def test_optimizer_prefers_measurably_stronger_outcome():
    items = [
        candidate("a", .90, .80, .70, 80, 1),
        candidate("b", .70, .70, .95, 50, 1),
    ]
    receipt = RetrievalOutcomeOptimizer().optimize(items)
    assert receipt.decision is Decision.ALLOW
    assert receipt.selected[0].candidate_id == "a"


def test_quality_and_relevance_floors_remove_weak_results():
    items = [candidate("weak", .59, .99, 1.0, 10, 1)]
    receipt = RetrievalOutcomeOptimizer().optimize(items)
    assert receipt.decision is Decision.REFUSE
    assert "no_candidate_satisfies_constraints" in receipt.reasons


def test_latency_ceiling_is_enforced():
    items = [candidate("slow", .99, .99, 1.0, 501, 1)]
    assert RetrievalOutcomeOptimizer().optimize(items).decision is Decision.REFUSE


def test_total_cost_budget_selects_best_affordable_set():
    items = [
        candidate("best", .99, .99, 1.0, 10, 6),
        candidate("second", .90, .90, 1.0, 10, 4),
        candidate("third", .89, .89, 1.0, 10, 4),
    ]
    receipt = RetrievalOutcomeOptimizer().optimize(
        items, RetrievalConstraints(max_total_cost_units=10, limit=3)
    )
    assert [x.candidate_id for x in receipt.selected] == ["best", "second"]
    assert receipt.total_cost_units == 10


def test_ties_are_deterministic_by_latency_cost_then_id():
    items = [
        candidate("z", .8, .8, .8, 20, 1),
        candidate("a", .8, .8, .8, 10, 1),
    ]
    receipt = RetrievalOutcomeOptimizer().optimize(items)
    assert [x.candidate_id for x in receipt.selected] == ["a", "z"]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_candidate_metrics_fail_closed(bad):
    with pytest.raises(ValueError):
        RetrievalOutcomeOptimizer().optimize([candidate("bad", bad, .8, .8, 10, 1)])


def test_out_of_range_metrics_and_negative_cost_rejected():
    with pytest.raises(ValueError, match="out_of_range"):
        RetrievalOutcomeOptimizer().optimize([candidate("bad", 1.1, .8, .8, 10, 1)])
    with pytest.raises(ValueError, match="negative"):
        RetrievalOutcomeOptimizer().optimize([candidate("bad", .8, .8, .8, 10, -1)])
