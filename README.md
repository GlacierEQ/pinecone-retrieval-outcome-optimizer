# Retrieval Outcome Optimizer

Independent GlacierEQ portfolio exhibit aligned to **Pinecone** operating themes.

> **Not affiliated.** This repository is not affiliated with, endorsed by, employed by, or deployed at Pinecone. No proprietary access, production deployment, customer impact, or company partnership is claimed.

## Implemented mechanism

`RetrievalOutcomeOptimizer` turns retrieval selection into a measurable constrained decision instead of a generic allow/refuse gate.

Each candidate carries relevance, quality, freshness, latency, and cost. A request declares minimum quality/relevance/freshness, maximum latency, total cost budget, and result limit. Candidates outside the contract are removed; the remaining set is deterministically ranked and selected without exceeding cumulative cost.

The mechanism fails closed for non-finite values, invalid metric ranges, negative budgets, and requests with no candidate satisfying the declared outcome constraints.

## Proof surface

- `src/retrieval_outcome_optimizer.py` — deterministic optimizer
- `tests/test_retrieval_outcome_optimizer.py` — quality, latency, cost, tie-break, invalid-domain tests
- `scripts/operate.py` — direct optimization execution
- `.github/workflows/tests.yml` — pytest + operate CI

## Current boundary

This is a synthetic reference optimizer. It does not call Pinecone APIs or claim measured production search quality. The next gate is replay against a labeled retrieval dataset and then a disposable index.
