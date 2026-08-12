# Retrieval Outcome Optimizer

Independent GlacierEQ portfolio exhibit aligned to **Pinecone** operating themes.

> **Not affiliated.** This repository is not affiliated with, endorsed by, employed by, or deployed at Pinecone. No proprietary access, production deployment, customer impact, or company partnership is claimed.

## What it does

This repository treats retrieval as an **outcome optimization problem**, not merely an ANN-recall problem.

It now has two connected mechanisms:

1. **`RetrievalOutcomeOptimizer`** selects candidates under explicit relevance, quality, freshness, latency, cost, and result-count constraints. It fails closed when the declared evidence budget cannot be satisfied.
2. **`OutcomeReplayOptimizer`** replays labeled downstream outcomes across competing retrieval policies and promotes a policy only when labeled coverage and a configurable evidence margin are strong enough. It evaluates task success, answer quality, freshness, latency, and cost together, then shrinks tiny samples toward a neutral prior so one lucky query cannot win a policy decision.

The second mechanism closes the previous gap between proxy retrieval metrics and downstream answer/task success.

## Policy replay model

A portable `RetrievalPolicy` captures namespace strategy, filter profile, reranker, `top_k`, and freshness window. Each `LabeledOutcome` binds a query/policy pair to task success, answer quality, latency, cost, freshness satisfaction, and an evidence identifier.

Replay is deterministic and fail-closed for:

- unknown policy observations;
- duplicated evidence identities;
- non-finite or out-of-range measurements;
- thin labeled coverage;
- an improvement too small to clear the configured promotion margin;
- an incumbent that still wins after replay.

Every replay returns a SHA-256 evidence digest over policies, observations, scores, and incumbent context.

## Run it

### Development

```bash
python -m pytest -q
python scripts/operate.py
```

### Build and install

```bash
python -m pip install build
python -m build
python -m pip install dist/*.whl
```

### Replay labeled outcomes

```bash
retrieval-outcome-replay \
  examples/labeled_outcome_replay.json \
  --incumbent baseline \
  --output replay-receipt.json
```

The included example produces a `PROMOTE` receipt for the outcome-aware policy because it clears both labeled-coverage and evidence-margin gates. A refusal exits non-zero, making the CLI usable directly in CI and deployment promotion workflows.

## Proof surface

- `src/retrieval_outcome_optimizer.py` — deterministic constrained candidate optimizer
- `src/outcome_replay_optimizer.py` — labeled downstream policy replay and promotion decision
- `src/outcome_replay_cli.py` — installable replay CLI
- `tests/test_retrieval_outcome_optimizer.py` — base optimizer behavior
- `tests/test_outcome_replay_optimizer.py` — labeled replay, thin-data refusal, margin refusal, and evidence-integrity tests
- `tests/test_adversarial.py` — adversarial mechanism coverage
- `scripts/operate.py` — cold-start base optimization execution
- `examples/labeled_outcome_replay.json` — reproducible labeled replay fixture
- `.github/workflows/tests.yml` — pytest, cold-start operation, wheel build/install, and installed-CLI replay verification

## Current boundary

This is an independent reference implementation. The labeled replay fixture is synthetic and does not claim measured Pinecone production quality. The next external-evidence gate is replay against an independently labeled retrieval dataset and then a disposable vector index. The repository no longer depends on that future work for its core mechanism to function, package, install, or execute.
