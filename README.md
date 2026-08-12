# Retrieval Outcome Optimizer

Independent GlacierEQ portfolio exhibit aligned to **Pinecone** operating themes.

> **Not affiliated.** This repository is not affiliated with, endorsed by, employed by, or deployed at Pinecone. No proprietary access, production deployment, customer impact, or company partnership is claimed.

## Problem

Vector search increasingly appears as a built-in feature of general-purpose databases. A specialized retrieval system earns its keep only if retrieval policy can be tied to the outcome that matters downstream: answer quality, task success, freshness, latency, and cost.

## System

**Retrieval Outcome Optimizer** treats retrieval as an outcome-optimization problem rather than an ANN-recall endpoint.

The repository now contains four connected execution layers:

1. **Constrained candidate optimization** selects retrieval candidates under explicit relevance, quality, freshness, latency, total-cost, and result-count constraints. Invalid metrics and unsatisfied evidence budgets fail closed.
2. **Labeled outcome replay** compares competing retrieval policies on downstream task success, answer quality, freshness, latency, and cost. Empirical-Bayes shrinkage prevents a tiny lucky sample from winning a promotion decision, while explicit sample and promotion-margin requirements bound policy changes.
3. **Schema-validated labeled corpus ingestion + disposable vector index** loads real vector/query/relevance records with provenance and integrity checks, then executes deterministic exact-cosine search with namespace selection, metadata filters, top-k, and optional quality/freshness reranking.
4. **Reproducible indexed experiments** run multiple retrieval policies against the labeled index, capture every observed hit and downstream outcome, replay the policies, and emit a content-addressed experiment receipt containing corpus identity, policy configuration, query traces, decision evidence, and experiment digest.

The system is independent of Pinecone APIs. The disposable index is a real searchable index implemented in this repository, not a mocked response fixture.

## Install and run

```bash
python -m pytest -q
python scripts/operate.py
python -m pip install build
python -m build
python -m pip install dist/*.whl
```

### Replay pre-observed labeled outcomes

```bash
retrieval-outcome-replay \
  examples/labeled_outcome_replay.json \
  --incumbent baseline \
  --output replay-receipt.json
```

### Run a complete indexed experiment

```bash
retrieval-outcome-experiment \
  examples/labeled_vector_corpus.json \
  examples/indexed_experiment_config.json \
  --output experiment-receipt.json
```

The indexed example loads the corpus, builds the disposable vector index, executes both policies query-by-query, captures observed retrieval traces, calculates downstream outcome metrics, and emits the final policy replay decision. Refusal/error paths use non-zero exits so both commands can participate directly in CI or a promotion workflow.

## Evidence and failure behavior

The corpus loader refuses malformed vectors, dimension mismatches, duplicate identifiers, non-finite metrics, and relevance labels that reference unknown documents.

The vector index refuses invalid dimensions, top-k values, filters, and reranker modes. Policy replay refuses unknown-policy observations, duplicate evidence identities, insufficient labeled coverage, an improvement below the required margin, and an incumbent that still wins.

Every major evidence object is content-addressed:

- corpus digest;
- retrieval-trace evidence IDs;
- replay evidence digest;
- experiment digest.

## Proof surface

| Capability | Implementation | Behavioral proof |
|---|---|---|
| Constrained candidate optimizer | `src/retrieval_outcome_optimizer.py` | `tests/test_retrieval_outcome_optimizer.py`, `tests/test_adversarial.py` |
| Labeled policy replay | `src/outcome_replay_optimizer.py` | `tests/test_outcome_replay_optimizer.py` |
| Labeled corpus ingestion | `src/labeled_corpus.py` | `tests/test_indexed_experiment.py` |
| Disposable exact vector index | `src/disposable_vector_index.py` | `tests/test_indexed_experiment.py` |
| Indexed policy experiment | `src/retrieval_experiment.py` | `tests/test_indexed_experiment.py` |
| Installed CLIs | `src/outcome_replay_cli.py`, `src/retrieval_experiment_cli.py` | `.github/workflows/tests.yml` |
| Reproducible indexed evidence | `examples/labeled_vector_corpus.json`, `examples/indexed_experiment_config.json` | `.github/workflows/tests.yml` |

## Current boundary

This is an independent local/reference retrieval evaluation system. The repository-owned indexed corpus is synthetic reference evidence designed to prove the full execution path; it is not Pinecone production data and does not establish production search quality, customer impact, or company affiliation. External real-world corpora can strengthen empirical evidence without changing the implemented material capability model. Terminal `CRYSTALLIZED` status is earned only when the exact branch head has zero material gaps and its build, behavioral tests, installed runtime, examples, and documentation proof are green.
