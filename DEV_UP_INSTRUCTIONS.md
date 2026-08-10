# DEV_UP_INSTRUCTIONS — implementation receipt

**Repository:** `GlacierEQ/pinecone-retrieval-outcome-optimizer`  
**Company lens:** Pinecone (independent; no affiliation)  
**Innovation:** Retrieval Outcome Optimizer

## Completed implementation

The generic scaffold has been replaced by a measurable retrieval optimizer with explicit relevance, quality, freshness, latency, cumulative cost, and result-count constraints.

### Shipped boundaries

- weak relevance/quality/freshness candidates are excluded
- latency ceilings are enforced
- cumulative cost is never exceeded
- deterministic score + latency + cost + id tie-breaking
- non-finite/out-of-range/negative metrics fail closed
- no qualifying result returns an explicit refusal receipt

## Verification contract

`python -m pytest -q` and `python scripts/operate.py` must pass for the current head. This proves the synthetic optimizer, not production Pinecone quality.

## Remaining next gate

Replay against a labeled retrieval set, measure recall/quality/cost tradeoffs, then validate against a disposable Pinecone index while keeping all outcome claims receipt-bound.
