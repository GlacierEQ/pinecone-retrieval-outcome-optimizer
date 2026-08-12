from __future__ import annotations

import json
from pathlib import Path

import pytest

from disposable_vector_index import DisposableVectorIndex
from labeled_corpus import load_labeled_corpus
from retrieval_experiment import ExperimentPolicy, run_retrieval_experiment


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "examples" / "labeled_vector_corpus.json"
CONFIG = ROOT / "examples" / "indexed_experiment_config.json"


def test_loads_schema_validated_corpus_with_stable_digest() -> None:
    first = load_labeled_corpus(CORPUS)
    second = load_labeled_corpus(CORPUS)
    assert first.dimension == 3
    assert len(first.documents) == 6
    assert len(first.queries) == 3
    assert first.corpus_digest == second.corpus_digest
    assert len(first.corpus_digest) == 64


def test_disposable_index_runs_real_cosine_search_and_namespace_filter() -> None:
    corpus = load_labeled_corpus(CORPUS)
    index = DisposableVectorIndex(corpus.documents)
    query = corpus.queries[0]
    baseline = index.search(query.vector, top_k=1, filter_equals={"language": "en"})
    fresh = index.search(
        query.vector,
        top_k=1,
        namespace="fresh",
        filter_equals={"language": "en"},
        reranker="quality_freshness",
    )
    assert baseline[0].document_id == "legacy-q1"
    assert fresh[0].document_id == "fresh-q1"
    assert fresh[0].freshness > baseline[0].freshness


def test_end_to_end_experiment_promotes_better_downstream_policy() -> None:
    corpus = load_labeled_corpus(CORPUS)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    policies = [ExperimentPolicy(**row) for row in config["policies"]]
    receipt = run_retrieval_experiment(
        corpus,
        policies,
        incumbent_policy_id=config["incumbent_policy_id"],
        min_samples_per_policy=config["min_samples_per_policy"],
        min_promotion_margin=config["min_promotion_margin"],
    )
    assert receipt["replay"]["decision"] == "PROMOTE"
    assert receipt["replay"]["selected_policy_id"] == "outcome-aware"
    assert len(receipt["experiment_digest"]) == 64
    assert len(receipt["traces"]) == 6
    baseline = [row for row in receipt["traces"] if row["policy_id"] == "baseline"]
    aware = [row for row in receipt["traces"] if row["policy_id"] == "outcome-aware"]
    assert sum(row["task_success"] for row in baseline) == 0.0
    assert sum(row["task_success"] for row in aware) == 3.0


def test_corpus_rejects_unknown_relevance_reference(tmp_path: Path) -> None:
    payload = json.loads(CORPUS.read_text(encoding="utf-8"))
    payload["queries"][0]["relevant_document_ids"] = ["does-not-exist"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="query_references_unknown_document"):
        load_labeled_corpus(path)


def test_corpus_rejects_dimension_mismatch(tmp_path: Path) -> None:
    payload = json.loads(CORPUS.read_text(encoding="utf-8"))
    payload["documents"][0]["vector"] = [1.0, 0.0]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="document_dimension_mismatch"):
        load_labeled_corpus(path)


def test_index_refuses_invalid_reranker() -> None:
    corpus = load_labeled_corpus(CORPUS)
    index = DisposableVectorIndex(corpus.documents)
    with pytest.raises(ValueError, match="reranker_invalid"):
        index.search(corpus.queries[0].vector, top_k=1, reranker="magic")
