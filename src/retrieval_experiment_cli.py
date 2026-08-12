"""CLI for reproducible labeled retrieval-policy experiments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from labeled_corpus import load_labeled_corpus
from retrieval_experiment import ExperimentPolicy, run_retrieval_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run competing retrieval policies against a labeled disposable vector index")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        corpus = load_labeled_corpus(args.corpus)
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("config_must_be_object")
        policies_raw = config.get("policies")
        if not isinstance(policies_raw, list):
            raise ValueError("policies_missing")
        policies = [ExperimentPolicy(**row) for row in policies_raw]
        receipt = run_retrieval_experiment(
            corpus,
            policies,
            incumbent_policy_id=config.get("incumbent_policy_id"),
            min_samples_per_policy=int(config.get("min_samples_per_policy", 3)),
            min_promotion_margin=float(config.get("min_promotion_margin", 0.02)),
        )
        rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        sys.stdout.write(rendered)
        return 0 if receipt["replay"]["decision"] == "PROMOTE" else 2
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(json.dumps({"decision": "ERROR", "reason": str(exc)}, sort_keys=True) + "\n")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
