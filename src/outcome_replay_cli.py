"""CLI for labeled retrieval-outcome replay."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from outcome_replay_optimizer import (
    LabeledOutcome,
    OutcomeReplayOptimizer,
    ReplayDecision,
    RetrievalPolicy,
)


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("input_must_be_object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay labeled retrieval outcomes and select a policy only when the evidence margin is sufficient."
    )
    parser.add_argument("input", type=Path, help="JSON file containing policies[] and outcomes[]")
    parser.add_argument("--incumbent")
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--min-margin", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        payload = _load(args.input)
        policies = [RetrievalPolicy(**row) for row in payload.get("policies", [])]
        outcomes = [LabeledOutcome(**row) for row in payload.get("outcomes", [])]
        optimizer = OutcomeReplayOptimizer(
            min_samples_per_policy=args.min_samples,
            min_promotion_margin=args.min_margin,
        )
        receipt = optimizer.replay(
            policies,
            outcomes,
            incumbent_policy_id=args.incumbent or payload.get("incumbent_policy_id"),
        )
        rendered = json.dumps(receipt.as_dict(), indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        sys.stdout.write(rendered)
        return 0 if receipt.decision is ReplayDecision.PROMOTE else 2
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(json.dumps({"decision": "ERROR", "reason": str(exc)}, sort_keys=True) + "\n")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
