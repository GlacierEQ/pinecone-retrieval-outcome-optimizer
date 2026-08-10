#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval_outcome_optimizer import RetrievalCandidate, RetrievalOutcomeOptimizer


def main() -> int:
    candidates = (
        RetrievalCandidate("high-quality", .94, .90, .85, 70.0, 2.0),
        RetrievalCandidate("fresh-fast", .82, .82, .98, 35.0, 1.5),
        RetrievalCandidate("weak", .40, .95, 1.0, 20.0, 1.0),
    )
    receipt = RetrievalOutcomeOptimizer().optimize(candidates)
    print(json.dumps(receipt.as_dict(), sort_keys=True))
    return 0 if receipt.decision.value == "ALLOW" and receipt.selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
