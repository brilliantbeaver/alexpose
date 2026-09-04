#!/usr/bin/env python3
"""Read-only paper-profile inventory, cohort, target, and split audit."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from laterality.config import canonical_json_digest, load_context  # noqa: E402
from laterality.data import prepare_cohort  # noqa: E402
from laterality.splitting import build_source_splits  # noqa: E402


def main() -> None:
    context = load_context(SUITE_ROOT / "config" / "protocol.json", profile="paper")
    cohort = prepare_cohort(context)
    split_config = context.protocol["splits"]
    splits = build_source_splits(
        cohort.table,
        context.protocol["data"]["conditions"],
        outer_folds=int(split_config["outer_folds"]),
        inner_folds=int(split_config["inner_folds"]),
        seed=int(split_config["seed"]),
    )
    exclusion_counts = Counter(
        str(item["reason"]) for item in cohort.attrition.get("exclusions", [])
    )
    payload = {
        "schema": "neurips_laterality_real_data_dry_audit/v1",
        "evidence_created": False,
        "protocol_digest": context.protocol_digest,
        "context_digest": context.context_digest,
        "cohort_digest": cohort.cohort_digest,
        "split_digest": canonical_json_digest(splits),
        "inventory": cohort.attrition["inventory"],
        "attrition": {
            "input_sequences": int(cohort.attrition["input_sequences"]),
            "accepted_sequences": int(cohort.attrition["accepted_sequences"]),
            "excluded_sequences": int(cohort.attrition["excluded_sequences"]),
            "accepted_sources": int(cohort.attrition["accepted_sources"]),
            "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        },
        "target_contract": cohort.attrition["target_contract"],
        "source_census": splits["source_census"],
        "folds": [
            {
                "fold": int(fold["fold"]),
                "train_sources": len(fold["train_sources"]),
                "test_sources": len(fold["test_sources"]),
                "test_source_counts": fold["test_source_counts"],
                "inner_readout_folds": len(fold["inner_readout_folds"]),
            }
            for fold in splits["folds"]
        ],
        "claim_boundary": context.protocol["claim_boundary"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
