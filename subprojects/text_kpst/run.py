"""Run the quarterly patent-text pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from preprocessing.patent_detail.canonical import build_canonical_dataset
from subprojects.text_kpst.src.aggregation import aggregate
from subprojects.text_kpst.src.scalable import build_vectors, score_all_quarters

CONFIG = Path(__file__).with_name("config") / "default.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "canonical", "vectors", "score", "aggregate"],
        default="all",
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    raw = root / cfg["patent_csv"]
    work = root / cfg["work_dir"]
    output = root / cfg["output_dir"]

    if args.stage in {"all", "canonical"}:
        build_canonical_dataset(raw, work, shard_count=cfg["shard_count"])

    if args.stage in {"all", "vectors"}:
        build_vectors(
            work / "canonical_patents",
            work / "vectors",
            text_fields=cfg["text_fields"],
        )

    if args.stage in {"all", "score"}:
        score_all_quarters(
            work / "vectors",
            work / "patent_scores",
            window=cfg["window_quarters"],
        )

    if args.stage in {"all", "aggregate"}:
        aggregate(
            work,
            work / "patent_scores",
            output,
            start_quarter=cfg["formal_sample_start_quarter"],
        )


if __name__ == "__main__":
    main()
