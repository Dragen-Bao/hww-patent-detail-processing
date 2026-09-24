"""Command-line runner for the Chinese-patent KPST text pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from preprocessing.patent_detail.canonical import build_canonical_dataset
from subprojects.text_kpst.src.aggregation import aggregate_outputs
from subprojects.text_kpst.src.scalable import (
    build_vector_shards,
    score_all_years,
    tokenize_and_build_vocabulary,
)

DEFAULT_CONFIG = Path(__file__).with_name("config") / "default.json"


def _resolve(root: Path, value: str | None) -> Path | None:
    if value in {None, ""}:
        return None
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def run_pipeline(config_path: Path, *, stage: str = "all", force: bool = False) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = Path(__file__).resolve().parents[2]
    raw_csv = _resolve(project_root, config["patent_csv"])
    canonical_root = _resolve(project_root, config["canonical_output_root"])
    output_root = _resolve(project_root, config["output_root"])
    assert raw_csv is not None and canonical_root is not None and output_root is not None
    vector_root = output_root / "vectors"
    score_root = output_root / "patent_scores"
    results: dict[str, object] = {}

    if stage in {"all", "canonical"}:
        manifest = canonical_root / "canonical_manifest.json"
        if force or not manifest.exists():
            results["canonical"] = build_canonical_dataset(
                raw_csv,
                canonical_root,
                shard_count=int(config.get("canonical_shard_count", 128)),
                buffer_records=int(config.get("canonical_buffer_records", 2000)),
            )
        else:
            results["canonical"] = json.loads(manifest.read_text(encoding="utf-8"))

    if stage in {"all", "vectors"}:
        vector_manifest = vector_root / "vector_manifest.json"
        if force or not vector_manifest.exists():
            vector_root.mkdir(parents=True, exist_ok=True)
            vocabulary, counts = tokenize_and_build_vocabulary(
                canonical_root / "canonical_patents",
                vector_root,
                text_fields=list(config["text_fields"]),
                stopwords_path=_resolve(project_root, config.get("stopwords_path")),
                user_dictionary_path=_resolve(project_root, config.get("user_dictionary_path")),
                keep_single_char=bool(config.get("keep_single_char", True)),
                min_corpus_frequency=int(config.get("min_corpus_frequency", 1)),
            )
            results["vocabulary_size"] = len(vocabulary)
            results["comparison_patents_by_year"] = counts
            results["vectors"] = build_vector_shards(vector_root, vocabulary)
        else:
            results["vectors"] = json.loads(vector_manifest.read_text(encoding="utf-8"))

    if stage in {"all", "score"}:
        score_manifest = score_root / "score_manifest.json"
        if force or not score_manifest.exists():
            results["scores"] = score_all_years(
                vector_root,
                score_root,
                window_years=int(config.get("window_years", 5)),
            )
        else:
            results["scores"] = json.loads(score_manifest.read_text(encoding="utf-8"))

    if stage in {"all", "aggregate"}:
        aggregation_manifest = output_root / "aggregation_manifest.json"
        if force or not aggregation_manifest.exists():
            results["aggregation"] = aggregate_outputs(
                canonical_root,
                score_root,
                output_root,
                formal_sample_start_year=int(config.get("formal_sample_start_year", 1994)),
            )
        else:
            results["aggregation"] = json.loads(aggregation_manifest.read_text(encoding="utf-8"))

    output_root.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "config": config,
        "stage": stage,
        "force": force,
        "results": results,
        "method_notes": {
            "sample": "granted invention patents available in the raw canonical universe",
            "bidf": "equation (3): log(N_prior/(1+df_prior)), strict prior application date",
            "method_6": "backward same IPC main-group domain; forward all IPC; same applicant excluded",
            "applicant_identity": "normalized applicant name + applicant address",
            "field_adjustment": "equations (9)-(10)",
            "future_window": "incomplete last-five-year scores are retained but explicitly flagged",
        },
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Chinese-patent KPST text pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        choices=["all", "canonical", "vectors", "score", "aggregate"],
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run_pipeline(args.config, stage=args.stage, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
