"""Build versioned bilingual IPC catalogs from downloaded official sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ipc_dictionary import merge_version_dictionary, parse_wipo_concordance
from ipc_sources import sha256_file


SECTION_LETTERS = "ABCDEFGH"


def _source_map(source_dir: Path) -> dict[str, dict[str, object]]:
    manifest_path = source_dir / "ipc_source_manifest.json"
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {str(row["filename"]): row for row in payload.get("sources", [])}


def _source_hash(source_dir: Path, filename: str, manifest: dict[str, dict[str, object]]) -> str:
    record = manifest.get(filename)
    if record and record.get("sha256"):
        return str(record["sha256"])
    return sha256_file(source_dir / filename)


def _cnipa_files(source_dir: Path, version: str) -> list[Path]:
    return [
        source_dir / f"{version}版IPC分类表-{section}部.pdf"
        for section in SECTION_LETTERS
    ]


def _decorate(
    rows: list[dict[str, object]],
    source_dir: Path,
    version: str,
    scheme_filename: str,
) -> list[dict[str, object]]:
    manifest = _source_map(source_dir)
    scheme_hash = _source_hash(source_dir, scheme_filename, manifest)
    decorated: list[dict[str, object]] = []
    for row in rows:
        zh_source = row.get("title_zh_source")
        zh_hash = _source_hash(source_dir, str(zh_source), manifest) if zh_source else None
        title_zh = row.get("title_zh")
        status = "matched" if title_zh else "missing_source"
        code = str(row["code"])
        main_group_code = code.split("/", 1)[0] + "/00" if "/" in code else None
        decorated.append(
            {
                "ipc_code": code,
                "ipc_level": row["level"],
                "parent_code": row["parent_code"],
                "main_group_code": main_group_code,
                "title_en": row.get("title_en"),
                "title_zh": title_zh,
                "title_zh_status": status,
                "is_valid": bool(row.get("is_valid_symbol")),
                "scheme_version": version,
                "source_en": scheme_filename,
                "source_zh": zh_source,
                "source_sha256_en": scheme_hash,
                "source_sha256_zh": zh_hash,
                "source_symbol": row.get("source_symbol"),
                "source_kind": row.get("source_kind"),
                "parse_note": "official WIPO scheme joined to CNIPA section PDF"
                if title_zh
                else "no stable Chinese title matched from CNIPA section PDF",
            }
        )
    return decorated


def build_dictionary(source_dir: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for version, scheme_filename, valid_filename, title_list_filename in (
        ("2025.01", "ipc_scheme_20250101.zip", "ipc_valid_symbols_20250101.zip", "EN_ipc_title_list_20250101.zip"),
        ("2026.01", "ipc_scheme_20260101.zip", "ipc_valid_symbols_20260101.zip", "EN_ipc_title_list_20260101.zip"),
    ):
        source_rows = merge_version_dictionary(
            source_dir / scheme_filename,
            source_dir / valid_filename,
            _cnipa_files(source_dir, version),
            version,
            source_dir / title_list_filename,
        )
        rows.extend(_decorate(source_rows, source_dir, version, scheme_filename))

    dictionary = pd.DataFrame(rows).sort_values(["scheme_version", "ipc_code"])
    if dictionary.duplicated(["scheme_version", "ipc_code"]).any():
        raise ValueError("Duplicate IPC code within one scheme version")

    version_parquet = output_dir / "ipc_dictionary_versions_2025_2026.parquet"
    version_csv = output_dir / "ipc_dictionary_versions_2025_2026.csv"
    dictionary.to_parquet(version_parquet, index=False)
    dictionary.to_csv(version_csv, index=False, encoding="utf-8-sig")

    current = dictionary[dictionary["scheme_version"] == "2026.01"].copy()
    current_parquet = output_dir / "ipc_dictionary_2026.01.parquet"
    current_csv = output_dir / "ipc_dictionary_2026.01.csv"
    current.to_parquet(current_parquet, index=False)
    current.to_csv(current_csv, index=False, encoding="utf-8-sig")

    concordance_rows = parse_wipo_concordance(source_dir / "ipc_concordancelist_20260101.zip")
    concordance = pd.DataFrame(concordance_rows)
    concordance_path = output_dir / "ipc_concordance_2025_2026.csv"
    concordance.to_csv(concordance_path, index=False, encoding="utf-8-sig")

    summary = {
        "rows_by_version": dictionary.groupby("scheme_version").size().to_dict(),
        "levels_current": current["ipc_level"].value_counts().sort_index().to_dict(),
        "title_status_current": current["title_zh_status"].value_counts().to_dict(),
        "example_2025": dictionary[
            (dictionary["scheme_version"] == "2025.01")
            & (dictionary["ipc_code"] == "H01L21/3065")
        ].to_dict("records"),
        "example_2026": dictionary[
            (dictionary["scheme_version"] == "2026.01")
            & (dictionary["ipc_code"] == "H01L21/3065")
        ].to_dict("records"),
    }
    summary_path = output_dir / "ipc_dictionary_build_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return {
        "versions_parquet": version_parquet,
        "versions_csv": version_csv,
        "current_parquet": current_parquet,
        "current_csv": current_csv,
        "concordance_csv": concordance_path,
        "summary": summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_dictionary(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
