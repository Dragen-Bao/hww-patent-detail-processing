"""Streaming audit utilities for IPC columns in the large patent-detail CSV."""

from __future__ import annotations

import csv
import io
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

try:
    from .ipc_dictionary import normalize_ipc_code
except ImportError:  # pragma: no cover - direct CLI execution
    from ipc_dictionary import normalize_ipc_code


ROW_START_RE = re.compile(r'^\s*"?=?"{1,4}\d{6}"{1,4}\s*,')
EXPECTED_COLUMNS = 31
IPC_ALL_INDEX = 25
IPC_MAIN_INDEX = 26


def _looks_like_record_start(line: str) -> bool:
    """Detect the export's six-digit ID at the beginning of a physical row."""

    return bool(ROW_START_RE.match(line))


def iter_logical_records(path: Path) -> Iterator[list[str]]:
    """Yield CSV rows after joining physical lines belonging to one patent.

    The source export contains line breaks inside long text fields. Its first
    field is a six-digit quoted ID, which gives us a conservative record
    boundary without loading the 20GB file into memory.
    """

    buffer: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for physical_line in handle:
            line = physical_line.rstrip("\r\n")
            if _looks_like_record_start(line):
                if buffer:
                    yield _parse_prefix(buffer)
                buffer = [line]
            elif buffer:
                buffer.append(line)
        if buffer:
            yield _parse_prefix(buffer)


def _parse_prefix(lines: list[str]) -> list[str]:
    """Parse a logical row, retaining enough leading columns for IPC fields."""

    # IPC columns occur before the large abstract/claims fields. Joining
    # wrapped lines is sufficient for this audit and avoids multiline CSV
    # parsing ambiguity in the original export.
    logical = "".join(lines)
    try:
        return next(csv.reader(io.StringIO(logical), strict=False))
    except csv.Error:
        # The export has irregular quoting in the ID column. The IPC fields
        # are still recoverable by splitting its leading columns.
        return logical.split(",")


def _split_codes(value: str | None) -> list[str]:
    if not value:
        return []
    codes: list[str] = []
    for raw in re.split(r"[;；]", value):
        code = normalize_ipc_code(raw)
        if code and code not in codes:
            codes.append(code)
    return codes


def _header_indices(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader([handle.readline().rstrip("\r\n")]), [])
    normalized = [str(value).strip().lower() for value in header]
    all_index = next((i for i, value in enumerate(normalized) if "ipc分类号" in value or "ipc分类" in value), IPC_ALL_INDEX)
    main_index = next((i for i, value in enumerate(normalized) if "ipc主分类号" in value or "ipc主分类" in value), IPC_MAIN_INDEX)
    return all_index, main_index


def iter_observed_ipc(path: Path) -> Iterator[dict[str, object]]:
    """Yield one normalized row per observed all/main IPC code."""

    all_index, main_index = _header_indices(path)
    for record_no, row in enumerate(iter_logical_records(path), start=1):
        if len(row) <= all_index:
            continue
        all_codes = _split_codes(row[all_index])
        main_codes = _split_codes(row[main_index]) if len(row) > main_index else []
        for field_source, codes in (("all", all_codes), ("main", main_codes)):
            for code in codes:
                yield {
                    "record_no": record_no,
                    "code": code,
                    "ipc_code": code,
                    "field_source": field_source,
                    "is_main": field_source == "main",
                }


def _raw_codes(value: str | None) -> list[tuple[str, str | None]]:
    if not value:
        return []
    return [(raw.strip(), normalize_ipc_code(raw)) for raw in re.split(r"[;；]", value) if raw.strip()]


def _load_dictionary(path: Path) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    if path.suffix.lower() == ".csv":
        dictionary = pd.read_csv(path, encoding="utf-8-sig")
    else:
        dictionary = pd.read_parquet(path)
    if "scheme_version" not in dictionary:
        dictionary["scheme_version"] = "2026.01"
    current = {
        str(row["ipc_code"]): row.to_dict()
        for _, row in dictionary[dictionary["scheme_version"] == "2026.01"].iterrows()
    }
    historical = {
        str(row["ipc_code"]): row.to_dict()
        for _, row in dictionary[dictionary["scheme_version"] == "2025.01"].iterrows()
    }
    return current, historical


def audit_observed_ipc(
    patent_csv: Path,
    dictionary_path: Path,
    output_dir: Path,
    progress_every: int = 100_000,
) -> dict[str, Path]:
    """Stream the patent detail CSV and classify every observed IPC token."""

    output_dir.mkdir(parents=True, exist_ok=True)
    current, historical = _load_dictionary(dictionary_path)
    occurrences: Counter[str] = Counter()
    all_occurrences: Counter[str] = Counter()
    main_occurrences: Counter[str] = Counter()
    malformed: Counter[str] = Counter()
    logical_records = 0
    records_with_all = 0
    records_with_main = 0

    input_bytes = patent_csv.stat().st_size

    all_index, main_index = _header_indices(patent_csv)
    for logical_records, row in enumerate(iter_logical_records(patent_csv), start=1):
        field_values = {
            "all": row[all_index] if len(row) > all_index else "",
            "main": row[main_index] if len(row) > main_index else "",
        }
        for field_source, value in field_values.items():
            tokens = _raw_codes(value)
            if tokens:
                if field_source == "all":
                    records_with_all += 1
                else:
                    records_with_main += 1
            for raw_code, code in tokens:
                if code is None:
                    malformed[raw_code] += 1
                    continue
                occurrences[code] += 1
                if field_source == "all":
                    all_occurrences[code] += 1
                else:
                    main_occurrences[code] += 1
        if logical_records % progress_every == 0:
            print(f"scanned logical_records={logical_records:,}", flush=True)

    rows: list[dict[str, object]] = []
    for code, count in sorted(occurrences.items()):
        current_row = current.get(code)
        historical_row = historical.get(code)
        if current_row and bool(current_row.get("is_valid", True)):
            status = "valid_in_2026"
            dictionary_row = current_row
        elif historical_row and bool(historical_row.get("is_valid", True)):
            status = "valid_in_2025_only"
            dictionary_row = historical_row
        elif current_row:
            status = "present_but_not_valid_in_2026"
            dictionary_row = current_row
        else:
            status = "unmatched"
            dictionary_row = {}
        rows.append(
            {
                "ipc_code": code,
                "occurrence_count": count,
                "all_field_occurrences": all_occurrences[code],
                "main_field_occurrences": main_occurrences[code],
                "status": status,
                "dictionary_version_used": dictionary_row.get("scheme_version"),
                "ipc_level": dictionary_row.get("ipc_level"),
                "parent_code": dictionary_row.get("parent_code"),
                "main_group_code": dictionary_row.get("main_group_code"),
                "title_zh_status": dictionary_row.get("title_zh_status"),
                "title_en": dictionary_row.get("title_en"),
                "title_zh": dictionary_row.get("title_zh"),
            }
        )

    coverage = pd.DataFrame(rows)
    coverage_path = output_dir / "observed_ipc_coverage.csv"
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    unmatched = coverage[coverage["status"] == "unmatched"] if not coverage.empty else coverage
    unmatched_path = output_dir / "observed_ipc_unmatched.csv"
    unmatched.to_csv(unmatched_path, index=False, encoding="utf-8-sig")
    malformed_frame = pd.DataFrame(
        [{"raw_value": raw_value, "occurrence_count": count} for raw_value, count in malformed.most_common()]
    )
    malformed_path = output_dir / "observed_ipc_malformed.csv"
    malformed_frame.to_csv(malformed_path, index=False, encoding="utf-8-sig")

    normalized_occurrences = sum(occurrences.values())
    valid_2026_occurrences = int(
        coverage.loc[coverage["status"] == "valid_in_2026", "occurrence_count"].sum()
    ) if not coverage.empty else 0
    valid_2025_only_occurrences = int(
        coverage.loc[coverage["status"] == "valid_in_2025_only", "occurrence_count"].sum()
    ) if not coverage.empty else 0
    current_rate = valid_2026_occurrences / normalized_occurrences if normalized_occurrences else 0
    historical_rate = valid_2025_only_occurrences / normalized_occurrences if normalized_occurrences else 0
    report_path = output_dir / "patent_ipc_audit_report.md"
    report = f"""# Patent IPC coverage audit

Generated: {datetime.now(timezone.utc).isoformat()}

Input: `{patent_csv}` ({input_bytes:,} bytes)

## Result

- Logical patent records scanned: **{logical_records:,}**
- Records with non-empty `IPC分类号`: **{records_with_all:,}**
- Records with non-empty `IPC主分类号`: **{records_with_main:,}**
- Distinct normalized IPC codes: **{len(coverage):,}**
- Normalized IPC occurrences: **{normalized_occurrences:,}**
- Malformed/non-normalizable tokens: **{sum(malformed.values()):,}** across **{len(malformed):,}** raw values
- Valid in 2026.01: **{valid_2026_occurrences:,}** occurrences ({current_rate:.2%})
- Valid only in 2025.01 bridge: **{valid_2025_only_occurrences:,}** occurrences ({historical_rate:.2%})
- Unmatched normalized codes: **{len(unmatched):,}**

## Interpretation

The 2026.01 rate is current-version dictionary coverage, not proof that every historical patent was classified under 2026.01. Codes in the 2025.01-only bucket are retained for historical interpretation and must not be silently rewritten to a 2026.01 code.

`IPC分类号` and `IPC主分类号` are reported separately in `observed_ipc_coverage.csv`; downstream indicators should choose deliberately between all classifications and the main classification.
"""
    report_path.write_text(report, encoding="utf-8")
    summary_path = output_dir / "patent_ipc_audit_summary.json"
    summary_path.write_text(
        pd.Series(
            {
                "input_bytes": input_bytes,
                "logical_records": logical_records,
                "distinct_normalized_codes": len(coverage),
                "normalized_occurrences": normalized_occurrences,
                "malformed_occurrences": sum(malformed.values()),
                "valid_2026_occurrences": valid_2026_occurrences,
                "valid_2025_only_occurrences": valid_2025_only_occurrences,
                "unmatched_distinct_codes": len(unmatched),
            }
        ).to_json(force_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(report)
    return {
        "coverage": coverage_path,
        "unmatched": unmatched_path,
        "malformed": malformed_path,
        "report": report_path,
        "summary": summary_path,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--patent-csv", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit_observed_ipc(args.patent_csv, args.dictionary, args.output_dir)


if __name__ == "__main__":
    main()
