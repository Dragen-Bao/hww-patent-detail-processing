"""Canonical patent layer shared by IPC and patent-text subprojects.

The raw patent CSV contains wrapped text fields and duplicate application rows
created by firm-relation mappings. This module converts it into one canonical
row per patent application plus one listed-parent/application edge table.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

EXPECTED_COLUMNS = 31
ROW_START_RE = re.compile(r'^\s*"?=?"{1,4}\d{6}"{1,4}\s*,')
BASELINE_RELATIONS = {"上市公司本身", "上市公司的子公司"}

FIELD_INDEX = {
    "stkcd_raw": 0,
    "company_name": 2,
    "registered_address": 3,
    "relation": 4,
    "stock_short_name": 5,
    "listed_industry": 6,
    "listed_board": 7,
    "patent_title": 8,
    "patent_type": 9,
    "applicant": 10,
    "applicant_type": 11,
    "applicant_address": 12,
    "applicant_region": 13,
    "applicant_city": 14,
    "applicant_district": 15,
    "application_no": 16,
    "application_date": 17,
    "application_year": 18,
    "publication_no": 19,
    "publication_date": 20,
    "publication_year": 21,
    "grant_no": 22,
    "grant_date": 23,
    "grant_year": 24,
    "ipc_all_raw": 25,
    "ipc_main_raw": 26,
    "inventors": 27,
    "abstract": 28,
    "main_claim": 29,
}


def normalize_unicode_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = "".join(
        character if not unicodedata.category(character).startswith("C") else " "
        for character in text
    )
    return re.sub(r"\s+", " ", text).strip()


def normalize_stock_code(value: object) -> str | None:
    text = normalize_unicode_text(value).replace("=", "").replace('"', "").replace("'", "")
    if not text:
        return None
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    match = re.search(r"\d{1,6}", text)
    if not match:
        return None
    code = match.group(0)
    return code.zfill(6) if len(code) <= 6 else None


def normalize_date(value: object) -> str | None:
    text = normalize_unicode_text(value)
    if not text:
        return None
    timestamp = pd.to_datetime(text, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.strftime("%Y-%m-%d")


def normalize_patent_type(value: object) -> str:
    text = normalize_unicode_text(value)
    mapping = {
        "发明申请": "invention",
        "发明公布": "invention",
        "发明授权": "invention",
        "实用新型": "utility_model",
        "外观设计": "design",
    }
    return mapping.get(text, "other")


def normalize_ipc_code(value: object) -> str | None:
    text = normalize_unicode_text(value).upper()
    if not text:
        return None
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"([A-H]\d{2}[A-Z]\d{1,4}/\d{1,6})[IN]$", r"\1", text)
    match = re.match(r"^([A-H]\d{2}[A-Z](?:\d{1,4}/\d{1,6})?)", text)
    return match.group(1) if match else None


def first_ipc(value: object) -> str | None:
    text = normalize_unicode_text(value)
    for raw in re.split(r"[;；]", text):
        code = normalize_ipc_code(raw)
        if code:
            return code
    return None


def all_ipc_codes(value: object) -> list[str]:
    text = normalize_unicode_text(value)
    codes: list[str] = []
    for raw in re.split(r"[;；]", text):
        code = normalize_ipc_code(raw)
        if code and code not in codes:
            codes.append(code)
    return codes


def ipc_main_group(code: object) -> str | None:
    """Return the IPC main-group key, e.g. H01M4/02 -> H01M4."""
    normalized = normalize_ipc_code(code)
    if not normalized:
        return None
    if "/" in normalized:
        return normalized.split("/", 1)[0]
    return normalized if len(normalized) > 4 else None


def normalize_applicant_identity(applicant: object, address: object) -> str:
    """Paper-style identity: normalized applicant name + applicant address."""
    name = normalize_unicode_text(applicant).lower()
    addr = normalize_unicode_text(address).lower()
    addr = re.sub(r"^\d{6}\s*", "", addr)
    name = re.sub(r"[\s,，。.;；:：()（）\-_/]+", "", name)
    addr = re.sub(r"[\s,，。.;；:：()（）\-_/]+", "", addr)
    if not name:
        return ""
    return f"{name}||{addr}" if addr else name


def _parse_first_line(line: str) -> list[str]:
    try:
        return next(csv.reader([line.rstrip("\r\n")], strict=False))
    except csv.Error:
        return []


def _parse_fallback_lines(lines: list[str]) -> list[str]:
    try:
        return next(csv.reader(io.StringIO("".join(lines)), strict=False))
    except csv.Error:
        return "".join(lines).split(",")


def _get(row: list[str], field: str) -> str:
    index = FIELD_INDEX[field]
    return row[index].strip() if len(row) > index else ""


def iter_full_logical_records(path: Path) -> Iterator[list[str]]:
    """Yield complete logical rows from the wrapped-line raw CSV."""
    current_row: list[str] | None = None
    fallback_lines: list[str] | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for physical_line in handle:
            if ROW_START_RE.match(physical_line):
                if fallback_lines is not None:
                    row = _parse_fallback_lines(fallback_lines)
                    yield row + [""] * max(0, EXPECTED_COLUMNS - len(row))
                    fallback_lines = None
                if current_row is not None:
                    yield current_row
                    current_row = None
                first_line = _parse_first_line(physical_line)
                if len(first_line) >= EXPECTED_COLUMNS:
                    current_row = first_line + [""] * max(0, EXPECTED_COLUMNS - len(first_line))
                else:
                    fallback_lines = [physical_line]
            elif fallback_lines is not None:
                fallback_lines.append(physical_line)
                recovered = _parse_fallback_lines(fallback_lines)
                if len(recovered) >= EXPECTED_COLUMNS:
                    current_row = recovered + [""] * max(0, EXPECTED_COLUMNS - len(recovered))
                    fallback_lines = None
            elif current_row is not None:
                continuation = physical_line.rstrip("\r\n")
                target = (
                    FIELD_INDEX["main_claim"]
                    if not _get(current_row, "abstract")
                    else FIELD_INDEX["abstract"]
                )
                current_row[target] = f"{current_row[target]}\n{continuation}".strip()
        if fallback_lines is not None:
            row = _parse_fallback_lines(fallback_lines)
            yield row + [""] * max(0, EXPECTED_COLUMNS - len(row))
        if current_row is not None:
            yield current_row


def record_from_row(row: list[str]) -> dict[str, object]:
    application_date = normalize_date(_get(row, "application_date"))
    grant_date = normalize_date(_get(row, "grant_date"))
    patent_type_group = normalize_patent_type(_get(row, "patent_type"))
    relation = normalize_unicode_text(_get(row, "relation"))
    applicant = normalize_unicode_text(_get(row, "applicant"))
    applicant_address = normalize_unicode_text(_get(row, "applicant_address"))
    main_ipc = first_ipc(_get(row, "ipc_main_raw"))
    grant_no = normalize_unicode_text(_get(row, "grant_no"))
    return {
        "stkcd": normalize_stock_code(_get(row, "stkcd_raw")),
        "stkcd_raw": _get(row, "stkcd_raw"),
        "company_name": normalize_unicode_text(_get(row, "company_name")),
        "registered_address": normalize_unicode_text(_get(row, "registered_address")),
        "relation": relation,
        "stock_short_name": normalize_unicode_text(_get(row, "stock_short_name")),
        "listed_industry": normalize_unicode_text(_get(row, "listed_industry")),
        "listed_board": normalize_unicode_text(_get(row, "listed_board")),
        "patent_title": normalize_unicode_text(_get(row, "patent_title")),
        "patent_type": normalize_unicode_text(_get(row, "patent_type")),
        "patent_type_group": patent_type_group,
        "applicant": applicant,
        "applicant_type": normalize_unicode_text(_get(row, "applicant_type")),
        "applicant_address": applicant_address,
        "applicant_region": normalize_unicode_text(_get(row, "applicant_region")),
        "applicant_city": normalize_unicode_text(_get(row, "applicant_city")),
        "applicant_district": normalize_unicode_text(_get(row, "applicant_district")),
        "applicant_identity_key": normalize_applicant_identity(applicant, applicant_address),
        "application_no": normalize_unicode_text(_get(row, "application_no")),
        "application_date": application_date,
        "publication_no": normalize_unicode_text(_get(row, "publication_no")),
        "publication_date": normalize_date(_get(row, "publication_date")),
        "grant_no": grant_no,
        "grant_date": grant_date,
        "ipc_all_raw": normalize_unicode_text(_get(row, "ipc_all_raw")),
        "ipc_main_raw": normalize_unicode_text(_get(row, "ipc_main_raw")),
        "ipc_codes": all_ipc_codes(_get(row, "ipc_all_raw")),
        "ipc_main_code": main_ipc,
        "ipc_subclass": main_ipc[:4] if main_ipc else None,
        "ipc_domain_key": ipc_main_group(main_ipc),
        "inventors": normalize_unicode_text(_get(row, "inventors")),
        "abstract": normalize_unicode_text(_get(row, "abstract")),
        "main_claim": normalize_unicode_text(_get(row, "main_claim")),
        "baseline_firm_eligible": int(relation in BASELINE_RELATIONS),
        "is_granted_invention": int(
            patent_type_group == "invention" and bool(grant_date or grant_no)
        ),
    }


def _mode_nonempty(values: Iterable[object]) -> str:
    cleaned = [normalize_unicode_text(value) for value in values]
    cleaned = [value for value in cleaned if value]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    return max(counts, key=lambda value: (counts[value], len(value), value))


def _longest(values: Iterable[object]) -> str:
    cleaned = [normalize_unicode_text(value) for value in values]
    return max(cleaned, key=len, default="")


def canonicalize_records(
    records: Iterable[dict[str, object]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one row per application and one listed-parent/application edge table."""
    frame = pd.DataFrame([dict(record) for record in records])
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame = frame[frame["application_no"].astype(str).str.len().gt(0)].copy()
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    canonical_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    for application_no, group in frame.groupby("application_no", sort=False, observed=True):
        app_dates = sorted({value for value in group["application_date"] if value})
        grant_dates = sorted({value for value in group["grant_date"] if value})
        main_codes = [value for value in group["ipc_main_code"] if isinstance(value, str) and value]
        main_code = _mode_nonempty(main_codes)
        applicant = _mode_nonempty(group["applicant"])
        address = _mode_nonempty(group["applicant_address"])
        patent_types = set(group["patent_type_group"].astype(str))
        patent_type_group = (
            "invention"
            if "invention" in patent_types
            else "utility_model"
            if "utility_model" in patent_types
            else "design"
            if "design" in patent_types
            else "other"
        )
        grant_no = _mode_nonempty(group["grant_no"])
        grant_date = grant_dates[0] if grant_dates else None
        abstract = _longest(group["abstract"])
        main_claim = _longest(group["main_claim"])
        patent_title = _longest(group["patent_title"])
        relation_values = sorted(
            {normalize_unicode_text(v) for v in group["relation"] if normalize_unicode_text(v)}
        )
        parent_values = sorted(
            {str(v) for v in group["stkcd"] if isinstance(v, str) and v}
        )
        all_codes = sorted(
            {code for codes in group["ipc_codes"] for code in (codes if isinstance(codes, list) else [])}
        )
        application_date = app_dates[0] if len(app_dates) == 1 else None
        canonical_rows.append(
            {
                "application_no": application_no,
                "application_date": application_date,
                "application_year": int(application_date[:4]) if application_date else None,
                "publication_no": _mode_nonempty(group["publication_no"]),
                "publication_date": _mode_nonempty(group["publication_date"]),
                "grant_no": grant_no,
                "grant_date": grant_date,
                "patent_type": _mode_nonempty(group["patent_type"]),
                "patent_type_group": patent_type_group,
                "patent_title": patent_title,
                "abstract": abstract,
                "main_claim": main_claim,
                "applicant": applicant,
                "applicant_type": _mode_nonempty(group["applicant_type"]),
                "applicant_address": address,
                "applicant_region": _mode_nonempty(group["applicant_region"]),
                "applicant_city": _mode_nonempty(group["applicant_city"]),
                "applicant_district": _mode_nonempty(group["applicant_district"]),
                "applicant_identity_key": normalize_applicant_identity(applicant, address),
                "inventors": _mode_nonempty(group["inventors"]),
                "ipc_all_raw": "; ".join(all_codes),
                "ipc_main_raw": _mode_nonempty(group["ipc_main_raw"]),
                "ipc_main_code": main_code or None,
                "ipc_subclass": main_code[:4] if main_code else None,
                "ipc_domain_key": ipc_main_group(main_code),
                "relation_values": " | ".join(relation_values),
                "listed_parent_values": " | ".join(parent_values),
                "baseline_firm_eligible": int(group["baseline_firm_eligible"].max()),
                "is_granted_invention": int(
                    patent_type_group == "invention" and bool(grant_no or grant_date)
                ),
                "text_present": int(bool(patent_title or abstract or main_claim)),
                "source_record_count": int(len(group)),
                "application_date_conflict": int(len(app_dates) > 1),
                "applicant_conflict": int(
                    group["applicant_identity_key"].replace("", pd.NA).nunique(dropna=True) > 1
                ),
                "text_version_conflict": int(
                    group["abstract"].replace("", pd.NA).nunique(dropna=True) > 1
                    or group["main_claim"].replace("", pd.NA).nunique(dropna=True) > 1
                ),
            }
        )
        for (stkcd, _), edge in group[group["stkcd"].notna()].groupby(
            ["stkcd", "application_no"], observed=True
        ):
            edge_rows.append(
                {
                    "listed_parent": stkcd,
                    "application_no": application_no,
                    "company_name": _mode_nonempty(edge["company_name"]),
                    "relation_values": " | ".join(sorted(set(edge["relation"].astype(str)))),
                    "baseline_firm_eligible": int(edge["baseline_firm_eligible"].max()),
                }
            )
    canonical = pd.DataFrame(canonical_rows).sort_values("application_no").reset_index(drop=True)
    edges = pd.DataFrame(edge_rows)
    if not edges.empty:
        edges = (
            edges.drop_duplicates(["listed_parent", "application_no"])
            .sort_values(["listed_parent", "application_no"])
            .reset_index(drop=True)
        )
    return canonical, edges


def _stable_shard(application_no: str, shard_count: int) -> int:
    digest = hashlib.blake2b(application_no.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % shard_count


def _write_partitioned(frame: pd.DataFrame, root: Path, prefix: str) -> None:
    if frame.empty:
        return
    years = frame["application_date"].astype("string").str[:4]
    for year, part in frame.assign(_year=years).groupby("_year", dropna=False):
        label = "unknown" if pd.isna(year) or not str(year).isdigit() else str(year)
        directory = root / f"application_year={label}"
        directory.mkdir(parents=True, exist_ok=True)
        part.drop(columns="_year").to_parquet(
            directory / f"{prefix}.parquet", index=False, compression="zstd"
        )


def build_canonical_dataset(
    raw_csv: Path,
    output_root: Path,
    *,
    shard_count: int = 128,
    buffer_records: int = 2000,
) -> dict[str, object]:
    """Stream the raw CSV and build reusable canonical patent/firm-edge tables."""
    raw_root = output_root / "raw_parts"
    raw_root.mkdir(parents=True, exist_ok=True)
    buffers: dict[int, list[dict[str, object]]] = {i: [] for i in range(shard_count)}
    part_numbers: Counter[int] = Counter()
    logical_records = 0
    parse_failures = 0

    def flush(shard: int) -> None:
        if not buffers[shard]:
            return
        frame = pd.DataFrame(buffers[shard])
        path = raw_root / f"shard-{shard:04d}-{part_numbers[shard]:05d}.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        part_numbers[shard] += 1
        buffers[shard].clear()

    for row in iter_full_logical_records(raw_csv):
        logical_records += 1
        if len(row) < EXPECTED_COLUMNS:
            parse_failures += 1
            continue
        record = record_from_row(row)
        application_no = str(record.get("application_no") or "")
        if not application_no:
            parse_failures += 1
            continue
        shard = _stable_shard(application_no, shard_count)
        buffers[shard].append(record)
        if len(buffers[shard]) >= buffer_records:
            flush(shard)
    for shard in range(shard_count):
        flush(shard)

    canonical_root = output_root / "canonical_patents"
    edge_root = output_root / "firm_edges"
    patent_count = 0
    edge_count = 0
    granted_invention_count = 0
    for shard in range(shard_count):
        paths = sorted(raw_root.glob(f"shard-{shard:04d}-*.parquet"))
        if not paths:
            continue
        raw = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        canonical, edges = canonicalize_records(raw.to_dict(orient="records"))
        _write_partitioned(canonical, canonical_root, f"shard-{shard:04d}")
        edges_with_date = edges.merge(
            canonical[["application_no", "application_date"]],
            on="application_no",
            how="left",
        )
        _write_partitioned(edges_with_date, edge_root, f"shard-{shard:04d}")
        patent_count += len(canonical)
        edge_count += len(edges)
        granted_invention_count += int(canonical["is_granted_invention"].sum())

    summary = {
        "logical_records": logical_records,
        "parse_failures": parse_failures,
        "canonical_application_count": patent_count,
        "firm_edge_count": edge_count,
        "granted_invention_count": granted_invention_count,
        "shard_count": shard_count,
        "buffer_records": buffer_records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "canonical_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "BASELINE_RELATIONS",
    "build_canonical_dataset",
    "canonicalize_records",
    "ipc_main_group",
    "iter_full_logical_records",
    "normalize_applicant_identity",
    "normalize_ipc_code",
    "record_from_row",
]
