"""Build canonical patent records and first-version quarterly IPC metrics.

The module deliberately keeps the patent-detail layer separate from HWW/NVA and
uses the existing logical-record scanner for the malformed, wrapped-line CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subprojects.ipc.src.audit_patent_ipc import ROW_START_RE  # noqa: E402
from subprojects.ipc.src.ipc_dictionary import normalize_ipc_code  # noqa: E402
from preprocessing.patent_detail.config import CONFIG_PATH, load_config  # noqa: E402


EXPECTED_COLUMNS = 31
DEFAULT_CONFIG = load_config(CONFIG_PATH, PROJECT_ROOT)
DEFAULT_SHARD_COUNT = int(DEFAULT_CONFIG["shard_count"])
DEFAULT_SHARD_BUFFER_RECORDS = int(DEFAULT_CONFIG["shard_buffer_records"])
HISTORY_START_DATE = str(DEFAULT_CONFIG["history_start_date"])
FORMAL_SAMPLE_START = str(DEFAULT_CONFIG["formal_sample_start"])
PIPELINE_VERSION = str(DEFAULT_CONFIG["pipeline_version"])
IPC_DICTIONARY_VERSION = str(DEFAULT_CONFIG["ipc_dictionary_version"])
FIRM_SCOPE_VERSION = str(DEFAULT_CONFIG["firm_scope_version"])
COUNTING_RULE_VERSION = str(DEFAULT_CONFIG["counting_rule_version"])
BASELINE_PATENT_TYPES = {"invention", "utility_model"}
IPC_FINE_LEVEL = str(DEFAULT_CONFIG.get("ipc_fine_level", "subclass"))
IPC_COARSE_LEVEL = str(DEFAULT_CONFIG.get("ipc_coarse_level", "class"))
RELATEDNESS_FORMULA = str(DEFAULT_CONFIG.get("relatedness_formula", "Nij/sqrt(Ni*Nj)"))
DISTANCE_FORMULA = str(DEFAULT_CONFIG.get("distance_formula", "1-relatedness"))
CORE_COLUMNS = [
    "application_no",
    "application_date",
    "publication_date",
    "grant_date",
    "publication_no",
    "grant_no",
    "patent_type",
    "patent_type_group",
    "raw_ipc_values",
    "ipc_subclasses",
    "ipc_main_subclass",
    "source_record_count",
    "application_date_conflict",
    "publication_date_conflict",
    "grant_date_conflict",
    "ipc_conflict_flag",
    "metadata_conflict_flag",
]
EDGE_COLUMNS = [
    "stkcd",
    "listed_parent",
    "stkcd_raw",
    "application_no",
    "company_name",
    "relation",
    "relation_type",
    "baseline_eligible",
    "applicant",
    "applicant_type",
]
UNIVERSE_COLUMNS = CORE_COLUMNS + ["eligible_firm_count", "all_firm_count"]
RAW_RECORD_COLUMNS = [
    "stkcd_raw",
    "stkcd",
    "company_name",
    "relation",
    "applicant",
    "applicant_type",
    "application_no",
    "application_date",
    "publication_date",
    "grant_date",
    "publication_no",
    "grant_no",
    "patent_type",
    "patent_type_group",
    "ipc_all_raw",
    "ipc_main_raw",
]
RAW_RECORD_SCHEMA = pa.schema([pa.field(column, pa.string()) for column in RAW_RECORD_COLUMNS])
IPC_FULL_RE = re.compile(r"^[A-H]\d{2}[A-Z](?:\d{1,4}/\d{1,6})?$")
IPC_SUBCLASS_RE = re.compile(r"^[A-H]\d{2}[A-Z]$")
IPC_SUFFIX_RE = re.compile(
    r"([A-H]\s*\d{2}\s*[A-Z]\s*\d{1,4}\s*/\s*\d{1,6})[IN]$",
    flags=re.IGNORECASE,
)

FIELD_INDEX = {
    "stkcd_raw": 0,
    "company_name": 2,
    "relation": 4,
    "applicant": 10,
    "applicant_type": 11,
    "application_no": 16,
    "application_date": 17,
    "publication_no": 19,
    "publication_date": 20,
    "grant_no": 22,
    "grant_date": 23,
    "patent_type": 9,
    "ipc_all_raw": 25,
    "ipc_main_raw": 26,
}


def normalize_stock_code(value: object) -> str | None:
    """Return a six-character A-share code without losing leading zeroes."""

    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace("=", "").replace('"', "").replace("'", "")
    if not text:
        return None
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    if not re.fullmatch(r"\d{1,6}", text):
        match = re.search(r"\d{1,6}", text)
        if not match:
            return None
        text = match.group(0)
    return text.zfill(6) if len(text) <= 6 else None


def normalize_date(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    timestamp = pd.to_datetime(str(value).strip(), errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.strftime("%Y-%m-%d")


def normalize_patent_type(value: object) -> str:
    """Map observed lifecycle labels to a stable patent-type category."""

    text = "" if value is None else str(value).strip()
    explicit = {
        "发明申请": "invention",
        "发明授权": "invention",
        "发明公布": "invention",
        "实用新型": "utility_model",
        "外观设计": "design",
    }
    if text in explicit:
        return explicit[text]
    return "other"


def _clean_ipc_raw(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def parse_ipc_tokens(value: object) -> list[dict[str, str | None]]:
    """Normalize IPC tokens while retaining malformed-token provenance."""

    raw_value = _clean_ipc_raw(value)
    if not raw_value:
        return []
    rows: list[dict[str, str | None]] = []
    seen: set[tuple[str, str]] = set()
    for raw in re.split(r"[;；]", raw_value):
        raw = raw.strip()
        if not raw:
            continue
        normalized = normalize_ipc_code(raw)
        status = "normalized"
        if normalized is None:
            stripped = IPC_SUFFIX_RE.sub(r"\1", raw)
            normalized = normalize_ipc_code(stripped)
            status = "suffix_stripped" if normalized else "unmatched"
        subclass = normalized[:4] if normalized and IPC_SUBCLASS_RE.match(normalized[:4]) else None
        key = (raw, normalized or "")
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "raw_ipc": raw,
                "normalized_full_ipc": normalized,
                "ipc_subclass": subclass,
                "mapping_status": status,
            }
        )
    return rows


def parse_ipc_subclasses(value: object) -> list[str]:
    """Return sorted unique four-character subclass codes."""

    codes = {row["ipc_subclass"] for row in parse_ipc_tokens(value) if row["ipc_subclass"]}
    return sorted(str(code) for code in codes)


def _record_from_row(row: list[str]) -> dict[str, object]:
    def get(name: str) -> str:
        index = FIELD_INDEX[name]
        return row[index].strip() if len(row) > index else ""

    return {
        "stkcd_raw": get("stkcd_raw"),
        "stkcd": normalize_stock_code(get("stkcd_raw")),
        "company_name": get("company_name"),
        "relation": get("relation"),
        "applicant": get("applicant"),
        "applicant_type": get("applicant_type"),
        "application_no": get("application_no"),
        "application_date": normalize_date(get("application_date")),
        "publication_date": normalize_date(get("publication_date")),
        "grant_date": normalize_date(get("grant_date")),
        "publication_no": get("publication_no"),
        "grant_no": get("grant_no"),
        "patent_type": get("patent_type"),
        "patent_type_group": normalize_patent_type(get("patent_type")),
        "ipc_all_raw": get("ipc_all_raw"),
        "ipc_main_raw": get("ipc_main_raw"),
    }


def _parse_fast_first_line(line: str) -> list[str]:
    try:
        return next(csv.reader([line.rstrip("\r\n")], strict=False))
    except csv.Error:
        return []


def _parse_fallback_lines(lines: list[str]) -> list[str]:
    try:
        return next(csv.reader(io.StringIO("".join(lines)), strict=False))
    except csv.Error:
        return "".join(lines).split(",")


def _iter_fast_logical_records(path: Path) -> Iterator[list[str]]:
    """Read the first 27 columns without materializing long text columns."""

    fallback_lines: list[str] | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for physical_line in handle:
            if ROW_START_RE.match(physical_line):
                if fallback_lines is not None:
                    yield _parse_fallback_lines(fallback_lines)
                first_line = _parse_fast_first_line(physical_line)
                if len(first_line) >= 27:
                    yield first_line + [""] * max(0, EXPECTED_COLUMNS - len(first_line))
                    fallback_lines = None
                else:
                    fallback_lines = [physical_line]
            elif fallback_lines is not None:
                fallback_lines.append(physical_line)
                recovered = _parse_fallback_lines(fallback_lines)
                if len(recovered) >= 27:
                    yield recovered + [""] * max(0, EXPECTED_COLUMNS - len(recovered))
                    fallback_lines = None
        if fallback_lines is not None:
            yield _parse_fallback_lines(fallback_lines)


def iter_selected_records(
    path: Path,
    failures: list[dict[str, object]] | None = None,
    stats: dict[str, int] | None = None,
) -> Iterator[dict[str, object]]:
    """Yield normalized fields from logical records and record parse failures."""

    for record_no, row in enumerate(_iter_fast_logical_records(path), start=1):
        if stats is not None:
            stats["logical_records_seen"] = stats.get("logical_records_seen", 0) + 1
        if len(row) != EXPECTED_COLUMNS:
            if failures is not None:
                failures.append(
                    {
                        "record_no": record_no,
                        "failure_type": "field_count",
                        "field_count": len(row),
                    }
                )
            continue
        record = _record_from_row(row)
        if not record["application_no"]:
            if failures is not None:
                failures.append(
                    {"record_no": record_no, "failure_type": "missing_application_no"}
                )
            continue
        if not record["application_date"] and failures is not None:
            failures.append(
                {"record_no": record_no, "failure_type": "invalid_application_date"}
            )
        failures_count = sum(
            1 for token in parse_ipc_tokens(record["ipc_all_raw"]) if token["mapping_status"] == "unmatched"
        )
        if failures_count and failures is not None:
            failures.append(
                {
                    "record_no": record_no,
                    "failure_type": "unmatched_ipc_token",
                    "count": failures_count,
                }
            )
        yield record


def _eligible_relation(relation: object) -> int:
    return int(str(relation).strip() in {"上市公司本身", "上市公司的子公司"})


def _nonempty_values(rows: list[dict[str, object]], field: str) -> set[str]:
    return {str(row.get(field) or "").strip() for row in rows if str(row.get(field) or "").strip()}


def canonicalize_records(
    records: Iterable[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build core, firm-edge, and baseline-universe frames from a bounded batch."""

    normalized_records: list[dict[str, object]] = []
    for record in records:
        normalized = dict(record)
        if not normalized.get("stkcd"):
            normalized["stkcd"] = normalize_stock_code(normalized.get("stkcd_raw"))
        if not normalized.get("patent_type_group"):
            normalized["patent_type_group"] = normalize_patent_type(
                normalized.get("patent_type")
            )
        normalized_records.append(normalized)

    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in normalized_records:
        application_no = str(record.get("application_no") or "").strip()
        if application_no:
            groups[application_no].append(record)

    core_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    for application_no, group in groups.items():
        app_dates = _nonempty_values(group, "application_date")
        publication_dates = _nonempty_values(group, "publication_date")
        grant_dates = _nonempty_values(group, "grant_date")
        subclasses = sorted(
            {
                token["ipc_subclass"]
                for record in group
                for token in parse_ipc_tokens(record.get("ipc_all_raw"))
                if token["ipc_subclass"]
            }
        )
        main_subclasses = sorted(
            {
                token["ipc_subclass"]
                for record in group
                for token in parse_ipc_tokens(record.get("ipc_main_raw"))
                if token["ipc_subclass"]
            }
        )
        core_rows.append(
            {
                "application_no": application_no,
                "application_date": next(iter(app_dates), None) if len(app_dates) == 1 else None,
                "publication_date": next(iter(publication_dates), None) if len(publication_dates) == 1 else None,
                "grant_date": next(iter(grant_dates), None) if len(grant_dates) == 1 else None,
                "publication_no": next(iter(_nonempty_values(group, "publication_no")), None),
                "grant_no": next(iter(_nonempty_values(group, "grant_no")), None),
                "patent_type": "; ".join(sorted(_nonempty_values(group, "patent_type"))),
                "patent_type_group": (
                    "invention"
                    if any(row.get("patent_type_group") == "invention" for row in group)
                    else next(
                        (row.get("patent_type_group") for row in group if row.get("patent_type_group")),
                        "other",
                    )
                ),
                "raw_ipc_values": "; ".join(sorted(_nonempty_values(group, "ipc_all_raw"))),
                "ipc_subclasses": subclasses,
                "ipc_main_subclass": main_subclasses[0] if main_subclasses else None,
                "source_record_count": len(group),
                "application_date_conflict": int(len(app_dates) > 1),
                "publication_date_conflict": int(len(publication_dates) > 1),
                "grant_date_conflict": int(len(grant_dates) > 1),
                "ipc_conflict_flag": int(len({tuple(parse_ipc_subclasses(r.get("ipc_all_raw"))) for r in group}) > 1),
                "metadata_conflict_flag": int(
                    any(
                        len(_nonempty_values(group, field)) > 1
                        for field in ("publication_no", "grant_no", "patent_type")
                    )
                ),
            }
        )
        for record in group:
            stkcd = record.get("stkcd")
            if not stkcd:
                continue
            edge_rows.append(
                {
                    "stkcd": str(stkcd),
                    "listed_parent": str(stkcd),
                    "stkcd_raw": record.get("stkcd_raw"),
                    "application_no": application_no,
                    "company_name": record.get("company_name"),
                    "relation": record.get("relation"),
                    "relation_type": record.get("relation"),
                    "baseline_eligible": _eligible_relation(record.get("relation")),
                    "applicant": record.get("applicant"),
                    "applicant_type": record.get("applicant_type"),
                }
            )

    core = pd.DataFrame(core_rows, columns=CORE_COLUMNS).sort_values("application_no").reset_index(drop=True)
    edges = pd.DataFrame(edge_rows, columns=EDGE_COLUMNS)
    if not edges.empty:
        edges = (
            edges.groupby(["stkcd", "application_no"], as_index=False)
            .agg(
                listed_parent=("listed_parent", "first"),
                stkcd_raw=("stkcd_raw", "first"),
                company_name=("company_name", "first"),
                relation=(
                    "relation",
                    lambda values: "; ".join(
                        sorted({str(value).strip() for value in values if str(value).strip()})
                    ),
                ),
                relation_type=(
                    "relation_type",
                    lambda values: "; ".join(
                        sorted({str(value).strip() for value in values if str(value).strip()})
                    ),
                ),
                baseline_eligible=("baseline_eligible", "max"),
                applicant=("applicant", "first"),
                applicant_type=("applicant_type", "first"),
            )
        )
        edges = edges.sort_values(["stkcd", "application_no"]).reset_index(drop=True)
    if edges.empty:
        eligible = pd.DataFrame(
            columns=["application_no", "eligible_firm_count", "all_firm_count"]
        )
    else:
        all_counts = edges.groupby("application_no")["stkcd"].nunique()
        eligible_counts = (
            edges.loc[edges["baseline_eligible"].eq(1)]
            .groupby("application_no")["stkcd"]
            .nunique()
        )
        eligible = pd.DataFrame(
            {
                "eligible_firm_count": eligible_counts,
                "all_firm_count": all_counts,
            }
        ).reset_index()
    universe = core.merge(eligible, on="application_no", how="left")
    universe["eligible_firm_count"] = universe["eligible_firm_count"].fillna(0).astype("int64")
    universe["all_firm_count"] = universe["all_firm_count"].fillna(0).astype("int64")
    universe = universe[
        (universe["eligible_firm_count"] > 0)
        & universe["patent_type_group"].isin(BASELINE_PATENT_TYPES)
        & universe["application_date"].notna()
        & universe["application_date_conflict"].eq(0)
    ].reset_index(drop=True)
    universe = universe.reindex(columns=UNIVERSE_COLUMNS)
    return core, edges, universe


def _quarter_from_date(value: object) -> str:
    timestamp = pd.Timestamp(value)
    return f"{timestamp.year}Q{timestamp.quarter}"


def _ipc_edges_from_universe(
    universe: pd.DataFrame,
    current_subclasses: set[str] | None = None,
    historical_subclasses: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_subclasses = current_subclasses or set()
    historical_subclasses = historical_subclasses or set()
    edge_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    for row in universe.to_dict(orient="records"):
        tokens = parse_ipc_tokens(row.get("raw_ipc_values"))
        codes: dict[str, dict[str, object]] = {}
        for token in tokens:
            subclass = token.get("ipc_subclass")
            if not subclass:
                mapping_status = "unmatched"
            elif not current_subclasses and not historical_subclasses:
                mapping_status = "current_subclass"
            elif subclass in current_subclasses:
                mapping_status = "current_subclass"
            elif subclass in historical_subclasses:
                mapping_status = "historical_bridge_subclass"
            else:
                mapping_status = "unmatched"
            mapping_rows.append(
                {
                    "application_no": row["application_no"],
                    "raw_ipc": token.get("raw_ipc"),
                    "normalized_full_ipc": token.get("normalized_full_ipc"),
                    "ipc_subclass": subclass,
                    "mapping_status": mapping_status,
                    "taxonomy_version": (
                        "2026.01"
                        if mapping_status == "current_subclass"
                        else "2025.01"
                        if mapping_status == "historical_bridge_subclass"
                        else None
                    ),
                    "mapping_source": (
                        "ipc_dictionary_2026.01"
                        if mapping_status == "current_subclass"
                        else "ipc_dictionary_2025.01_bridge"
                        if mapping_status == "historical_bridge_subclass"
                        else None
                    ),
                }
            )
            if subclass and mapping_status != "unmatched":
                codes.setdefault(str(subclass), {"mapping_status": mapping_status, "raw_ipc": token.get("raw_ipc")})
        subclasses = sorted(codes)
        if not subclasses:
            continue
        weight = 1.0 / len(subclasses)
        main = row.get("ipc_main_subclass")
        quarter = _quarter_from_date(row["application_date"])
        for subclass in subclasses:
            edge_rows.append(
                {
                    "application_no": row["application_no"],
                    "application_date": row["application_date"],
                    "quarter": quarter,
                    "ipc_subclass": subclass,
                    "fractional_weight": weight,
                    "is_main_ipc": int(subclass == main),
                    "mapping_status": codes[subclass]["mapping_status"],
                    "taxonomy_version": (
                        "2026.01"
                        if codes[subclass]["mapping_status"] == "current_subclass"
                        else "2025.01"
                    ),
                    "raw_ipc": codes[subclass]["raw_ipc"],
                }
            )
    return (
        pd.DataFrame(
            edge_rows,
            columns=[
                "application_no",
                "application_date",
                "quarter",
                "ipc_subclass",
                "fractional_weight",
                "is_main_ipc",
                "mapping_status",
                "taxonomy_version",
                "raw_ipc",
            ],
        ),
        pd.DataFrame(
            mapping_rows,
            columns=[
                "application_no",
                "raw_ipc",
                "normalized_full_ipc",
                "ipc_subclass",
                "mapping_status",
                "taxonomy_version",
                "mapping_source",
            ],
        ),
    )


def _pair_id(ipc_a: str, ipc_b: str) -> str:
    return f"{ipc_a}|{ipc_b}"


def _pair_relatedness(
    ipc_a: str,
    ipc_b: str,
    ipc_counts: Counter[str],
    pair_counts: Counter[str],
) -> float:
    """Return PIT co-classification relatedness from records strictly before now."""

    count_a = int(ipc_counts[ipc_a])
    count_b = int(ipc_counts[ipc_b])
    if count_a <= 0 or count_b <= 0:
        return 0.0
    return float(pair_counts[_pair_id(ipc_a, ipc_b)] / math.sqrt(count_a * count_b))


def _pair_distance(
    ipc_a: str,
    ipc_b: str,
    ipc_counts: Counter[str],
    pair_counts: Counter[str],
) -> float:
    return float(max(0.0, min(1.0, 1.0 - _pair_relatedness(ipc_a, ipc_b, ipc_counts, pair_counts))))


def _pair_events_from_code_map(
    code_map: dict[str, tuple[str, ...]],
    dates: dict[str, str],
    seen_counts: Counter[str] | None = None,
    seen_ipc_counts: Counter[str] | None = None,
    total_pairs: int = 0,
    alpha: float = 1.0,
    taxonomy_size: int = 655,
) -> tuple[pd.DataFrame, Counter[str], Counter[str], int]:
    seen_counts = seen_counts if seen_counts is not None else Counter()
    seen_ipc_counts = seen_ipc_counts if seen_ipc_counts is not None else Counter()
    theoretical_pairs = taxonomy_size * (taxonomy_size - 1) // 2
    output: list[dict[str, object]] = []
    for date, app_items in itertools.groupby(
        sorted(code_map, key=lambda app: (dates[app], app)), key=lambda app: dates[app]
    ):
        apps = list(app_items)
        frozen_total = total_pairs
        frozen_ipc_counts = seen_ipc_counts.copy()
        date_rows: list[dict[str, object]] = []
        for application_no in apps:
            codes = code_map[application_no]
            for ipc_a, ipc_b in itertools.combinations(codes, 2):
                pair_id = _pair_id(ipc_a, ipc_b)
                historical_count = int(seen_counts[pair_id])
                probability = (historical_count + alpha) / (frozen_total + alpha * theoretical_pairs)
                technological_distance = _pair_distance(
                    ipc_a,
                    ipc_b,
                    frozen_ipc_counts,
                    seen_counts,
                )
                date_rows.append(
                    {
                        "application_no": application_no,
                        "application_date": date,
                        "quarter": _quarter_from_date(date),
                        "ipc_a": ipc_a,
                        "ipc_b": ipc_b,
                        "pair_id": pair_id,
                        "historical_pair_count": historical_count,
                        "historical_total_pair_count": frozen_total,
                        "historical_first": int(historical_count == 0),
                        "smoothed_pair_probability": probability,
                        "pair_rarity": -math.log(probability),
                        "technological_relatedness": 1.0 - technological_distance,
                        "technological_distance": technological_distance,
                    }
                )
        output.extend(date_rows)
        for application_no in apps:
            codes = code_map[application_no]
            for ipc_code in codes:
                seen_ipc_counts[ipc_code] += 1
            for ipc_a, ipc_b in itertools.combinations(codes, 2):
                seen_counts[_pair_id(ipc_a, ipc_b)] += 1
                total_pairs += 1
    return pd.DataFrame(output), seen_counts, seen_ipc_counts, total_pairs


def build_pair_events(
    universe: pd.DataFrame,
    core: pd.DataFrame | None = None,
    alpha: float = 1.0,
    taxonomy_size: int = 655,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build IPC edges and PIT pair events from an in-memory universe."""

    del core
    ipc_edges, _ = _ipc_edges_from_universe(universe)
    code_map = {
        application_no: tuple(sorted(group["ipc_subclass"].unique()))
        for application_no, group in ipc_edges.groupby("application_no")
    }
    dates = universe.set_index("application_no")["application_date"].astype(str).to_dict()
    pairs, _, _, _ = _pair_events_from_code_map(
        code_map, dates, alpha=alpha, taxonomy_size=taxonomy_size
    )
    return pairs, ipc_edges


def _entropy_decomposition(mass: pd.Series) -> tuple[float, float]:
    """Decompose subclass entropy into within-class related and between-class unrelated parts."""

    if mass.empty or mass.sum() <= 0:
        return float("nan"), float("nan")
    class_mass = mass.groupby(mass.index.to_series().str[:3].to_numpy()).sum()
    class_probabilities = class_mass / class_mass.sum()
    unrelated = float(-(class_probabilities * class_probabilities.map(math.log)).sum())
    related = 0.0
    for class_code, class_total in class_mass.items():
        within = mass.loc[mass.index.to_series().str[:3].eq(class_code)] / class_total
        related += float((class_total / mass.sum()) * -(within * within.map(math.log)).sum())
    return float(related), unrelated


def _rao_stirling_from_mass(
    mass: pd.Series,
    distance_lookup: dict[tuple[str, str], float],
) -> float:
    """Compute Rao–Stirling using a distance snapshot fixed before the quarter."""

    if mass.empty or mass.sum() <= 0:
        return float("nan")
    probabilities = (mass / mass.sum()).to_dict()
    base = 1.0 - sum(value * value for value in probabilities.values())
    relatedness_correction = 0.0
    for (ipc_a, ipc_b), distance in distance_lookup.items():
        if ipc_a not in probabilities or ipc_b not in probabilities:
            continue
        relatedness = 1.0 - float(distance)
        relatedness_correction += 2.0 * probabilities[ipc_a] * probabilities[ipc_b] * relatedness
    return float(max(0.0, base - relatedness_correction))


def _distance_lookup_from_pairs(pairs_q: pd.DataFrame) -> dict[tuple[str, str], float]:
    if pairs_q.empty or "technological_distance" not in pairs_q:
        return {}
    lookup: dict[tuple[str, str], float] = {}
    for row in pairs_q[["ipc_a", "ipc_b", "technological_distance"]].itertuples(index=False):
        lookup[tuple(sorted((str(row.ipc_a), str(row.ipc_b))))] = float(row.technological_distance)
    return lookup


def _historical_distance_lookup(
    codes: Iterable[str],
    ipc_counts: Counter[str],
    pair_counts: Counter[str],
) -> dict[tuple[str, str], float]:
    code_set = {str(code) for code in codes}
    lookup: dict[tuple[str, str], float] = {}
    for pair_id in pair_counts:
        ipc_a, ipc_b = str(pair_id).split("|", 1)
        if ipc_a in code_set and ipc_b in code_set:
            lookup[(ipc_a, ipc_b)] = _pair_distance(ipc_a, ipc_b, ipc_counts, pair_counts)
    return lookup


def _build_relatedness_distance_matrix(
    ipc_counts: Counter[str],
    pair_counts: Counter[str],
    as_of_date: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    codes = sorted(ipc_counts)
    for index, ipc_a in enumerate(codes):
        for ipc_b in codes[index:]:
            if ipc_a == ipc_b:
                relatedness = 1.0
                distance = 0.0
                cooccurrence = int(ipc_counts[ipc_a])
            else:
                relatedness = _pair_relatedness(ipc_a, ipc_b, ipc_counts, pair_counts)
                distance = 1.0 - relatedness
                cooccurrence = int(pair_counts[_pair_id(ipc_a, ipc_b)])
            rows.append(
                {
                    "ipc_a": ipc_a,
                    "ipc_b": ipc_b,
                    "ipc_count_a": int(ipc_counts[ipc_a]),
                    "ipc_count_b": int(ipc_counts[ipc_b]),
                    "cooccurrence_count": cooccurrence,
                    "relatedness": relatedness,
                    "distance": distance,
                    "as_of_date": as_of_date,
                }
            )
    return pd.DataFrame(rows)


def _quarter_row(
    universe_q: pd.DataFrame,
    ipc_edges_q: pd.DataFrame,
    pairs_q: pd.DataFrame,
    pre_sample: bool,
    type_summary: dict[str, int] | None = None,
    rao_distance_lookup: dict[tuple[str, str], float] | None = None,
) -> dict[str, object]:
    type_summary = type_summary or {}
    patent_count = int(universe_q["application_no"].nunique())
    invention_count = int(
        universe_q.loc[universe_q["patent_type_group"].eq("invention"), "application_no"].nunique()
    )
    utility_count = int(
        universe_q.loc[universe_q["patent_type_group"].eq("utility_model"), "application_no"].nunique()
    )
    design_count = int(type_summary.get("design_patent_count", 0))
    other_count = int(type_summary.get("other_patent_count", 0))
    eligible_all_count = int(type_summary.get("eligible_all_patent_count", patent_count))
    edge_counts = ipc_edges_q.groupby("application_no").size() if not ipc_edges_q.empty else pd.Series(dtype=int)
    coded_count = int(edge_counts.size)
    multi_count = int((edge_counts >= 2).sum())
    single_count = int((edge_counts == 1).sum())
    missing_count = max(0, patent_count - coded_count)
    if ipc_edges_q.empty:
        entropy = float("nan")
        hhi = float("nan")
        unique_subclasses = 0
        related_variety = float("nan")
        unrelated_variety = float("nan")
        average_scope = float("nan")
        median_scope = float("nan")
        rao_stirling = float("nan")
    else:
        mass = ipc_edges_q.groupby("ipc_subclass")["fractional_weight"].sum()
        probabilities = mass / mass.sum()
        entropy = float(-(probabilities * probabilities.map(math.log)).sum())
        hhi = float((probabilities * probabilities).sum())
        unique_subclasses = int(mass.size)
        related_variety, unrelated_variety = _entropy_decomposition(mass)
        scope = ipc_edges_q.groupby("application_no")["ipc_subclass"].nunique()
        average_scope = float(scope.mean()) if not scope.empty else float("nan")
        median_scope = float(scope.median()) if not scope.empty else float("nan")
        rao_stirling = _rao_stirling_from_mass(mass, rao_distance_lookup or {})
    if pairs_q.empty:
        novel_patent_count = 0
        all_pair_count = 0
        novel_pair_count = 0
        rarity_mean = float("nan")
        rarity_pair_weighted = float("nan")
    else:
        novel_by_app = pairs_q.groupby("application_no")["historical_first"].max()
        novel_patent_count = int(novel_by_app.sum())
        all_pair_count = int(len(pairs_q))
        novel_pair_count = int(pairs_q["historical_first"].sum())
        patent_rarity = pairs_q.groupby("application_no")["pair_rarity"].mean()
        rarity_mean = float(patent_rarity.mean())
        rarity_pair_weighted = float(pairs_q["pair_rarity"].mean())
        if "technological_distance" in pairs_q:
            patent_distance = pairs_q.groupby("application_no")["technological_distance"].mean()
            average_distance = float(patent_distance.mean()) if not patent_distance.empty else float("nan")
            median_distance = float(patent_distance.median()) if not patent_distance.empty else float("nan")
        else:
            average_distance = float("nan")
            median_distance = float("nan")
    if pairs_q.empty:
        average_distance = float("nan")
        median_distance = float("nan")
    return {
        "quarter": str(universe_q["quarter"].iloc[0]) if "quarter" in universe_q else _quarter_from_date(universe_q["application_date"].iloc[0]),
        "patent_count": patent_count,
        "invention_patent_count": invention_count,
        "utility_model_patent_count": utility_count,
        "design_patent_count": design_count,
        "other_patent_count": other_count,
        "eligible_all_patent_count": eligible_all_count,
        "design_patent_share": design_count / eligible_all_count if eligible_all_count else float("nan"),
        "ipc_coded_patent_count": coded_count,
        "ipc_coding_rate": coded_count / patent_count if patent_count else float("nan"),
        "ipc_missing_patent_count": missing_count,
        "ipc_missing_rate": missing_count / patent_count if patent_count else float("nan"),
        "unique_ipc_subclass_count": unique_subclasses,
        "ipc_entropy": entropy,
        "ipc_hhi": hhi,
        "ipc_one_minus_hhi": 1.0 - hhi if math.isfinite(hhi) else float("nan"),
        "average_patent_scope": average_scope,
        "median_patent_scope": median_scope,
        "related_variety": related_variety,
        "unrelated_variety": unrelated_variety,
        "average_technological_distance": average_distance,
        "median_technological_distance": median_distance,
        "rao_stirling_diversity": rao_stirling,
        "single_ipc_patent_count": single_count,
        "multi_ipc_patent_count": multi_count,
        "multi_ipc_patent_share": multi_count / coded_count if coded_count else float("nan"),
        "novel_patent_count": novel_patent_count,
        "novel_patent_share": novel_patent_count / multi_count if multi_count else float("nan"),
        "all_ipc_pair_count": all_pair_count,
        "novel_ipc_pair_count": novel_pair_count,
        "novel_pair_share": novel_pair_count / all_pair_count if all_pair_count else float("nan"),
        "pair_rarity_mean": rarity_mean,
        "pair_rarity_pair_weighted_mean": rarity_pair_weighted,
        "history_start_date": HISTORY_START_DATE,
        "pre_sample_initialization_flag": int(pre_sample),
        "ipc_dictionary_version": IPC_DICTIONARY_VERSION,
        "firm_scope_version": FIRM_SCOPE_VERSION,
        "counting_rule_version": COUNTING_RULE_VERSION,
    }


def _add_rolling_metrics(
    metrics: pd.DataFrame, mass_by_quarter: dict[str, pd.Series]
) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    metrics = metrics.copy()
    for column in (
        "patent_count",
        "invention_patent_count",
        "novel_patent_count",
        "all_ipc_pair_count",
        "novel_ipc_pair_count",
    ):
        metrics[f"{column}_4q"] = pd.NA
    for column in ("multi_ipc_patent_share", "novel_patent_share", "novel_pair_share"):
        metrics[f"{column}_4q"] = pd.NA
    for column in ("ipc_entropy", "ipc_hhi", "ipc_one_minus_hhi"):
        metrics[f"{column}_4q"] = pd.NA
    for index in range(3, len(metrics)):
        window = metrics.iloc[index - 3 : index + 1]
        for column in (
            "patent_count",
            "invention_patent_count",
            "novel_patent_count",
            "all_ipc_pair_count",
            "novel_ipc_pair_count",
        ):
            metrics.loc[metrics.index[index], f"{column}_4q"] = int(window[column].sum())
        multi_denominator = window["ipc_coded_patent_count"].sum()
        multi_numerator = window["multi_ipc_patent_count"].sum()
        metrics.loc[metrics.index[index], "multi_ipc_patent_share_4q"] = (
            multi_numerator / multi_denominator if multi_denominator else float("nan")
        )
        novel_denominator = window["multi_ipc_patent_count"].sum()
        metrics.loc[metrics.index[index], "novel_patent_share_4q"] = (
            window["novel_patent_count"].sum() / novel_denominator if novel_denominator else float("nan")
        )
        pair_denominator = window["all_ipc_pair_count"].sum()
        metrics.loc[metrics.index[index], "novel_pair_share_4q"] = (
            window["novel_ipc_pair_count"].sum() / pair_denominator if pair_denominator else float("nan")
        )
        masses = [mass_by_quarter.get(str(metrics.iloc[j]["quarter"]), pd.Series(dtype=float)) for j in range(index - 3, index + 1)]
        pooled = pd.concat(masses).groupby(level=0).sum() if any(not mass.empty for mass in masses) else pd.Series(dtype=float)
        if not pooled.empty and pooled.sum() > 0:
            probabilities = pooled / pooled.sum()
            entropy = float(-(probabilities * probabilities.map(math.log)).sum())
            hhi = float((probabilities * probabilities).sum())
            metrics.loc[metrics.index[index], "ipc_entropy_4q"] = entropy
            metrics.loc[metrics.index[index], "ipc_hhi_4q"] = hhi
            metrics.loc[metrics.index[index], "ipc_one_minus_hhi_4q"] = 1.0 - hhi
    return metrics


def aggregate_quarterly_metrics(
    universe: pd.DataFrame, ipc_edges: pd.DataFrame, pairs: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate a bounded canonical universe into continuous quarterly rows."""

    if universe.empty:
        return pd.DataFrame()
    universe = universe.copy()
    universe["quarter"] = universe["application_date"].map(_quarter_from_date)
    if not ipc_edges.empty and "quarter" not in ipc_edges:
        ipc_edges = ipc_edges.copy()
        ipc_edges["quarter"] = ipc_edges["application_date"].map(_quarter_from_date)
    if not pairs.empty and "quarter" not in pairs:
        pairs = pairs.copy()
        pairs["quarter"] = pairs["application_date"].map(_quarter_from_date)
    start = pd.Period(universe["application_date"].min(), freq="Q")
    end = pd.Period(universe["application_date"].max(), freq="Q")
    all_periods = pd.period_range(start, end, freq="Q")
    rows: list[dict[str, object]] = []
    masses: dict[str, pd.Series] = {}
    for period in all_periods:
        quarter = f"{period.year}Q{period.quarter}"
        universe_q = universe[universe["quarter"] == quarter]
        edges_q = ipc_edges[ipc_edges["quarter"] == quarter] if not ipc_edges.empty else ipc_edges
        pairs_q = pairs[pairs["quarter"] == quarter] if not pairs.empty else pairs
        if universe_q.empty:
            row = {
                "quarter": quarter,
                "patent_count": 0,
                "invention_patent_count": 0,
                "utility_model_patent_count": 0,
                "design_patent_count": 0,
                "other_patent_count": 0,
                "eligible_all_patent_count": 0,
                "design_patent_share": float("nan"),
                "ipc_coded_patent_count": 0,
                "ipc_coding_rate": float("nan"),
                "ipc_missing_patent_count": 0,
                "ipc_missing_rate": float("nan"),
                "unique_ipc_subclass_count": 0,
                "ipc_entropy": float("nan"),
                "ipc_hhi": float("nan"),
                "ipc_one_minus_hhi": float("nan"),
                "average_patent_scope": float("nan"),
                "median_patent_scope": float("nan"),
                "related_variety": float("nan"),
                "unrelated_variety": float("nan"),
                "average_technological_distance": float("nan"),
                "median_technological_distance": float("nan"),
                "rao_stirling_diversity": float("nan"),
                "single_ipc_patent_count": 0,
                "multi_ipc_patent_count": 0,
                "multi_ipc_patent_share": float("nan"),
                "novel_patent_count": 0,
                "novel_patent_share": float("nan"),
                "all_ipc_pair_count": 0,
                "novel_ipc_pair_count": 0,
                "novel_pair_share": float("nan"),
                "pair_rarity_mean": float("nan"),
                "pair_rarity_pair_weighted_mean": float("nan"),
                "history_start_date": HISTORY_START_DATE,
                "pre_sample_initialization_flag": int(period.year <= 1993),
                "ipc_dictionary_version": IPC_DICTIONARY_VERSION,
                "firm_scope_version": FIRM_SCOPE_VERSION,
                "counting_rule_version": COUNTING_RULE_VERSION,
            }
        else:
            row = _quarter_row(
                universe_q,
                edges_q,
                pairs_q,
                period.year <= 1993,
                rao_distance_lookup=_distance_lookup_from_pairs(pairs_q),
            )
        rows.append(row)
        masses[quarter] = (
            edges_q.groupby("ipc_subclass")["fractional_weight"].sum()
            if not edges_q.empty
            else pd.Series(dtype=float)
        )
    metrics = pd.DataFrame(rows)
    return _add_rolling_metrics(metrics, masses)


def _stable_shard(application_no: str, shard_count: int) -> int:
    digest = hashlib.blake2b(application_no.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % shard_count


def _write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def _write_sharded_candidates(
    patent_csv: Path,
    output_root: Path,
    shard_count: int,
    buffer_records: int = DEFAULT_SHARD_BUFFER_RECORDS,
) -> dict[str, object]:
    checkpoint = output_root / "01_parse_complete.json"
    if checkpoint.exists():
        summary = json.loads(checkpoint.read_text(encoding="utf-8"))
        current_stat = patent_csv.stat()
        if (
            summary.get("input_bytes") != current_stat.st_size
            or summary.get("input_mtime_ns") != current_stat.st_mtime_ns
            or summary.get("shard_count") != shard_count
            or summary.get("buffer_records") != buffer_records
        ):
            raise RuntimeError(
                "01_parse_complete.json belongs to a different input fingerprint; "
                "use a new output directory"
            )
        return summary
    shard_dir = output_root / "raw_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    existing_shards = list(shard_dir.glob("shard-*.parquet"))
    if existing_shards:
        raise RuntimeError(
            "raw_shards already contains files but 01_parse_complete.json is missing; "
            "use a new output directory or inspect the incomplete run before retrying"
        )
    buffers: dict[int, list[dict[str, object]]] = defaultdict(list)
    failures: list[dict[str, object]] = []
    shard_paths: dict[int, Path] = {}
    scan_stats: dict[str, int] = {"logical_records_seen": 0}
    successful_records = 0
    writers: dict[int, pq.ParquetWriter] = {}
    started = time.monotonic()

    def flush(shard: int) -> None:
        if not buffers[shard]:
            return
        path = shard_dir / f"shard-{shard:04d}.parquet"
        rows = [
            {
                column: None if row.get(column) is None else str(row.get(column))
                for column in RAW_RECORD_COLUMNS
            }
            for row in buffers[shard]
        ]
        table = pa.Table.from_pylist(rows, schema=RAW_RECORD_SCHEMA)
        writer = writers.get(shard)
        if writer is None:
            writer = pq.ParquetWriter(path, RAW_RECORD_SCHEMA, compression="zstd")
            writers[shard] = writer
        writer.write_table(table)
        shard_paths[shard] = path
        buffers[shard].clear()

    try:
        for record in iter_selected_records(patent_csv, failures, scan_stats):
            successful_records += 1
            shard = _stable_shard(str(record["application_no"]), shard_count)
            buffers[shard].append(record)
            if len(buffers[shard]) >= buffer_records:
                flush(shard)
            if scan_stats["logical_records_seen"] % 100_000 == 0:
                elapsed = time.monotonic() - started
                print(
                    f"parse logical_records={scan_stats['logical_records_seen']:,} "
                    f"successful={successful_records:,} elapsed_seconds={elapsed:.1f}",
                    flush=True,
                )
        for shard in list(buffers):
            flush(shard)
    finally:
        for writer in writers.values():
            writer.close()

    failure_frame = pd.DataFrame(failures[:10_000])
    _write_dataframe(failure_frame, output_root / "parse_failures.parquet")
    failure_record_numbers = {row.get("record_no") for row in failures}
    summary = {
        "input_file": str(patent_csv),
        "input_bytes": patent_csv.stat().st_size,
        "input_mtime_ns": patent_csv.stat().st_mtime_ns,
        "pipeline_version": PIPELINE_VERSION,
        "logical_records_seen": scan_stats["logical_records_seen"],
        "successful_records": successful_records,
        "failure_record_count": len(failure_record_numbers),
        "failure_sample_count": len(failures),
        "failure_counts": dict(Counter(str(row["failure_type"]) for row in failures)),
        "shard_count": shard_count,
        "buffer_records": buffer_records,
        "shards_written": len(shard_paths),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_root / "parse_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checkpoint.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _write_partitioned(frame: pd.DataFrame, directory: Path, prefix: str) -> None:
    if frame.empty:
        return
    frame = frame.copy()
    years = pd.to_datetime(frame["application_date"], errors="coerce").dt.year
    frame["application_year"] = years.astype("Int64")
    for year, part in frame.groupby("application_year", dropna=False):
        label = "unknown" if pd.isna(year) else str(int(year))
        _write_dataframe(part.drop(columns=["application_year"]), directory / f"application_year={label}" / f"{prefix}.parquet")


def _load_taxonomy(dictionary_path: Path) -> tuple[set[str], set[str], int]:
    dictionary = pd.read_parquet(dictionary_path)
    current = set(
        dictionary.loc[
            (dictionary["scheme_version"] == "2026.01")
            & dictionary["is_valid"].eq(True)
            & dictionary["ipc_level"].eq("subclass"),
            "ipc_code",
        ].astype(str)
    )
    historical = set(
        dictionary.loc[
            (dictionary["scheme_version"] == "2025.01")
            & dictionary["is_valid"].eq(True)
            & dictionary["ipc_level"].eq("subclass"),
            "ipc_code",
        ].astype(str)
    )
    return current, historical, len(current)


def _process_shards(output_root: Path) -> dict[str, object]:
    checkpoint = output_root / "02_core_complete.json"
    if checkpoint.exists():
        return json.loads(checkpoint.read_text(encoding="utf-8"))
    core_dir = output_root / "patent_core"
    edge_dir = output_root / "patent_firm_edge"
    universe_dir = output_root / "patent_universe_baseline"
    excluded_dir = output_root / "audit" / "excluded_records"
    relation_counts: Counter[str] = Counter()
    core_count = edge_count = universe_count = 0
    source_record_total = 0
    universe_year_counts: Counter[int] = Counter()
    eligible_type_counts: Counter[tuple[str, str]] = Counter()
    excluded_rows: list[dict[str, object]] = []
    conflict_rows: list[dict[str, object]] = []
    for shard_path in sorted((output_root / "raw_shards").glob("shard-*.parquet")):
        records = pd.read_parquet(shard_path).to_dict(orient="records")
        core, edges, universe = canonicalize_records(records)
        shard_id = shard_path.stem
        _write_partitioned(core, core_dir, shard_id)
        _write_partitioned(edges.assign(application_date=edges["application_no"].map(core.set_index("application_no")["application_date"].to_dict())), edge_dir, shard_id) if not edges.empty else None
        _write_partitioned(universe, universe_dir, shard_id)
        core_count += len(core)
        source_record_total += int(core["source_record_count"].sum())
        edge_count += len(edges)
        universe_count += len(universe)
        eligible_application_ids = set(
            edges.loc[edges["baseline_eligible"].eq(1), "application_no"]
            if not edges.empty
            else []
        )
        eligible_core = core.loc[
            core["application_no"].isin(eligible_application_ids)
            & core["application_date"].notna()
            & core["application_date_conflict"].eq(0)
        ].copy()
        if not eligible_core.empty:
            eligible_core["quarter"] = eligible_core["application_date"].map(_quarter_from_date)
            for (quarter, patent_type), count in eligible_core.groupby(
                ["quarter", "patent_type_group"]
            ).size().items():
                eligible_type_counts[(str(quarter), str(patent_type))] += int(count)
        if not universe.empty:
            universe_year_counts.update(
                pd.to_datetime(universe["application_date"], errors="coerce")
                .dt.year.dropna()
                .astype(int)
                .tolist()
            )
        if not edges.empty:
            for value in edges["relation"].fillna("").astype(str):
                relation_counts.update(
                    relation.strip() for relation in value.split("; ") if relation.strip()
                )
        for _, row in core[core["application_date_conflict"].eq(1)].iterrows():
            conflict_rows.append(row.to_dict())
        if not core.empty:
            eligible_map = {
                application_no: 1
                for application_no in eligible_application_ids
            }
            for _, row in core.iterrows():
                app = row["application_no"]
                reason = []
                if app not in eligible_map:
                    reason.append("no_baseline_relation")
                elif row.get("patent_type_group") not in BASELINE_PATENT_TYPES:
                    reason.append("excluded_from_ipc_universe")
                if not row.get("application_date"):
                    reason.append("invalid_application_date")
                if row.get("application_date_conflict"):
                    reason.append("application_date_conflict")
                if reason:
                    excluded_rows.append({"application_no": app, "exclusion_reason": ";".join(reason)})
    _write_dataframe(pd.DataFrame(excluded_rows), excluded_dir / "excluded_records.parquet")
    _write_dataframe(pd.DataFrame(conflict_rows), output_root / "audit" / "date_conflict_audit.parquet")
    (output_root / "audit" / "firm_relation_audit.json").parent.mkdir(parents=True, exist_ok=True)
    (output_root / "audit" / "firm_relation_audit.json").write_text(
        json.dumps(dict(relation_counts), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "unique_application_count": core_count,
        "firm_edge_count": edge_count,
        "baseline_universe_count": universe_count,
        "baseline_universe_year_counts": dict(universe_year_counts),
        "relation_counts": dict(relation_counts),
        "date_conflict_count": len(conflict_rows),
        "excluded_count": len(excluded_rows),
        "eligible_application_type_counts": {
            patent_type: int(sum(count for (_, kind), count in eligible_type_counts.items() if kind == patent_type))
            for patent_type in sorted({kind for _, kind in eligible_type_counts})
        },
    }
    (output_root / "02_core_complete.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "03_firm_edges_complete.json").write_text(
        json.dumps({"firm_edge_count": edge_count}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "audit" / "patent_dedup_audit.json").write_text(
        json.dumps(
            {
                "source_record_count": source_record_total,
                "unique_application_count": core_count,
                "deduplication_reduction": source_record_total - core_count,
                "duplicate_record_count": source_record_total - core_count,
                "date_conflict_count": len(conflict_rows),
                "definition": "application_no is an application-level proxy key, not a patent-family identifier",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    type_rows: list[dict[str, object]] = []
    quarters = sorted({quarter for quarter, _ in eligible_type_counts})
    for quarter in quarters:
        row = {"quarter": quarter}
        for patent_type, column in (
            ("invention", "invention_patent_count"),
            ("utility_model", "utility_model_patent_count"),
            ("design", "design_patent_count"),
            ("other", "other_patent_count"),
        ):
            row[column] = int(eligible_type_counts.get((quarter, patent_type), 0))
        row["eligible_all_patent_count"] = int(
            row["invention_patent_count"]
            + row["utility_model_patent_count"]
            + row["design_patent_count"]
            + row["other_patent_count"]
        )
        type_rows.append(row)
    type_quarterly = pd.DataFrame(type_rows)
    _write_dataframe(type_quarterly, output_root / "audit" / "patent_type_quarterly.parquet")
    type_quarterly.to_csv(
        output_root / "audit" / "patent_type_quarterly.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary


def _partition_files(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.parquet")) if directory.exists() else []


def _process_ipc_and_pairs(
    output_root: Path,
    current_subclasses: set[str],
    historical_subclasses: set[str],
    taxonomy_size: int,
    alpha: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    universe_files = _partition_files(output_root / "patent_universe_baseline")
    by_year: dict[int, list[Path]] = defaultdict(list)
    for path in universe_files:
        match = re.search(r"application_year=(\d{4})", str(path))
        if match:
            by_year[int(match.group(1))].append(path)
    edge_dir = output_root / "patent_ipc_edge"
    pair_dir = output_root / "patent_ipc_pair"
    mapping_counts: Counter[str] = Counter()
    unmatched_examples: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    mass_by_quarter: dict[str, pd.Series] = {}
    seen_counts: Counter[str] = Counter()
    seen_ipc_counts: Counter[str] = Counter()
    total_pairs = 0
    observed_history_years: Counter[int] = Counter()
    history_quarter_rows: list[dict[str, object]] = []
    pre_sample_seen_pair_count = 0
    pre_sample_pair_occurrence_count = 0
    type_summary_by_quarter: dict[str, dict[str, int]] = {}
    type_summary_path = output_root / "audit" / "patent_type_quarterly.parquet"
    if type_summary_path.exists():
        type_summary = pd.read_parquet(type_summary_path)
        type_summary_by_quarter = {
            str(row["quarter"]): {
                key: int(row[key])
                for key in type_summary.columns
                if key != "quarter" and pd.notna(row[key])
            }
            for _, row in type_summary.iterrows()
        }
    for year in sorted(by_year):
        year_universe = pd.concat([pd.read_parquet(path) for path in by_year[year]], ignore_index=True)
        year_universe["quarter"] = year_universe["application_date"].map(_quarter_from_date)
        for quarter, universe_q in year_universe.groupby("quarter", sort=True):
            quarter_start_pair_counts = seen_counts.copy()
            quarter_start_ipc_counts = seen_ipc_counts.copy()
            edge_q, mapping_q = _ipc_edges_from_universe(
                universe_q, current_subclasses, historical_subclasses
            )
            if not mapping_q.empty:
                mapping_counts.update(mapping_q["mapping_status"].astype(str))
                if len(unmatched_examples) < 10_000:
                    unmatched_examples.extend(
                        mapping_q.loc[
                            mapping_q["mapping_status"].eq("unmatched")
                        ].head(10_000 - len(unmatched_examples)).to_dict(orient="records")
                    )
            if not edge_q.empty:
                _write_dataframe(edge_q, edge_dir / f"quarter={quarter}" / "part-000.parquet")
                code_map = {
                    app: tuple(sorted(group["ipc_subclass"].unique()))
                    for app, group in edge_q.groupby("application_no")
                }
                dates = universe_q.set_index("application_no")["application_date"].astype(str).to_dict()
                pairs_q, seen_counts, seen_ipc_counts, total_pairs = _pair_events_from_code_map(
                    code_map,
                    dates,
                    seen_counts=seen_counts,
                    seen_ipc_counts=seen_ipc_counts,
                    total_pairs=total_pairs,
                    alpha=alpha,
                    taxonomy_size=taxonomy_size,
                )
            else:
                pairs_q = pd.DataFrame()
            if not pairs_q.empty:
                _write_dataframe(pairs_q, pair_dir / f"quarter={quarter}" / "part-000.parquet")
            if year <= 1993:
                observed_history_years[year] += len(universe_q)
                pre_sample_seen_pair_count = len(seen_counts)
                pre_sample_pair_occurrence_count = total_pairs
            rao_distance_lookup = _historical_distance_lookup(
                edge_q["ipc_subclass"].unique() if not edge_q.empty else [],
                quarter_start_ipc_counts,
                quarter_start_pair_counts,
            )
            row = _quarter_row(
                universe_q,
                edge_q,
                pairs_q,
                year <= 1993,
                type_summary=type_summary_by_quarter.get(str(quarter)),
                rao_distance_lookup=rao_distance_lookup,
            )
            rows.append(row)
            history_quarter_rows.append(
                {
                    "quarter": quarter,
                    "year": year,
                    "patent_count": row["patent_count"],
                    "multi_ipc_patent_count": row["multi_ipc_patent_count"],
                    "unique_pair_count": int(pairs_q["pair_id"].nunique()) if not pairs_q.empty else 0,
                    "cumulative_seen_pair_count": len(seen_counts),
                    "cumulative_pair_occurrence_count": total_pairs,
                    "pre_sample_initialization_flag": int(year <= 1993),
                }
            )
            mass_by_quarter[quarter] = (
                edge_q.groupby("ipc_subclass")["fractional_weight"].sum()
                if not edge_q.empty
                else pd.Series(dtype=float)
            )
    metrics = pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True) if rows else pd.DataFrame()
    if not metrics.empty:
        min_q = min(
            pd.Period(HISTORY_START_DATE, freq="Q"),
            pd.Period(metrics["quarter"].iloc[0].replace("Q", "-Q"), freq="Q"),
        )
        max_q = pd.Period(metrics["quarter"].iloc[-1].replace("Q", "-Q"), freq="Q")
        expected = pd.period_range(min_q, max_q, freq="Q")
        missing = [f"{period.year}Q{period.quarter}" for period in expected if f"{period.year}Q{period.quarter}" not in set(metrics["quarter"])]
        if missing:
            blank_rows = []
            for quarter in missing:
                year = int(quarter[:4])
                blank_rows.append({
                    "quarter": quarter,
                    "patent_count": 0,
                    "invention_patent_count": 0,
                    "utility_model_patent_count": 0,
                    "design_patent_count": 0,
                    "other_patent_count": 0,
                    "eligible_all_patent_count": 0,
                    "design_patent_share": float("nan"),
                    "ipc_coded_patent_count": 0,
                    "ipc_coding_rate": float("nan"),
                    "ipc_missing_patent_count": 0,
                    "ipc_missing_rate": float("nan"),
                    "unique_ipc_subclass_count": 0,
                    "ipc_entropy": float("nan"),
                    "ipc_hhi": float("nan"),
                    "ipc_one_minus_hhi": float("nan"),
                    "average_patent_scope": float("nan"),
                    "median_patent_scope": float("nan"),
                    "related_variety": float("nan"),
                    "unrelated_variety": float("nan"),
                    "average_technological_distance": float("nan"),
                    "median_technological_distance": float("nan"),
                    "rao_stirling_diversity": float("nan"),
                    "single_ipc_patent_count": 0,
                    "multi_ipc_patent_count": 0,
                    "multi_ipc_patent_share": float("nan"),
                    "novel_patent_count": 0,
                    "novel_patent_share": float("nan"),
                    "all_ipc_pair_count": 0,
                    "novel_ipc_pair_count": 0,
                    "novel_pair_share": float("nan"),
                    "pair_rarity_mean": float("nan"),
                    "pair_rarity_pair_weighted_mean": float("nan"),
                    "history_start_date": HISTORY_START_DATE,
                    "pre_sample_initialization_flag": int(year <= 1993),
                    "ipc_dictionary_version": IPC_DICTIONARY_VERSION,
                    "firm_scope_version": FIRM_SCOPE_VERSION,
                    "counting_rule_version": COUNTING_RULE_VERSION,
                })
            metrics = pd.concat([metrics, pd.DataFrame(blank_rows)], ignore_index=True).sort_values("quarter").reset_index(drop=True)
    metrics = _add_rolling_metrics(metrics, mass_by_quarter)
    history = {
        "history_start_date": HISTORY_START_DATE,
        "formal_sample_start": FORMAL_SAMPLE_START,
        "pre_sample_record_count": int(sum(observed_history_years.values())),
        "pre_sample_year_counts": dict(observed_history_years),
        "pre_sample_seen_pair_count": pre_sample_seen_pair_count,
        "pre_sample_pair_occurrence_count": pre_sample_pair_occurrence_count,
        "seen_pair_count_at_end": len(seen_counts),
        "total_pair_occurrences_at_end": total_pairs,
        "mapping_counts": dict(mapping_counts),
        "ipc_subclass_counts_at_end": dict(seen_ipc_counts),
        "alpha": alpha,
        "taxonomy_subclass_count": taxonomy_size,
        "theoretical_pair_universe": taxonomy_size * (taxonomy_size - 1) // 2,
    }
    as_of_date = str(metrics["quarter"].iloc[-1]) if not metrics.empty else HISTORY_START_DATE
    relatedness_matrix = _build_relatedness_distance_matrix(
        seen_ipc_counts,
        seen_counts,
        as_of_date,
    )
    _write_dataframe(
        relatedness_matrix,
        output_root / "diagnostics" / "ipc_relatedness_distance_matrix.parquet",
    )
    relatedness_matrix.to_csv(
        output_root / "diagnostics" / "ipc_relatedness_distance_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_dataframe(
        pd.DataFrame(
            [{"mapping_status": key, "occurrence_count": value} for key, value in mapping_counts.items()]
        ),
        output_root / "audit" / "ipc_mapping_audit.parquet",
    )
    _write_dataframe(
        pd.DataFrame(unmatched_examples),
        output_root / "audit" / "ipc_unmatched_examples.parquet",
    )
    history_quarterly = pd.DataFrame(history_quarter_rows)
    _write_dataframe(history_quarterly, output_root / "audit" / "pair_history_audit.parquet")
    if not history_quarterly.empty:
        _write_dataframe(
            history_quarterly.groupby("year", as_index=False).agg(
                patent_count=("patent_count", "sum"),
                multi_ipc_patent_count=("multi_ipc_patent_count", "sum"),
                unique_pair_count=("unique_pair_count", "sum"),
                cumulative_seen_pair_count=("cumulative_seen_pair_count", "max"),
                cumulative_pair_occurrence_count=("cumulative_pair_occurrence_count", "max"),
            ),
            output_root / "audit" / "pre_sample_yearly_audit.parquet",
        )
    (output_root / "audit" / "pair_history_audit.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "04_ipc_edges_complete.json").write_text(
        json.dumps({"mapping_counts": dict(mapping_counts)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "05_pair_history_complete.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics, history


def _write_period_summary(metrics: pd.DataFrame, output_root: Path) -> None:
    periods = {
        "1994–1999": (1994, 1999),
        "2000–2009": (2000, 2009),
        "2010–2019": (2010, 2019),
        "2020–2026": (2020, 2026),
    }
    metric_columns = [
        "patent_count",
        "invention_patent_count",
        "utility_model_patent_count",
        "design_patent_count",
        "eligible_all_patent_count",
        "ipc_coded_patent_count",
        "ipc_coding_rate",
        "ipc_entropy",
        "ipc_hhi",
        "ipc_one_minus_hhi",
        "average_patent_scope",
        "median_patent_scope",
        "related_variety",
        "unrelated_variety",
        "average_technological_distance",
        "median_technological_distance",
        "rao_stirling_diversity",
        "multi_ipc_patent_share",
        "novel_pair_share",
        "novel_patent_share",
        "pair_rarity_mean",
    ]
    year = metrics["quarter"].astype(str).str[:4].astype(int)
    rows: list[dict[str, object]] = []
    for label, (start, end) in periods.items():
        subset = metrics.loc[year.between(start, end)]
        for metric in metric_columns:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            rows.append(
                {
                    "period": label,
                    "metric": metric,
                    "quarter_count": int(len(subset)),
                    "nonnull_quarter_count": int(values.size),
                    "sum": float(values.sum()) if not values.empty else float("nan"),
                    "mean": float(values.mean()) if not values.empty else float("nan"),
                    "median": float(values.median()) if not values.empty else float("nan"),
                    "first": float(values.iloc[0]) if not values.empty else float("nan"),
                    "last": float(values.iloc[-1]) if not values.empty else float("nan"),
                    "min": float(values.min()) if not values.empty else float("nan"),
                    "max": float(values.max()) if not values.empty else float("nan"),
                }
            )
    summary = pd.DataFrame(rows)
    _write_dataframe(summary, output_root / "diagnostics" / "period_summary.parquet")
    summary.to_csv(
        output_root / "diagnostics" / "period_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _write_diagnostics(metrics: pd.DataFrame, output_root: Path, skip_plots: bool = False) -> None:
    diagnostics_dir = output_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    _write_dataframe(metrics, diagnostics_dir / "quarterly_metrics_snapshot.parquet")
    metrics.to_csv(diagnostics_dir / "quarterly_metrics_snapshot.csv", index=False, encoding="utf-8-sig")
    _write_period_summary(metrics, output_root)
    if skip_plots or metrics.empty:
        return
    import matplotlib.pyplot as plt

    x = list(range(len(metrics)))
    figure, axes = plt.subplots(4, 2, figsize=(15, 14), sharex=True)
    panels = [
        (["patent_count", "invention_patent_count", "utility_model_patent_count"], "Patent counts"),
        (["ipc_coding_rate", "design_patent_share"], "Coverage and excluded design share"),
        (["ipc_entropy", "related_variety", "unrelated_variety"], "Entropy decomposition"),
        (["ipc_hhi", "ipc_one_minus_hhi"], "HHI concentration"),
        (["average_patent_scope", "median_patent_scope"], "Patent scope"),
        (["average_technological_distance", "median_technological_distance", "rao_stirling_diversity"], "Distance and Rao–Stirling"),
        (["novel_pair_share", "novel_patent_share"], "Historical novelty shares"),
        (["pair_rarity_mean"], "Pair rarity"),
    ]
    formal_quarter = _quarter_from_date(FORMAL_SAMPLE_START)
    for axis, (columns, title) in zip(axes.flat, panels):
        for column in columns:
            axis.plot(x, metrics[column], linewidth=1.0, label=column)
        if formal_quarter in metrics["quarter"].astype(str).tolist():
            sample_index = metrics.index[metrics["quarter"].eq(formal_quarter)]
            if len(sample_index):
                axis.axvline(sample_index[0], color="black", linestyle="--", linewidth=0.8)
        axis.set_title(title)
        axis.legend(fontsize=7, loc="best", frameon=False)
        axis.grid(alpha=0.25)
    tick_positions = list(range(0, len(metrics), 12))
    axes[-1, 0].set_xticks(tick_positions, metrics["quarter"].iloc[tick_positions], rotation=60, ha="right")
    axes[-1, 1].set_xticks(tick_positions, metrics["quarter"].iloc[tick_positions], rotation=60, ha="right")
    figure.suptitle("IPC innovation indicators by application quarter", y=0.995)
    figure.tight_layout()
    figure.savefig(diagnostics_dir / "ipc_innovation_quarterly_series.png", dpi=160)
    plt.close(figure)

    focus_columns = [
        "ipc_entropy",
        "ipc_hhi",
        "average_patent_scope",
        "related_variety",
        "unrelated_variety",
        "average_technological_distance",
        "rao_stirling_diversity",
        "novel_pair_share",
        "pair_rarity_mean",
    ]
    focus, focus_axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    for axis, column in zip(focus_axes.flat, focus_columns):
        axis.plot(x, metrics[column], linewidth=1.0, color="#2F4B66")
        if formal_quarter in metrics["quarter"].astype(str).tolist():
            sample_index = metrics.index[metrics["quarter"].eq(formal_quarter)]
            if len(sample_index):
                axis.axvline(sample_index[0], color="black", linestyle="--", linewidth=0.8)
        axis.set_title(column)
        axis.grid(alpha=0.25)
    for axis in focus_axes[-1, :]:
        axis.set_xticks(tick_positions, metrics["quarter"].iloc[tick_positions], rotation=60, ha="right")
    focus.suptitle("Focus IPC indicators", y=0.995)
    focus.tight_layout()
    focus.savefig(diagnostics_dir / "ipc_innovation_focus_series.png", dpi=160)
    plt.close(focus)


def _write_annual_reconciliation(metrics: pd.DataFrame, output_root: Path) -> dict[str, object]:
    """Compare detail-derived annual counts with the existing annual panel."""

    audit_dir = output_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    existing_path = PROJECT_ROOT / "data" / "processed" / "nva" / "上市公司研发费用与专利年度面板.parquet"
    if metrics.empty:
        result = {
            "status": "no_detail_metrics",
            "existing_panel": str(existing_path),
            "reason": "The detail pipeline produced no baseline applications.",
        }
        (audit_dir / "annual_patent_baseline_reconciliation.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return result

    detail = metrics.copy()
    detail["year"] = detail["quarter"].astype(str).str[:4].astype(int)
    detail_year = detail.groupby("year", as_index=False)[
        [
            "patent_count",
            "invention_patent_count",
            "utility_model_patent_count",
            "design_patent_count",
            "other_patent_count",
        ]
    ].sum()
    detail_year = detail_year.rename(
        columns={column: f"detail_{column}" for column in detail_year.columns if column != "year"}
    )
    if not existing_path.exists():
        result = {
            "status": "comparison_source_missing",
            "existing_panel": str(existing_path),
            "detail_year_count": len(detail_year),
            "reason": "No existing annual patent panel was found at the project canonical path.",
        }
        _write_dataframe(detail_year, audit_dir / "annual_patent_baseline_reconciliation.parquet")
        detail_year.to_csv(
            audit_dir / "annual_patent_baseline_reconciliation.csv",
            index=False,
            encoding="utf-8-sig",
        )
        (audit_dir / "annual_patent_baseline_reconciliation.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return result

    existing = pd.read_parquet(existing_path)
    old_columns = [
        "patents_independent_annual",
        "patents_invention_independent_annual",
        "patents_utility_model_independent_annual",
        "patents_design_independent_annual",
    ]
    missing_old_columns = [column for column in old_columns if column not in existing.columns]
    if "year" not in existing.columns or missing_old_columns:
        result = {
            "status": "comparison_source_incompatible",
            "existing_panel": str(existing_path),
            "missing_columns": missing_old_columns + ([] if "year" in existing.columns else ["year"]),
            "reason": "The existing panel does not expose the expected annual patent fields.",
        }
        _write_dataframe(detail_year, audit_dir / "annual_patent_baseline_reconciliation.parquet")
        (audit_dir / "annual_patent_baseline_reconciliation.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return result

    old_year = existing.groupby("year", as_index=False)[old_columns].sum()
    old_year = old_year.rename(
        columns={
            "patents_independent_annual": "existing_panel_patent_count",
            "patents_invention_independent_annual": "existing_panel_invention_count",
            "patents_utility_model_independent_annual": "existing_panel_utility_model_count",
            "patents_design_independent_annual": "existing_panel_design_count",
        }
    )
    reconciliation = detail_year.merge(old_year, on="year", how="outer").sort_values("year")
    for column in reconciliation.columns:
        if column != "year":
            reconciliation[column] = pd.to_numeric(reconciliation[column], errors="coerce")
    reconciliation["detail_minus_existing_patent_count"] = (
        reconciliation["detail_patent_count"] - reconciliation["existing_panel_patent_count"]
    )
    reconciliation["comparison_status"] = "directional_not_like_for_like"
    reconciliation["difference_reason"] = (
        "Detail data counts unique application_no once within listed-company+subsidiary scope; "
        "the existing annual panel is firm-year aggregate data with its own grant/timing and coverage rules."
    )
    _write_dataframe(reconciliation, audit_dir / "annual_patent_baseline_reconciliation.parquet")
    reconciliation.to_csv(
        audit_dir / "annual_patent_baseline_reconciliation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result = {
        "status": "completed_directional_comparison",
        "existing_panel": str(existing_path),
        "detail_year_count": len(detail_year),
        "existing_year_count": len(old_year),
        "difference_reason": reconciliation["difference_reason"].iloc[0],
    }
    (audit_dir / "annual_patent_baseline_reconciliation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def run_pipeline(
    patent_csv: Path,
    dictionary_path: Path,
    output_root: Path,
    shard_count: int = DEFAULT_SHARD_COUNT,
    alpha: float = 1.0,
    skip_plots: bool = False,
    buffer_records: int = DEFAULT_SHARD_BUFFER_RECORDS,
) -> dict[str, object]:
    """Run all stages and write the independent patent-detail products."""

    output_root.mkdir(parents=True, exist_ok=True)
    parse_summary = _write_sharded_candidates(
        patent_csv, output_root, shard_count, buffer_records=buffer_records
    )
    core_summary = _process_shards(output_root)
    metrics_path = output_root / "quarterly_ipc_innovation_metrics.parquet"
    existing_manifest_path = output_root / "run_manifest.json"
    existing_manifest = (
        json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest_path.exists()
        else {}
    )
    same_dictionary = str(Path(existing_manifest.get("ipc_dictionary", "")).resolve()) == str(
        dictionary_path.resolve()
    )
    required_schema = {
        "ipc_coding_rate",
        "average_patent_scope",
        "median_patent_scope",
        "related_variety",
        "unrelated_variety",
        "average_technological_distance",
        "median_technological_distance",
        "rao_stirling_diversity",
        "design_patent_share",
    }
    metrics_has_history_start = False
    metrics_has_new_schema = False
    if metrics_path.exists():
        existing_columns = set(pd.read_parquet(metrics_path, engine="pyarrow").columns)
        existing_metrics = pd.read_parquet(metrics_path, columns=["quarter"])
        metrics_has_history_start = (
            not existing_metrics.empty
            and existing_metrics["quarter"].min() == _quarter_from_date(HISTORY_START_DATE)
        )
        metrics_has_new_schema = required_schema <= existing_columns
    if (
        (output_root / "06_quarterly_metrics_complete.json").exists()
        and metrics_path.exists()
        and same_dictionary
        and metrics_has_history_start
        and existing_manifest.get("pipeline_version") == PIPELINE_VERSION
        and metrics_has_new_schema
    ):
        metrics = pd.read_parquet(metrics_path)
        history = json.loads(
            (output_root / "audit" / "pair_history_audit.json").read_text(encoding="utf-8")
        )
    else:
        current, historical, taxonomy_size = _load_taxonomy(dictionary_path)
        metrics, history = _process_ipc_and_pairs(
            output_root, current, historical, taxonomy_size, alpha=alpha
        )
        _write_dataframe(metrics, metrics_path)
        metrics.to_csv(
            output_root / "quarterly_ipc_innovation_metrics.csv", index=False, encoding="utf-8-sig"
        )
        formal_mask = pd.PeriodIndex(
            [pd.Period(value.replace("Q", "-Q"), freq="Q") for value in metrics["quarter"]]
        ) >= pd.Period(FORMAL_SAMPLE_START, freq="Q")
        formal_metrics = metrics.loc[formal_mask].reset_index(drop=True)
        _write_dataframe(
            formal_metrics,
            output_root / "quarterly_ipc_innovation_metrics_formal.parquet",
        )
        formal_metrics.to_csv(
            output_root / "quarterly_ipc_innovation_metrics_formal.csv",
            index=False,
            encoding="utf-8-sig",
        )
    formal_metrics_path = output_root / "quarterly_ipc_innovation_metrics_formal.parquet"
    if not formal_metrics_path.exists() and not metrics.empty:
        formal_mask = pd.PeriodIndex(
            [pd.Period(value.replace("Q", "-Q"), freq="Q") for value in metrics["quarter"]]
        ) >= pd.Period(FORMAL_SAMPLE_START, freq="Q")
        formal_metrics = metrics.loc[formal_mask].reset_index(drop=True)
        _write_dataframe(formal_metrics, formal_metrics_path)
        formal_metrics.to_csv(
            output_root / "quarterly_ipc_innovation_metrics_formal.csv",
            index=False,
            encoding="utf-8-sig",
        )
    (output_root / "06_quarterly_metrics_complete.json").write_text(
        json.dumps({"quarter_count": len(metrics)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_diagnostics(metrics, output_root, skip_plots=skip_plots)
    reconciliation = _write_annual_reconciliation(metrics, output_root)

    total_patents = int(metrics["patent_count"].sum()) if not metrics.empty else 0
    coded_patents = int(metrics["ipc_coded_patent_count"].sum()) if not metrics.empty else 0
    coverage = coded_patents / total_patents if total_patents else float("nan")
    formal_quarter = _quarter_from_date(FORMAL_SAMPLE_START)
    if metrics.empty:
        formal_metrics = metrics
    else:
        formal_mask = pd.PeriodIndex(
            [pd.Period(value.replace("Q", "-Q"), freq="Q") for value in metrics["quarter"]]
        ) >= pd.Period(FORMAL_SAMPLE_START, freq="Q")
        formal_metrics = metrics.loc[formal_mask].reset_index(drop=True)
    formal_quarters_complete = False
    if not formal_metrics.empty:
        formal_periods = pd.PeriodIndex(
            [pd.Period(value.replace("Q", "-Q"), freq="Q") for value in formal_metrics["quarter"]]
        )
        formal_quarters_complete = formal_periods.equals(
            pd.period_range(pd.Period(FORMAL_SAMPLE_START, freq="Q"), formal_periods.max(), freq="Q")
        )
    manifest = {
        "input_file": str(patent_csv),
        "input_bytes": patent_csv.stat().st_size,
        "input_mtime_ns": patent_csv.stat().st_mtime_ns,
        "pipeline_version": PIPELINE_VERSION,
        "ipc_dictionary": str(dictionary_path),
        "ipc_dictionary_version": IPC_DICTIONARY_VERSION,
        "firm_scope": FIRM_SCOPE_VERSION,
        "shard_count": shard_count,
        "buffer_records": buffer_records,
        "alpha": alpha,
        "history_start_date": HISTORY_START_DATE,
        "formal_sample_start": FORMAL_SAMPLE_START,
        "counting_rule_version": COUNTING_RULE_VERSION,
        "baseline_patent_types": sorted(BASELINE_PATENT_TYPES),
        "excluded_patent_types_from_ipc": ["design"],
        "ipc_fine_level": IPC_FINE_LEVEL,
        "ipc_coarse_level": IPC_COARSE_LEVEL,
        "relatedness_formula": RELATEDNESS_FORMULA,
        "distance_formula": DISTANCE_FORMULA,
        "same_application_global_deduplication": True,
        "same_day_batch_history_freeze": True,
        "relatedness_matrix": "diagnostics/ipc_relatedness_distance_matrix.parquet",
        "period_summary": "diagnostics/period_summary.csv",
        "parse_summary": parse_summary,
        "core_summary": core_summary,
        "pair_history": history,
        "ipc_coded_patent_rate": coverage,
        "formal_quarter_start": formal_quarter,
        "formal_quarter_count": len(formal_metrics),
        "formal_quarters_complete": formal_quarters_complete,
        "annual_reconciliation": reconciliation,
        "output_files": [str(path.relative_to(output_root)) for path in output_root.rglob("*") if path.is_file()],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    report = output_root / "ipc_innovation_metrics_implementation_report.md"
    novel_share = metrics["novel_pair_share"].dropna() if not metrics.empty else pd.Series(dtype=float)
    rarity = metrics["pair_rarity_mean"].dropna() if not metrics.empty else pd.Series(dtype=float)
    entropy = metrics["ipc_entropy"].dropna() if not metrics.empty else pd.Series(dtype=float)
    hhi = metrics["ipc_hhi"].dropna() if not metrics.empty else pd.Series(dtype=float)
    design_share = metrics["design_patent_share"].dropna() if not metrics.empty else pd.Series(dtype=float)
    design_total = int(metrics["design_patent_count"].sum()) if not metrics.empty else 0
    eligible_all_total = int(metrics["eligible_all_patent_count"].sum()) if not metrics.empty else 0
    design_share_overall = design_total / eligible_all_total if eligible_all_total else float("nan")
    scope = metrics["average_patent_scope"].dropna() if not metrics.empty else pd.Series(dtype=float)
    related = metrics["related_variety"].dropna() if not metrics.empty else pd.Series(dtype=float)
    unrelated = metrics["unrelated_variety"].dropna() if not metrics.empty else pd.Series(dtype=float)
    distance = metrics["average_technological_distance"].dropna() if not metrics.empty else pd.Series(dtype=float)
    rao = metrics["rao_stirling_diversity"].dropna() if not metrics.empty else pd.Series(dtype=float)
    report.write_text(
        "# IPC quarterly innovation metrics implementation report\n\n"
        "## Scope and counting rules\n\n"
        "- Formal IPC universe: invention + utility model applications only. Design applications remain in `patent_core`, `patent_firm_edge`, and the type audit, but never enter IPC metrics.\n"
        f"- Group scope: {FIRM_SCOPE_VERSION}; listed-company and controlled-subsidiary edges are one listed-company group. Joint ventures and associates are excluded from baseline.\n"
        "- `application_no` is deduplicated once globally for macro statistics; original applicant, relation type, listed parent, and company edges are retained for audit.\n"
        "- Historical initialization: 1985–1993; formal reporting: 1994Q1 onward. Pair history and relatedness are frozen for patents filed on the same date.\n\n"
        "## Requested definitions\n\n"
        f"- Patent Scope: unique 4-digit IPC `{IPC_FINE_LEVEL}` subclass count per coded patent; quarterly mean and median.\n"
        f"- Related Variety: within-{IPC_COARSE_LEVEL} entropy, `sum_g p_g H(subclass|g)`; Unrelated Variety: between-class entropy, `H(class)`. Their sum equals subclass entropy.\n"
        f"- Relatedness: `{RELATEDNESS_FORMULA}` from strictly prior co-classification counts; Distance: `{DISTANCE_FORMULA}`. Patent distance is the mean of its IPC-pair distances.\n"
        "- Rao–Stirling: `sum_i sum_j p_i p_j d_ij`, using a distance snapshot fixed at the start of each quarter.\n\n"
        "## Scan and universe\n\n"
        f"- Logical records scanned: {parse_summary['logical_records_seen']:,}\n"
        f"- Successful logical records: {parse_summary['successful_records']:,}\n"
        f"- Parse failure records: {parse_summary.get('failure_record_count', 0):,}\n"
        f"- Unique applications: {core_summary['unique_application_count']:,}\n"
        f"- Baseline universe applications: {core_summary['baseline_universe_count']:,}\n"
        f"- Invention + utility applications: {core_summary['baseline_universe_count']:,}\n"
        f"- IPC-coded patent rate: {coverage:.6%}\n"
        f"- Design applications in eligible group scope: {design_total:,}\n"
        f"- Design share among all eligible typed applications: {design_share_overall:.6%} (pooled); quarterly mean: {design_share.mean():.6%}\n"
        f"- Single/multi IPC distribution: {int(metrics['single_ipc_patent_count'].sum()) if not metrics.empty else 0:,} / {int(metrics['multi_ipc_patent_count'].sum()) if not metrics.empty else 0:,}\n"
        f"- Pair-history pre-sample records (1985–1993): {history['pre_sample_record_count']:,}\n"
        f"- Pre-sample cumulative seen pairs: {history.get('pre_sample_seen_pair_count', 0):,}\n"
        "- Pre-sample coverage note: observed in all nine years, but sparse; treat the 1994 history registry as limited rather than a deep technology-history panel.\n"
        f"- Quarters generated: {len(metrics):,}\n"
        f"- Formal quarters from {formal_quarter}: {len(formal_metrics):,}; contiguous: {formal_quarters_complete}\n"
        f"- Final cumulative pair occurrences: {history['total_pair_occurrences_at_end']:,}\n\n"
        "## Time-series diagnostics\n\n"
        f"- Historical-first pair share range: {novel_share.min():.6g} to {novel_share.max():.6g}\n"
        f"- Pair rarity mean range: {rarity.min():.6g} to {rarity.max():.6g}\n"
        f"- IPC entropy range: {entropy.min():.6g} to {entropy.max():.6g}\n"
        f"- IPC HHI range: {hhi.min():.6g} to {hhi.max():.6g}\n"
        f"- Average patent scope range: {scope.min():.6g} to {scope.max():.6g}\n"
        f"- Related variety range: {related.min():.6g} to {related.max():.6g}\n"
        f"- Unrelated variety range: {unrelated.min():.6g} to {unrelated.max():.6g}\n"
        f"- Average technological distance range: {distance.min():.6g} to {distance.max():.6g}\n"
        f"- Rao–Stirling range: {rao.min():.6g} to {rao.max():.6g}\n\n"
        "## Outputs\n\n"
        "- `quarterly_ipc_innovation_metrics.csv/parquet`: complete 1985Q1–latest quarterly sequence.\n"
        "- `quarterly_ipc_innovation_metrics_formal.csv/parquet`: 1994Q1–latest formal sequence.\n"
        "- `diagnostics/period_summary.csv/parquet`: 1994–1999, 2000–2009, 2010–2019, and 2020–2026 summaries.\n"
        "- `diagnostics/ipc_relatedness_distance_matrix.csv/parquet`: final long-form symmetric-upper-triangle matrix; historical snapshots used by each quarter are not replaced by this final matrix.\n"
        "- `diagnostics/ipc_innovation_quarterly_series.png` and `ipc_innovation_focus_series.png`: time-series diagnostics.\n\n"
        "## Reconciliation and limitations\n\n"
        f"- Annual baseline reconciliation: {reconciliation['status']}\n"
        "- The existing annual panel remains a directional comparison only; it is not overwritten.\n"
        "- `application_no` is an application-level proxy key, not a patent-family identifier.\n"
        "- The raw CSV was not modified. This layer does not modify HWW/NVA/Spec 6.\n",
        encoding="utf-8",
    )
    return {"metrics": metrics_path, "manifest": output_root / "run_manifest.json", "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config, PROJECT_ROOT)
    run_pipeline(
        Path(config["patent_csv"]),
        Path(config["dictionary"]),
        Path(config["output_root"]),
        shard_count=int(config["shard_count"]),
        alpha=float(config["alpha"]),
        skip_plots=bool(config["skip_plots"]),
        buffer_records=int(config["shard_buffer_records"]),
    )


if __name__ == "__main__":
    main()
