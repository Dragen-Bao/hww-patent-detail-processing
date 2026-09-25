"""Build the minimal canonical patent tables used by downstream research code."""

from __future__ import annotations

import csv
import hashlib
import io
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterator

import pandas as pd

EXPECTED_COLUMNS = 31
ROW_START_RE = re.compile(r'^\s*"?=?"{1,4}\d{6}"{1,4}\s*,')
BASELINE_RELATIONS = {"上市公司本身", "上市公司的子公司"}

FIELD_INDEX = {
    "stkcd": 0,
    "company_name": 2,
    "relation": 4,
    "patent_title": 8,
    "patent_type": 9,
    "applicant": 10,
    "applicant_address": 12,
    "application_no": 16,
    "application_date": 17,
    "grant_no": 22,
    "grant_date": 23,
    "ipc_all": 25,
    "ipc_main": 26,
    "abstract": 28,
    "main_claim": 29,
}


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = "".join(
        c if not unicodedata.category(c).startswith("C") else " " for c in text
    )
    return re.sub(r"\s+", " ", text).strip()


def normalize_stock_code(value: object) -> str | None:
    text = clean_text(value).replace("=", "").replace('"', "").replace("'", "")
    match = re.search(r"\d{1,6}", text)
    return match.group(0).zfill(6) if match else None


def normalize_date(value: object) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    value = pd.to_datetime(text, errors="coerce")
    return None if pd.isna(value) else value.strftime("%Y-%m-%d")


def application_quarter(value: object) -> str | None:
    date = normalize_date(value)
    if not date:
        return None
    return str(pd.Period(pd.Timestamp(date), freq="Q"))


def normalize_patent_type(value: object) -> str:
    text = clean_text(value)
    if text.startswith("发明"):
        return "invention"
    if text == "实用新型":
        return "utility_model"
    if text == "外观设计":
        return "design"
    return "other"


def normalize_ipc(value: object) -> str | None:
    text = re.sub(r"\s+", "", clean_text(value).upper())
    if not text:
        return None
    text = re.sub(r"([A-H]\d{2}[A-Z]\d{1,4}/\d{1,6})[IN]$", r"\1", text)
    match = re.match(r"([A-H]\d{2}[A-Z](?:\d{1,4}/\d{1,6})?)", text)
    return match.group(1) if match else None


def ipc_main_group(value: object) -> str | None:
    code = normalize_ipc(value)
    if not code or "/" not in code:
        return None
    return code.split("/", 1)[0]


def applicant_key(applicant: object, address: object) -> str:
    name = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", clean_text(applicant).lower())
    addr = re.sub(r"^\d{6}\s*", "", clean_text(address).lower())
    addr = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", addr)
    return f"{name}||{addr}" if name else ""


def _get(row: list[str], field: str) -> str:
    index = FIELD_INDEX[field]
    return row[index].strip() if len(row) > index else ""


def _parse(lines: list[str]) -> list[str]:
    try:
        return next(csv.reader(io.StringIO("".join(lines)), strict=False))
    except csv.Error:
        return "".join(lines).split(",")


def iter_records(path: Path) -> Iterator[list[str]]:
    """Read the malformed export while preserving wrapped abstract/claim fields."""
    buffer: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if ROW_START_RE.match(line):
                if buffer:
                    row = _parse(buffer)
                    yield row + [""] * max(0, EXPECTED_COLUMNS - len(row))
                buffer = [line]
            elif buffer:
                buffer.append(line)
        if buffer:
            row = _parse(buffer)
            yield row + [""] * max(0, EXPECTED_COLUMNS - len(row))


def row_to_record(row: list[str]) -> dict[str, object]:
    app_date = normalize_date(_get(row, "application_date"))
    grant_date = normalize_date(_get(row, "grant_date"))
    patent_type = normalize_patent_type(_get(row, "patent_type"))
    relation = clean_text(_get(row, "relation"))
    applicant = clean_text(_get(row, "applicant"))
    address = clean_text(_get(row, "applicant_address"))
    ipc_main = normalize_ipc(_get(row, "ipc_main"))
    return {
        "listed_parent": normalize_stock_code(_get(row, "stkcd")),
        "company_name": clean_text(_get(row, "company_name")),
        "relation": relation,
        "application_no": clean_text(_get(row, "application_no")),
        "application_date": app_date,
        "patent_type": patent_type,
        "patent_title": clean_text(_get(row, "patent_title")),
        "abstract": clean_text(_get(row, "abstract")),
        "main_claim": clean_text(_get(row, "main_claim")),
        "applicant": applicant,
        "applicant_address": address,
        "applicant_key": applicant_key(applicant, address),
        "ipc_all": clean_text(_get(row, "ipc_all")),
        "ipc_main": ipc_main,
        "ipc_main_group": ipc_main_group(ipc_main),
        "grant_no": clean_text(_get(row, "grant_no")),
        "grant_date": grant_date,
        "baseline_firm": int(relation in BASELINE_RELATIONS),
        "granted_invention": int(
            patent_type == "invention"
            and bool(clean_text(_get(row, "grant_no")) or grant_date)
        ),
    }


def _pick(values: pd.Series) -> str:
    values = [clean_text(v) for v in values if clean_text(v)]
    if not values:
        return ""
    counts = Counter(values)
    return max(counts, key=lambda x: (counts[x], len(x)))


def _longest(values: pd.Series) -> str:
    return max((clean_text(v) for v in values), key=len, default="")


def canonicalize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    patents: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []

    for application_no, group in frame.groupby("application_no", sort=False):
        dates = sorted({d for d in group["application_date"] if d})
        app_date = dates[0] if len(dates) == 1 else None
        patent_type = _pick(group["patent_type"])
        grant_no = _pick(group["grant_no"])
        grant_date = _pick(group["grant_date"])
        ipc_main = _pick(group["ipc_main"])
        applicant = _pick(group["applicant"])
        address = _pick(group["applicant_address"])

        patents.append({
            "application_no": application_no,
            "application_date": app_date,
            "application_quarter": application_quarter(app_date),
            "patent_type": patent_type,
            "patent_title": _longest(group["patent_title"]),
            "abstract": _longest(group["abstract"]),
            "main_claim": _longest(group["main_claim"]),
            "applicant": applicant,
            "applicant_address": address,
            "applicant_key": applicant_key(applicant, address),
            "ipc_all": _pick(group["ipc_all"]),
            "ipc_main": ipc_main or None,
            "ipc_main_group": ipc_main_group(ipc_main),
            "grant_no": grant_no,
            "grant_date": grant_date or None,
            "granted_invention": int(
                patent_type == "invention" and bool(grant_no or grant_date)
            ),
            "baseline_firm": int(group["baseline_firm"].max()),
        })

        for listed_parent, edge in group[group["listed_parent"].notna()].groupby("listed_parent"):
            edges.append({
                "listed_parent": listed_parent,
                "application_no": application_no,
                "baseline_firm": int(edge["baseline_firm"].max()),
            })

    return pd.DataFrame(patents), pd.DataFrame(edges)


def _shard(application_no: str, n: int) -> int:
    digest = hashlib.blake2b(application_no.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % n


def _write_by_quarter(frame: pd.DataFrame, root: Path, prefix: str) -> None:
    if frame.empty:
        return
    for quarter, part in frame.groupby("application_quarter", dropna=False):
        label = str(quarter) if pd.notna(quarter) and quarter else "unknown"
        directory = root / f"application_quarter={label}"
        directory.mkdir(parents=True, exist_ok=True)
        part.to_parquet(
            directory / f"{prefix}.parquet", index=False, compression="zstd"
        )


def build_canonical_dataset(
    raw_csv: Path,
    output_root: Path,
    *,
    shard_count: int = 128,
    buffer_size: int = 2000,
) -> None:
    """Create quarter-partitioned canonical_patents/ and firm_edges/."""
    if output_root.exists():
        shutil.rmtree(output_root)
    raw_root = output_root / "_raw_shards"
    raw_root.mkdir(parents=True)

    buffers = {i: [] for i in range(shard_count)}
    parts = Counter()

    def flush(shard: int) -> None:
        if not buffers[shard]:
            return
        pd.DataFrame(buffers[shard]).to_parquet(
            raw_root / f"{shard:04d}-{parts[shard]:05d}.parquet",
            index=False,
            compression="zstd",
        )
        parts[shard] += 1
        buffers[shard].clear()

    for row in iter_records(raw_csv):
        if len(row) < EXPECTED_COLUMNS:
            continue
        record = row_to_record(row)
        application_no = str(record["application_no"])
        if not application_no:
            continue
        shard = _shard(application_no, shard_count)
        buffers[shard].append(record)
        if len(buffers[shard]) >= buffer_size:
            flush(shard)

    for shard in range(shard_count):
        flush(shard)

    patent_root = output_root / "canonical_patents"
    edge_root = output_root / "firm_edges"

    for shard in range(shard_count):
        paths = sorted(raw_root.glob(f"{shard:04d}-*.parquet"))
        if not paths:
            continue
        raw = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        patents, edges = canonicalize(raw)
        _write_by_quarter(patents, patent_root, f"{shard:04d}")
        if not edges.empty:
            edges = edges.merge(
                patents[["application_no", "application_date", "application_quarter"]],
                on="application_no",
                how="left",
            )
            _write_by_quarter(edges, edge_root, f"{shard:04d}")

    shutil.rmtree(raw_root)
