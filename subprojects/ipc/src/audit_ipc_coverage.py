"""Audit IPC coding coverage over time and across patent/entity groups.

The audit uses the patent-detail pipeline's canonical application grain and
baseline scope.  Industry is read from the raw CSV because it was not retained
in the first intermediate layer.  The raw CSV is never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.patent_detail.pipeline import _iter_fast_logical_records  # noqa: E402


ELIGIBLE_RELATIONS = {"上市公司本身", "上市公司的子公司"}
RAW_APPLICATION_INDEX = 16
RAW_INDUSTRY_INDEX = 6
RAW_RELATION_INDEX = 4


def _clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _stable_shard(application_no: str, shard_count: int) -> int:
    digest = hashlib.blake2b(application_no.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % shard_count


def _collapse(values: pd.Series) -> str:
    unique = sorted({_clean(value) for value in values if _clean(value)})
    if not unique:
        return "__MISSING__"
    if len(unique) > 1:
        return "__MULTI__"
    return unique[0]


def _has_mapped_subclass(values: object, valid_subclasses: set[str]) -> bool:
    if values is None:
        return False
    return any(str(value) in valid_subclasses for value in values)  # type: ignore[operator]


def _file_map(root: Path, shard_count: int) -> dict[int, list[Path]]:
    mapping: dict[int, list[Path]] = defaultdict(list)
    for path in root.rglob("shard-*.parquet"):
        try:
            shard = int(path.stem.split("-")[-1])
        except ValueError:
            continue
        if 0 <= shard < shard_count:
            mapping[shard].append(path)
    return mapping


def _write_industry_spill(
    patent_csv: Path,
    spill_dir: Path,
    shard_count: int,
    progress_every: int = 100_000,
) -> dict[str, int]:
    """Write only eligible raw application-industry pairs to temporary shards."""

    spill_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        shard: (spill_dir / f"shard-{shard:04d}.tsv").open("w", encoding="utf-8", newline="")
        for shard in range(shard_count)
    }
    logical_records = 0
    eligible_rows = 0
    nonempty_industry_rows = 0
    invalid_rows = 0
    try:
        writers = {shard: csv.writer(handle, delimiter="\t", lineterminator="\n") for shard, handle in handles.items()}
        for logical_records, row in enumerate(_iter_fast_logical_records(patent_csv), start=1):
            if len(row) <= max(RAW_APPLICATION_INDEX, RAW_INDUSTRY_INDEX, RAW_RELATION_INDEX):
                invalid_rows += 1
                continue
            application_no = _clean(row[RAW_APPLICATION_INDEX])
            relation = _clean(row[RAW_RELATION_INDEX])
            if not application_no or relation not in ELIGIBLE_RELATIONS:
                continue
            eligible_rows += 1
            industry = _clean(row[RAW_INDUSTRY_INDEX])
            if industry:
                nonempty_industry_rows += 1
            writers[_stable_shard(application_no, shard_count)].writerow([application_no, industry])
            if logical_records % progress_every == 0:
                print(f"raw logical_records={logical_records:,}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    return {
        "logical_records": logical_records,
        "eligible_raw_rows": eligible_rows,
        "nonempty_industry_rows": nonempty_industry_rows,
        "invalid_rows": invalid_rows,
    }


def _load_industry_map(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["application_no", "industry_group"])
    frame = pd.read_csv(
        path,
        sep="\t",
        names=["application_no", "industry"],
        dtype="string",
        keep_default_na=False,
    )
    frame = frame[frame["industry"].str.strip().ne("")]
    if frame.empty:
        return pd.DataFrame(columns=["application_no", "industry_group"])
    return (
        frame.groupby("application_no", as_index=False)["industry"]
        .agg(_collapse)
        .rename(columns={"industry": "industry_group"})
    )


def _load_shard_apps(
    shard: int,
    core_files: dict[int, list[Path]],
    edge_files: dict[int, list[Path]],
    raw_shard: Path,
    industry_spill: Path,
    valid_subclasses: set[str],
) -> pd.DataFrame:
    core_parts = [
        pd.read_parquet(path, columns=["application_no", "application_date", "patent_type_group", "ipc_subclasses"])
        for path in sorted(core_files.get(shard, []))
    ]
    edge_parts = [
        pd.read_parquet(path, columns=["application_no", "relation", "applicant_type", "baseline_eligible"])
        for path in sorted(edge_files.get(shard, []))
    ]
    if not core_parts or not edge_parts:
        return pd.DataFrame()

    core = pd.concat(core_parts, ignore_index=True)
    edge = pd.concat(edge_parts, ignore_index=True)
    edge = edge[edge["baseline_eligible"].eq(1)].copy()
    if edge.empty:
        return pd.DataFrame()

    edge_group = edge.groupby("application_no", as_index=False).agg(
        relation_group=("relation", _collapse),
        applicant_type_group=("applicant_type", _collapse),
    )
    apps = core.merge(edge_group, on="application_no", how="inner", validate="one_to_one")
    if apps.empty:
        return apps

    industries = _load_industry_map(industry_spill)
    apps = apps.merge(industries, on="application_no", how="left", validate="one_to_one")
    apps["industry_group"] = apps["industry_group"].fillna("__MISSING__")
    apps["has_ipc"] = apps["ipc_subclasses"].map(
        lambda values: _has_mapped_subclass(values, valid_subclasses)
    )
    dates = pd.to_datetime(apps["application_date"], errors="coerce")
    apps["year"] = dates.dt.year.astype("Int64")
    apps["quarter"] = dates.dt.to_period("Q").astype("string").str.replace("-", "Q", regex=False)
    apps = apps[apps["year"].notna() & apps["quarter"].notna()].copy()
    apps["patent_type_group"] = apps["patent_type_group"].fillna("__MISSING__")
    return apps[[
        "application_no",
        "year",
        "quarter",
        "has_ipc",
        "patent_type_group",
        "industry_group",
        "relation_group",
        "applicant_type_group",
    ]]


def _summarize_apps(apps: pd.DataFrame, dimensions: dict[str, str]) -> list[pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    for dimension, column in dimensions.items():
        grouped = (
            apps.groupby(["year", "quarter", column, "has_ipc"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )
        if grouped.empty:
            continue
        pivot = grouped.pivot_table(
            index=["year", "quarter", column],
            columns="has_ipc",
            values="count",
            fill_value=0,
            aggfunc="sum",
        ).reset_index()
        pivot = pivot.rename(columns={False: "ipc_missing_patent_count", True: "ipc_coded_patent_count", column: "category"})
        for name in ["ipc_missing_patent_count", "ipc_coded_patent_count"]:
            if name not in pivot:
                pivot[name] = 0
        pivot["dimension"] = dimension
        pivot["patent_count"] = pivot["ipc_missing_patent_count"] + pivot["ipc_coded_patent_count"]
        outputs.append(pivot[[
            "year",
            "quarter",
            "dimension",
            "category",
            "patent_count",
            "ipc_coded_patent_count",
            "ipc_missing_patent_count",
        ]])
    return outputs


def audit_coverage(
    patent_csv: Path,
    dictionary_path: Path,
    output_dir: Path,
    patent_detail_root: Path,
    shard_count: int = 128,
    progress_every: int = 100_000,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    core_files = _file_map(patent_detail_root / "patent_core", shard_count)
    edge_files = _file_map(patent_detail_root / "patent_firm_edge", shard_count)
    raw_shards = {int(path.stem.split("-")[-1]): path for path in (patent_detail_root / "raw_shards").glob("shard-*.parquet")}
    if not core_files or not edge_files or len(raw_shards) < shard_count:
        raise FileNotFoundError("Existing patent-detail shard outputs are incomplete; run the canonical pipeline first.")
    dictionary = pd.read_parquet(dictionary_path)
    valid_subclasses = set(
        dictionary.loc[
            dictionary["is_valid"].eq(True) & dictionary["ipc_level"].eq("subclass"),
            "ipc_code",
        ].astype(str)
    )

    dimensions = {
        "patent_type": "patent_type_group",
        "industry": "industry_group",
        "enterprise_relation": "relation_group",
        "applicant_type": "applicant_type_group",
    }
    summary_parts: list[pd.DataFrame] = []
    total_apps = 0
    total_coded = 0
    total_missing = 0
    spill_dir = output_dir / "_ipc_coverage_spill"
    spill_manifest = spill_dir / "scan_manifest.json"
    scan_summary = None
    if spill_manifest.exists():
        candidate = json.loads(spill_manifest.read_text(encoding="utf-8"))
        expected = {"input_bytes": patent_csv.stat().st_size, "shard_count": shard_count}
        if all(candidate.get(key) == value for key, value in expected.items()) and all(
            (spill_dir / f"shard-{shard:04d}.tsv").exists() for shard in range(shard_count)
        ):
            scan_summary = candidate["scan_summary"]
            print("reusing completed industry spill", flush=True)
    if scan_summary is None:
        if spill_dir.exists():
            shutil.rmtree(spill_dir)
        scan_summary = _write_industry_spill(patent_csv, spill_dir, shard_count, progress_every)
        spill_manifest.write_text(
            json.dumps(
                {
                    "input_bytes": patent_csv.stat().st_size,
                    "input_mtime_ns": patent_csv.stat().st_mtime_ns,
                    "shard_count": shard_count,
                    "scan_summary": scan_summary,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    for shard in range(shard_count):
        apps = _load_shard_apps(
            shard,
            core_files,
            edge_files,
            raw_shards[shard],
            spill_dir / f"shard-{shard:04d}.tsv",
            valid_subclasses,
        )
        if apps.empty:
            continue
        total_apps += len(apps)
        total_coded += int(apps["has_ipc"].sum())
        total_missing += int((~apps["has_ipc"]).sum())
        overall = apps.assign(
            dimension="overall",
            category="ALL",
        )[["year", "quarter", "has_ipc", "dimension", "category"]]
        summary_parts.extend(_summarize_apps(overall, {"overall": "category"}))
        summary_parts.extend(_summarize_apps(apps, dimensions))
        if shard % 16 == 0:
            print(f"processed shard={shard:03d}/{shard_count - 1:03d}", flush=True)

    summary = pd.concat(summary_parts, ignore_index=True)
    summary = (
        summary.groupby(["year", "quarter", "dimension", "category"], as_index=False)[
            ["patent_count", "ipc_coded_patent_count", "ipc_missing_patent_count"]
        ]
        .sum()
    )
    summary["ipc_coded_rate"] = summary["ipc_coded_patent_count"] / summary["patent_count"]
    summary["ipc_missing_rate"] = summary["ipc_missing_patent_count"] / summary["patent_count"]
    summary = summary.sort_values(["year", "quarter", "dimension", "category"]).reset_index(drop=True)
    annual = (
        summary.groupby(["year", "dimension", "category"], as_index=False)[
            ["patent_count", "ipc_coded_patent_count", "ipc_missing_patent_count"]
        ]
        .sum()
    )
    annual["ipc_coded_rate"] = annual["ipc_coded_patent_count"] / annual["patent_count"]
    annual["ipc_missing_rate"] = annual["ipc_missing_patent_count"] / annual["patent_count"]

    quarterly_path = output_dir / "ipc_coding_coverage_by_dimension_quarterly.parquet"
    annual_path = output_dir / "ipc_coding_coverage_by_dimension_annual.parquet"
    summary.to_parquet(quarterly_path, index=False)
    annual.to_parquet(annual_path, index=False)
    summary.to_csv(output_dir / "ipc_coding_coverage_by_dimension_quarterly.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(output_dir / "ipc_coding_coverage_by_dimension_annual.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "unit": "unique application_no",
        "date": "canonical application_date",
        "scope": "listed company itself + subsidiary",
        "dictionary": str(dictionary_path),
        "ipc_coded_definition": "at least one canonical IPC subclass is recognized by the versioned dictionary",
        "industry_source": "raw CSV column 上市公司行业, restricted to eligible relation rows",
        "enterprise_type_proxies": {
            "enterprise_relation": "与上市公司关系",
            "applicant_type": "申请人类型",
        },
        "raw_input_bytes": patent_csv.stat().st_size,
        "shard_count": shard_count,
        "scan_summary": scan_summary,
        "baseline_application_count": total_apps,
        "ipc_coded_application_count": total_coded,
        "ipc_missing_application_count": total_missing,
        "overall_ipc_coded_rate": total_coded / total_apps if total_apps else None,
    }
    summary_json = output_dir / "ipc_coding_coverage_audit_summary.json"
    summary_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    overall = summary[summary["dimension"].eq("overall")].copy()
    earliest = overall.sort_values("quarter").head(1)
    latest = overall.sort_values("quarter").tail(1)
    formal_overall = overall[overall["year"].ge(1994)]
    formal_earliest = formal_overall.sort_values("quarter").head(1)
    report = output_dir / "ipc_coding_coverage_audit_report.md"
    report.write_text(
        "# IPC 编码率趋势与分组缺失审计\n\n"
        f"- 观测单位：{metadata['unit']}；日期：{metadata['date']}。\n"
        f"- 样本范围：{metadata['scope']}。\n"
        f"- 基线申请数：{total_apps:,}；IPC 已编码：{total_coded:,}；IPC 缺失：{total_missing:,}。\n"
        f"- 总体 IPC 编码率：{metadata['overall_ipc_coded_rate']:.2%}。\n"
        f"- 最早有基线申请的季度：{earliest['quarter'].iloc[0]}，编码率 {float(earliest['ipc_coded_rate'].iloc[0]):.2%}；正式样本首季 {formal_earliest['quarter'].iloc[0]}，编码率 {float(formal_earliest['ipc_coded_rate'].iloc[0]):.2%}；最近季度 {latest['quarter'].iloc[0]}，编码率 {float(latest['ipc_coded_rate'].iloc[0]):.2%}。\n"
        "- 1985Q1 没有基线申请，因此不应把该季度的空值解释为 0% 编码率。\n\n"
        "## 分组字段说明\n\n"
        "- `patent_type`：规范化的发明、实用新型、外观设计、其他。\n"
        "- `industry`：原始字段“上市公司行业”；多值记录记为 `__MULTI__`，无值记为 `__MISSING__`。\n"
        "- `enterprise_relation`：原始字段“与上市公司关系”，仅在当前基线范围内统计。\n"
        "- `applicant_type`：原始字段“申请人类型”。它不是国企/民企所有制字段；若要检查国企/民企，需要另接所有制分类。\n\n"
        "## 输出\n\n"
        "季度和年度表均同时给出分子、分母、编码率和缺失率。`__MISSING__`/`__MULTI__` 是字段质量状态，不应直接当成经济类别解释。\n",
        encoding="utf-8",
    )
    shutil.rmtree(spill_dir)
    return {
        "quarterly": quarterly_path,
        "annual": annual_path,
        "summary": summary_json,
        "report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patent-csv", type=Path, required=True)
    parser.add_argument("--patent-detail-root", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=100_000)
    args = parser.parse_args()
    outputs = audit_coverage(
        args.patent_csv,
        args.dictionary,
        args.output_dir,
        args.patent_detail_root,
        shard_count=args.shard_count,
        progress_every=args.progress_every,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
