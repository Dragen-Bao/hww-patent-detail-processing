"""Point-in-time patent-abstract text indicators.

This module is deliberately separate from the IPC production pipeline.  The
raw patent file is read only to recover text fields that were not retained in
the existing IPC shards.  Text novelty is a descriptive, predictive feature;
it is not labelled as patent quality and is not connected to HWW/Spec 6.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from subprojects.ipc.src.audit_patent_ipc import ROW_START_RE

from preprocessing.patent_detail.pipeline import (
    BASELINE_PATENT_TYPES,
    FIELD_INDEX,
    _stable_shard,
    normalize_date,
    normalize_stock_code,
    normalize_patent_type,
)


EXPECTED_COLUMNS = 31
ABSTRACT_INDEX = 28
CLAIM_INDEX = 29
DEFAULT_MIN_NGRAM = 2
DEFAULT_MAX_NGRAM = 4
DEFAULT_HASH_FEATURES = 2**16
TEXT_COLUMNS = [
    "stkcd_raw",
    "stkcd",
    "company_name",
    "relation",
    "relation_type",
    "relation_type_values",
    "listed_parent",
    "listed_parent_values",
    "applicant",
    "applicant_values",
    "applicant_type",
    "application_no",
    "application_date",
    "quarter",
    "patent_type_group",
    "baseline_eligible",
    "abstract_clean",
    "abstract_length_chars",
    "abstract_unique_char_count",
    "abstract_source_record_count",
    "abstract_version_count",
    "abstract_conflict_flag",
    "claim_present_flag",
    "claim_length_chars",
    "application_date_conflict",
]


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


def iter_full_logical_records(path: Path) -> Iterator[list[str]]:
    """Yield complete 31-column logical rows from the raw patent CSV."""

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
                target_index = CLAIM_INDEX if not current_row[ABSTRACT_INDEX].strip() else ABSTRACT_INDEX
                current_row[target_index] = (
                    f"{current_row[target_index]}\n{continuation}".strip()
                )
        if fallback_lines is not None:
            row = _parse_fallback_lines(fallback_lines)
            yield row + [""] * max(0, EXPECTED_COLUMNS - len(row))
        if current_row is not None:
            yield current_row


def clean_abstract(value: object) -> str:
    """Normalize an abstract without deleting Chinese technical content."""

    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = "".join(character if not unicodedata.category(character).startswith("C") else " " for character in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def char_ngram_counts(text: str, min_n: int = DEFAULT_MIN_NGRAM, max_n: int = DEFAULT_MAX_NGRAM) -> Counter[str, int]:
    """Return overlapping character n-gram counts for interpretable tests."""

    counts: Counter[str, int] = Counter()
    if not text:
        return counts
    for n in range(min_n, max_n + 1):
        for start in range(0, max(0, len(text) - n + 1)):
            counts[text[start : start + n]] += 1
    return counts


def _hash_ngram(token: str, n_features: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % n_features


def _quarter(value: str | None) -> str | None:
    if not value:
        return None
    period = pd.Period(value, freq="Q")
    return f"{period.year}Q{period.quarter}"


def _record_from_row(row: list[str]) -> dict[str, object]:
    def get(index: int) -> str:
        return row[index].strip() if len(row) > index else ""

    application_date = normalize_date(get(FIELD_INDEX["application_date"]))
    patent_type = get(FIELD_INDEX["patent_type"])
    relation = get(FIELD_INDEX["relation"])
    eligible = int(
        relation in {"上市公司本身", "上市公司的子公司"}
        and normalize_patent_type(patent_type) in BASELINE_PATENT_TYPES
        and bool(application_date)
    )
    return {
        "stkcd_raw": get(FIELD_INDEX["stkcd_raw"]),
        "stkcd": normalize_stock_code(get(FIELD_INDEX["stkcd_raw"])),
        "company_name": get(FIELD_INDEX["company_name"]),
        "relation": relation,
        "relation_type": relation,
        "listed_parent": normalize_stock_code(get(FIELD_INDEX["stkcd_raw"])),
        "applicant": get(FIELD_INDEX["applicant"]),
        "applicant_type": get(FIELD_INDEX["applicant_type"]),
        "application_no": get(FIELD_INDEX["application_no"]),
        "application_date": application_date,
        "quarter": _quarter(application_date),
        "patent_type_group": normalize_patent_type(patent_type),
        "baseline_eligible": eligible,
        "abstract_clean": clean_abstract(get(ABSTRACT_INDEX)),
        "claim_present_flag": int(bool(get(CLAIM_INDEX))),
        "claim_length_chars": len(get(CLAIM_INDEX)),
    }


def canonicalize_text_records(records: Iterable[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one application record and unique eligible group edges."""

    rows = [dict(record) for record in records if str(record.get("application_no") or "").strip()]
    if not rows:
        return pd.DataFrame(columns=TEXT_COLUMNS), pd.DataFrame(columns=TEXT_COLUMNS)
    frame = pd.DataFrame(rows)
    frame["application_no"] = frame["application_no"].astype(str).str.strip()

    def join_unique(values: Iterable[object]) -> str:
        return " | ".join(sorted({str(value).strip() for value in values if str(value).strip()}))
    grouped = frame.groupby("application_no", sort=False, observed=True)
    frame["_abstract_length"] = frame["abstract_clean"].astype(str).str.len()
    first_indices = frame.groupby("application_no", sort=False, observed=True)["_abstract_length"].idxmax()
    core = frame.loc[first_indices.to_numpy()].copy().set_index("application_no")
    core.index.name = "application_no"
    baseline_representative = (
        frame.loc[frame["baseline_eligible"].eq(1)]
        .drop_duplicates("application_no", keep="first")
        .set_index("application_no")
    )
    for column in [
        "stkcd_raw",
        "stkcd",
        "company_name",
        "relation",
        "relation_type",
        "listed_parent",
        "applicant",
        "applicant_type",
    ]:
        if column in baseline_representative.columns:
            core[column] = baseline_representative[column].reindex(core.index).combine_first(core[column])
    date_values = frame["application_date"].replace("", pd.NA)
    date_count = date_values.groupby(frame["application_no"], sort=False, observed=True).nunique(dropna=True)
    chosen_date = pd.to_datetime(date_values, errors="coerce").groupby(
        frame["application_no"], sort=False, observed=True
    ).min()
    valid_text = frame["abstract_clean"].astype(str).where(frame["abstract_clean"].astype(str).ne(""), pd.NA)
    version_count = valid_text.groupby(frame["application_no"], sort=False, observed=True).nunique(dropna=True)
    source_count = grouped.size()
    relation_values = grouped["relation_type"].agg(join_unique)
    parent_values = grouped["listed_parent"].agg(join_unique)
    applicant_values = grouped["applicant"].agg(join_unique)
    baseline_any = grouped["baseline_eligible"].max().astype(bool)
    claim_present = grouped["claim_present_flag"].max().astype(int)
    claim_length = grouped["claim_length_chars"].max().astype(int)

    core["application_date"] = chosen_date.dt.strftime("%Y-%m-%d")
    core["quarter"] = chosen_date.dt.to_period("Q").astype("string")
    core["abstract_length_chars"] = core["abstract_clean"].astype(str).str.len()
    core["abstract_unique_char_count"] = core["abstract_clean"].astype(str).map(lambda value: len(set(value)) if value else 0)
    core["abstract_source_record_count"] = source_count
    core["abstract_version_count"] = version_count
    core["abstract_conflict_flag"] = version_count.gt(1).astype(int)
    core["relation_type_values"] = relation_values
    core["listed_parent_values"] = parent_values
    core["applicant_values"] = applicant_values
    core["baseline_eligible"] = (baseline_any & date_count.le(1)).astype(int)
    core["claim_present_flag"] = claim_present
    core["claim_length_chars"] = claim_length
    core["application_date_conflict"] = date_count.gt(1).astype(int)
    core = core.reset_index()

    eligible = frame.loc[frame["baseline_eligible"].eq(1)].copy()
    if not eligible.empty:
        conflict_lookup = date_count.gt(1).astype(int).rename("application_date_conflict").reset_index()
        eligible = eligible.merge(conflict_lookup, on="application_no", how="left")
        eligible = eligible.loc[eligible["application_date_conflict"].eq(0)]
        edge_grouped = eligible.groupby(["listed_parent", "application_no"], sort=False, observed=True)
        edge_first = eligible.drop_duplicates(["listed_parent", "application_no"], keep="first").copy()
        edge_first = edge_first.merge(
            edge_grouped["relation_type"].agg(join_unique).rename("relation_type_values").reset_index(),
            on=["listed_parent", "application_no"],
            how="left",
        )
        edge_first = edge_first.merge(
            edge_grouped["applicant"].agg(join_unique).rename("applicant_values").reset_index(),
            on=["listed_parent", "application_no"],
            how="left",
        )
        edge_first["listed_parent_values"] = edge_first["listed_parent"]
        derived_columns = [
            "abstract_length_chars",
            "abstract_unique_char_count",
            "abstract_source_record_count",
            "abstract_version_count",
            "abstract_conflict_flag",
        ]
        edge_first = edge_first.merge(
            core[["application_no", *derived_columns]],
            on="application_no",
            how="left",
        )
        edges = edge_first[TEXT_COLUMNS].copy()
    else:
        edges = pd.DataFrame(columns=TEXT_COLUMNS)
    return core[TEXT_COLUMNS].copy(), edges


def _exact_counter_score_batch(batch: pd.DataFrame, history: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    output = batch.copy()
    for column in (
        "abstract_history_document_count",
        "abstract_unseen_ngram_share",
        "abstract_prior_centroid_similarity",
        "abstract_prior_centroid_novelty",
        "abstract_history_cold_start_flag",
    ):
        output[column] = np.nan
    df: Counter[str, int] = history.setdefault("document_frequency", Counter())
    term_sum: Counter[str, float] = history.setdefault("term_sum", Counter())
    document_count = int(history.get("document_count", 0))
    date_values = output.get("application_date", pd.Series([None] * len(output), index=output.index))
    date_values = date_values.fillna("").astype(str)
    for _, date_batch in output.groupby(date_values, sort=True):
        before = document_count
        centroid_norm = math.sqrt(sum(value * value for value in term_sum.values()))
        for index, row in date_batch.iterrows():
            counts = char_ngram_counts(str(row.get("abstract_clean") or ""))
            total = sum(counts.values())
            output.loc[index, "abstract_history_document_count"] = before
            output.loc[index, "abstract_history_cold_start_flag"] = int(before == 0)
            if total == 0:
                continue
            unseen = sum(count for token, count in counts.items() if df[token] == 0)
            output.loc[index, "abstract_unseen_ngram_share"] = unseen / total
            weights = {
                token: count * (math.log((before + 1) / (df[token] + 1)) + 1.0)
                for token, count in counts.items()
            }
            norm = math.sqrt(sum(value * value for value in weights.values()))
            if before > 0 and norm > 0 and centroid_norm > 0:
                dot = sum(value * term_sum[token] for token, value in weights.items()) / norm
                output.loc[index, "abstract_prior_centroid_similarity"] = dot / centroid_norm
                output.loc[index, "abstract_prior_centroid_novelty"] = 1.0 - float(
                    output.loc[index, "abstract_prior_centroid_similarity"]
                )
        for _, row in date_batch.iterrows():
            counts = char_ngram_counts(str(row.get("abstract_clean") or ""))
            total = sum(counts.values())
            if total == 0:
                continue
            weights = {
                token: count * (math.log((before + 1) / (df[token] + 1)) + 1.0)
                for token, count in counts.items()
            }
            norm = math.sqrt(sum(value * value for value in weights.values()))
            for token in counts:
                df[token] += 1
            if norm > 0:
                for token, value in weights.items():
                    term_sum[token] += value / norm
            document_count += 1
    history["document_count"] = document_count
    return output, history


@dataclass
class HashedHistoryState:
    """Online history state for the scalable text score."""

    document_count: int
    document_frequency: np.ndarray
    term_sum: np.ndarray
    n_features: int = DEFAULT_HASH_FEATURES


def _hashed_vectorizer(
    min_n: int = DEFAULT_MIN_NGRAM,
    max_n: int = DEFAULT_MAX_NGRAM,
    n_features: int = DEFAULT_HASH_FEATURES,
) -> HashingVectorizer:
    return HashingVectorizer(
        analyzer="char",
        ngram_range=(min_n, max_n),
        n_features=n_features,
        alternate_sign=False,
        lowercase=False,
        norm=None,
        dtype=np.float64,
    )


def _score_hashed_batch(
    batch: pd.DataFrame,
    history: HashedHistoryState,
    vectorizer: HashingVectorizer | None = None,
) -> tuple[pd.DataFrame, HashedHistoryState]:
    output = batch.copy()
    for column in (
        "abstract_history_document_count",
        "abstract_unseen_ngram_share",
        "abstract_prior_centroid_similarity",
        "abstract_prior_centroid_novelty",
        "abstract_history_cold_start_flag",
    ):
        output[column] = np.nan
    vectorizer = vectorizer or _hashed_vectorizer()
    date_values = output.get("application_date", pd.Series([None] * len(output), index=output.index))
    date_values = date_values.fillna("").astype(str)
    for _, date_batch in output.groupby(date_values, sort=True):
        valid_mask = date_batch["abstract_clean"].astype(str).str.len().gt(0)
        valid = date_batch.loc[valid_mask]
        before = history.document_count
        output.loc[date_batch.index, "abstract_history_document_count"] = before
        output.loc[date_batch.index, "abstract_history_cold_start_flag"] = int(before == 0)
        if valid.empty:
            continue
        matrix = vectorizer.transform(valid["abstract_clean"].astype(str).tolist()).tocsr()
        matrix.sum_duplicates()
        centroid_norm = float(np.linalg.norm(history.term_sum / before)) if before else 0.0
        centroid = history.term_sum / before if before else history.term_sum
        idf = np.log((before + 1.0) / (history.document_frequency + 1.0)) + 1.0
        weighted = matrix.copy().astype(np.float64)
        for row_number in range(weighted.shape[0]):
            start, end = weighted.indptr[row_number], weighted.indptr[row_number + 1]
            indices = weighted.indices[start:end]
            counts = weighted.data[start:end].copy()
            total = float(counts.sum())
            if total <= 0:
                continue
            output.loc[valid.index[row_number], "abstract_unseen_ngram_share"] = float(
                counts[history.document_frequency[indices] == 0].sum() / total
            )
            weighted.data[start:end] = counts * idf[indices]
            norm = float(np.linalg.norm(weighted.data[start:end]))
            if before and norm and centroid_norm:
                similarity = float(
                    np.dot(weighted.data[start:end] / norm, centroid[indices]) / centroid_norm
                )
                output.loc[valid.index[row_number], "abstract_prior_centroid_similarity"] = similarity
                output.loc[valid.index[row_number], "abstract_prior_centroid_novelty"] = 1.0 - similarity
            elif before == 0:
                weighted.data[start:end] = weighted.data[start:end] / norm if norm else weighted.data[start:end]
        if before:
            for row_number in range(weighted.shape[0]):
                start, end = weighted.indptr[row_number], weighted.indptr[row_number + 1]
                indices = weighted.indices[start:end]
                values = weighted.data[start:end]
                norm = float(np.linalg.norm(values))
                if norm:
                    weighted.data[start:end] = values / norm
        history.term_sum += np.asarray(weighted.sum(axis=0)).ravel()
        history.document_frequency += np.bincount(
            matrix.indices, minlength=history.n_features
        ).astype(history.document_frequency.dtype, copy=False)
        history.document_count += int(valid.shape[0])
    return output, history


def score_text_batch(batch: pd.DataFrame, history: Mapping[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    """Score a batch against history before updating it by application date."""

    if isinstance(history, HashedHistoryState):
        return _score_hashed_batch(batch, history)
    mutable_history = dict(history)
    return _exact_counter_score_batch(batch, mutable_history)


def _score_text_frame(
    text_core: pd.DataFrame,
    state: HashedHistoryState,
    vectorizer: HashingVectorizer,
) -> tuple[pd.DataFrame, HashedHistoryState]:
    if text_core.empty:
        return text_core.copy(), state
    parts: list[pd.DataFrame] = []
    ordered = text_core.sort_values(["application_date", "application_no"], kind="stable")
    for _, date_batch in ordered.groupby("application_date", sort=True):
        scored, state = _score_hashed_batch(date_batch, state, vectorizer)
        parts.append(scored)
    return pd.concat(parts, ignore_index=True).sort_values("application_no").reset_index(drop=True), state


def build_pit_text_metrics(
    text_core: pd.DataFrame,
    min_n: int = DEFAULT_MIN_NGRAM,
    max_n: int = DEFAULT_MAX_NGRAM,
    n_features: int = DEFAULT_HASH_FEATURES,
) -> pd.DataFrame:
    """Score application rows in date order with a strict prior-text history."""

    if text_core.empty:
        return text_core.copy()
    state = HashedHistoryState(
        document_count=0,
        document_frequency=np.zeros(n_features, dtype=np.int64),
        term_sum=np.zeros(n_features, dtype=np.float64),
    )
    scored, _ = _score_text_frame(text_core, state, _hashed_vectorizer(min_n, max_n, n_features))
    return scored


def run_text_metrics(
    text_root: Path,
    output_root: Path,
    minimum_group_patents: int = 10,
    min_n: int = DEFAULT_MIN_NGRAM,
    max_n: int = DEFAULT_MAX_NGRAM,
    n_features: int = DEFAULT_HASH_FEATURES,
) -> dict[str, object]:
    """Score text partitions chronologically and write all requested grains."""

    core_root = text_root / "text_core"
    edge_root = text_root / "text_firm_edge"
    metric_root = output_root / "text_patent_metrics"
    metric_root.mkdir(parents=True, exist_ok=True)
    state = HashedHistoryState(
        document_count=0,
        document_frequency=np.zeros(n_features, dtype=np.int64),
        term_sum=np.zeros(n_features, dtype=np.float64),
    )
    vectorizer = _hashed_vectorizer(min_n, max_n, n_features)
    market_parts: list[pd.DataFrame] = []
    group_parts: list[pd.DataFrame] = []
    patent_count = 0
    valid_count = 0
    years = sorted(
        path.name.split("=", 1)[1]
        for path in core_root.glob("application_year=*")
        if "=" in path.name
    )
    for year in years:
        core_paths = sorted((core_root / f"application_year={year}").glob("*.parquet"))
        if not core_paths:
            continue
        core = pd.concat([pd.read_parquet(path) for path in core_paths], ignore_index=True)
        core = core[core["baseline_eligible"].eq(1)].copy()
        core = core[core["application_date"].notna()].drop_duplicates("application_no")
        scored, state = _score_text_frame(core, state, vectorizer)
        scored.to_parquet(metric_root / f"metrics-{year}.parquet", index=False, compression="zstd")
        market_parts.append(aggregate_market_quarter(scored))
        patent_count += len(scored)
        valid_count += int(scored["abstract_clean"].astype(str).str.len().gt(0).sum())
        edge_paths = sorted((edge_root / f"application_year={year}").glob("*.parquet"))
        if edge_paths:
            edges = pd.concat([pd.read_parquet(path) for path in edge_paths], ignore_index=True)
            edges = edges[edges["baseline_eligible"].eq(1)].drop_duplicates(["listed_parent", "application_no"])
            group_input = edges[["listed_parent", "application_no", "quarter"]].merge(
                scored, on=["application_no", "quarter"], how="inner", suffixes=("", "_text")
            )
            group_parts.append(aggregate_group_quarter(group_input, minimum_patents=minimum_group_patents))
    market = pd.concat(market_parts, ignore_index=True) if market_parts else pd.DataFrame()
    groups = pd.concat(group_parts, ignore_index=True) if group_parts else pd.DataFrame()
    market.to_parquet(output_root / "quarterly_text_innovation_metrics.parquet", index=False, compression="zstd")
    groups.to_parquet(output_root / "firm_quarter_text_innovation_metrics.parquet", index=False, compression="zstd")
    market.to_csv(output_root / "quarterly_text_innovation_metrics.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(output_root / "firm_quarter_text_innovation_metrics.csv", index=False, encoding="utf-8-sig")
    return {
        "patent_count": patent_count,
        "abstract_valid_count": valid_count,
        "abstract_coverage_rate": valid_count / patent_count if patent_count else float("nan"),
        "history_document_count_at_end": state.document_count,
        "quarter_row_count": len(market),
        "firm_quarter_row_count": len(groups),
    }


def _write_text_diagnostics(output_root: Path) -> None:
    import matplotlib.pyplot as plt

    market_path = output_root / "quarterly_text_innovation_metrics.parquet"
    group_path = output_root / "firm_quarter_text_innovation_metrics.parquet"
    if not market_path.exists():
        return
    market = pd.read_parquet(market_path)
    groups = pd.read_parquet(group_path) if group_path.exists() else pd.DataFrame()
    diagnostics = output_root / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    x = market["quarter"].astype(str)
    positions = np.arange(len(market))
    tick_positions = positions[::8]
    tick_labels = x.iloc[::8]
    axes[0, 0].plot(positions, market["abstract_coverage_rate"], color="#2F4B66")
    axes[0, 0].set_title("Abstract coverage")
    axes[0, 0].set_ylabel("Share of baseline applications")
    axes[0, 1].plot(positions, market["abstract_prior_centroid_novelty_mean"], color="#2F4B66")
    axes[0, 1].set_title("PIT abstract novelty")
    axes[0, 1].set_ylabel("1 - prior centroid similarity")
    axes[1, 0].plot(positions, market["abstract_unseen_ngram_share_mean"], color="#2F4B66")
    axes[1, 0].set_title("Unseen character n-gram share")
    axes[1, 0].set_ylabel("Share")
    axes[1, 1].plot(positions, market["abstract_length_chars_mean"], color="#2F4B66")
    axes[1, 1].set_title("Abstract length")
    axes[1, 1].set_ylabel("Characters")
    for axis in axes.flat:
        axis.set_xticks(tick_positions, tick_labels, rotation=45, ha="right")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    figure.suptitle("Patent abstract text indicators", fontsize=15)
    figure.tight_layout()
    figure.savefig(diagnostics / "text_innovation_market_series.png", dpi=180)
    plt.close(figure)
    if not groups.empty:
        group_summary = (
            groups.groupby("quarter", as_index=False)
            .agg(
                group_count=("listed_parent", "nunique"),
                low_sample_share=("low_sample_flag", "mean"),
                group_patent_count=("patent_count", "sum"),
            )
        )
        group_summary.to_csv(diagnostics / "text_group_quarter_diagnostics.csv", index=False, encoding="utf-8-sig")


def run_text_pipeline(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.parents[3]
    raw_csv = project_root / str(config["patent_csv"])
    output_root = project_root / str(config["output_root"])
    extraction_summary_path = output_root / "text_extraction_summary.json"
    if not extraction_summary_path.exists() or not (output_root / "text_core").exists():
        extraction = run_text_extraction(
            raw_csv,
            output_root,
            shard_count=int(config["shard_count"]),
            buffer_records=int(config["buffer_records"]),
        )
    else:
        extraction = json.loads(extraction_summary_path.read_text(encoding="utf-8"))
    metrics = run_text_metrics(
        output_root,
        output_root,
        minimum_group_patents=int(config["minimum_group_patents"]),
        min_n=int(config["ngram_range"][0]),
        max_n=int(config["ngram_range"][1]),
        n_features=int(config["hash_features"]),
    )
    _write_text_diagnostics(output_root)
    manifest = {
        "pipeline_version": config["pipeline_version"],
        "config": config,
        "input_file": str(raw_csv.resolve()),
        "output_root": str(output_root.resolve()),
        "extraction": extraction,
        "metrics": metrics,
        "pit_rule": config["pit_rule"],
        "main_model_modified": False,
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    report = (
        "# Patent abstract text innovation indicators\n\n"
        "## Scope\n\n"
        "This is an isolated descriptive/predictive text-feature layer. It does not modify HWW, NVA, or Spec 6. "
        "The baseline contains invention and utility-model applications in listed-company plus controlled-subsidiary scope; "
        "design, associate, and joint-venture applications are retained for audit but excluded from baseline metrics.\n\n"
        "## Text and PIT method\n\n"
        "The source field is `摘要文本` from the raw CSV. Text is NFKC-normalized and whitespace-normalized. "
        f"The baseline uses character {config['ngram_range'][0]}–{config['ngram_range'][1]} grams in a fixed "
        f"{config['hash_features']:,}-dimensional HashingVectorizer space. "
        "For each application-date batch, document frequency and the TF-IDF centroid contain only strictly earlier application dates; "
        "all records on the same date are scored before the state is updated.\n\n"
        "`abstract_unseen_ngram_share` is the share of current text n-gram mass absent from earlier history. "
        "`abstract_prior_centroid_novelty` is one minus cosine similarity to the earlier-text TF-IDF centroid. "
        "These are text-content novelty proxies, not patent quality, technical validity, legal strength, or commercial value.\n\n"
        "## Output coverage\n\n"
        f"- Application-level baseline rows scored: {metrics['patent_count']:,}\n"
        f"- Applications with non-empty abstracts: {metrics['abstract_valid_count']:,}\n"
        f"- Abstract coverage: {metrics['abstract_coverage_rate']:.6%}\n"
        f"- Market-quarter rows: {metrics['quarter_row_count']:,}\n"
        f"- Group-quarter rows: {metrics['firm_quarter_row_count']:,}\n"
        f"- History documents at end: {metrics['history_document_count_at_end']:,}\n\n"
        "## Outputs\n\n"
        "- `quarterly_text_innovation_metrics.csv/parquet`: market-quarter indicators.\n"
        "- `firm_quarter_text_innovation_metrics.csv/parquet`: listed-company-group-quarter indicators.\n"
        "- `text_patent_metrics/`: application-level PIT scores partitioned by application year.\n"
        "- `diagnostics/text_innovation_market_series.png`: market time-series diagnostics.\n"
        "- `diagnostics/text_period_summary.csv/parquet`: requested period summaries.\n"
        "- `diagnostics/text_quality_audit.json`: structural and PIT quality audit.\n"
        "- `run_manifest.json`: input, configuration, PIT rule, and output summary.\n"
    )
    (output_root / "text_innovation_implementation_report.md").write_text(report, encoding="utf-8")
    return manifest


def _write_parquet_parts(frame: pd.DataFrame, directory: Path, prefix: str, part_number: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / f"{prefix}-{part_number:05d}.parquet", index=False, compression="zstd")


def _write_partitioned_text(frame: pd.DataFrame, root: Path, prefix: str) -> None:
    if frame.empty:
        return
    years = pd.to_datetime(frame["application_date"], errors="coerce").dt.year
    for year, part in frame.assign(_application_year=years).groupby("_application_year", dropna=False):
        label = "unknown" if pd.isna(year) else str(int(year))
        directory = root / f"application_year={label}"
        directory.mkdir(parents=True, exist_ok=True)
        part.drop(columns=["_application_year"]).to_parquet(
            directory / f"{prefix}.parquet", index=False, compression="zstd"
        )


def canonicalize_text_parts(
    output_root: Path,
    shard_count: int,
    logical_records: int,
    parse_failures: int,
    buffer_records: int,
) -> dict[str, object]:
    """Canonicalize persisted raw text parts without rescanning the source CSV."""

    raw_parts = output_root / "raw_text_parts"
    text_core_root = output_root / "text_core"
    text_edge_root = output_root / "text_firm_edge"
    core_count = 0
    edge_count = 0
    abstract_count = 0
    conflict_count = 0
    for shard in range(shard_count):
        paths = sorted(raw_parts.glob(f"shard-{shard:04d}-*.parquet"))
        if not paths:
            continue
        records = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        core, edges = canonicalize_text_records(records.to_dict(orient="records"))
        _write_partitioned_text(core, text_core_root, f"shard-{shard:04d}")
        _write_partitioned_text(edges, text_edge_root, f"shard-{shard:04d}")
        core_count += len(core)
        edge_count += len(edges)
        abstract_count += int(core["abstract_clean"].astype(str).str.len().gt(0).sum())
        conflict_count += int(core["abstract_conflict_flag"].sum())
    summary = {
        "logical_records": logical_records,
        "parse_failures": parse_failures,
        "application_count": core_count,
        "eligible_group_edge_count": edge_count,
        "abstract_present_application_count": abstract_count,
        "abstract_coverage_rate": abstract_count / core_count if core_count else float("nan"),
        "abstract_conflict_application_count": conflict_count,
        "shard_count": shard_count,
        "buffer_records": buffer_records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "text_extraction_summary.json").write_text(
        pd.Series(summary).to_json(indent=2, force_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def run_text_extraction(
    raw_csv: Path,
    output_root: Path,
    shard_count: int = 128,
    buffer_records: int = 2000,
) -> dict[str, object]:
    """Extract text-bearing rows, canonicalize applications, and write partitions."""

    raw_parts = output_root / "raw_text_parts"
    raw_parts.mkdir(parents=True, exist_ok=True)
    buffers: dict[int, list[dict[str, object]]] = {index: [] for index in range(shard_count)}
    part_numbers: Counter[int] = Counter()
    logical_records = 0
    parse_failures = 0

    def flush(shard: int) -> None:
        if not buffers[shard]:
            return
        part = pd.DataFrame(buffers[shard])
        _write_parquet_parts(part, raw_parts, f"shard-{shard:04d}", part_numbers[shard])
        part_numbers[shard] += 1
        buffers[shard].clear()

    for row in iter_full_logical_records(raw_csv):
        logical_records += 1
        if len(row) != EXPECTED_COLUMNS:
            parse_failures += 1
            continue
        record = _record_from_row(row)
        if not record["application_no"]:
            parse_failures += 1
            continue
        shard = _stable_shard(str(record["application_no"]), shard_count)
        buffers[shard].append(record)
        if len(buffers[shard]) >= buffer_records:
            flush(shard)
    for shard in range(shard_count):
        flush(shard)
    return canonicalize_text_parts(output_root, shard_count, logical_records, parse_failures, buffer_records)


def _aggregate_metric(group: pd.DataFrame, column: str) -> dict[str, float]:
    if column not in group.columns:
        return {
            f"{column}_mean": float("nan"),
            f"{column}_median": float("nan"),
            f"{column}_p90": float("nan"),
        }
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    if values.empty:
        return {
            f"{column}_mean": float("nan"),
            f"{column}_median": float("nan"),
            f"{column}_p90": float("nan"),
        }
    return {
        f"{column}_mean": float(values.mean()),
        f"{column}_median": float(values.median()),
        f"{column}_p90": float(values.quantile(0.9)),
    }


def aggregate_market_quarter(patents: pd.DataFrame) -> pd.DataFrame:
    """Aggregate application-level text metrics by application quarter."""

    if patents.empty:
        return pd.DataFrame()
    frame = patents.drop_duplicates("application_no").copy()
    metric_columns = [
        "abstract_prior_centroid_novelty",
        "abstract_unseen_ngram_share",
        "abstract_length_chars",
    ]
    rows: list[dict[str, object]] = []
    for quarter, group in frame.groupby("quarter", sort=True):
        valid = group["abstract_clean"].astype(str).str.len().gt(0)
        row: dict[str, object] = {
            "quarter": quarter,
            "patent_count": int(group["application_no"].nunique()),
            "abstract_valid_count": int(valid.sum()),
            "abstract_coverage_rate": float(valid.mean()),
        }
        for column in metric_columns:
            row.update(_aggregate_metric(group.loc[valid], column))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_group_quarter(patents: pd.DataFrame, minimum_patents: int = 10) -> pd.DataFrame:
    """Aggregate unique application-group edges by listed parent and quarter."""

    if patents.empty:
        return pd.DataFrame()
    frame = patents.drop_duplicates(["listed_parent", "application_no"]).copy()
    metric_columns = [
        "abstract_prior_centroid_novelty",
        "abstract_unseen_ngram_share",
        "abstract_length_chars",
    ]
    rows: list[dict[str, object]] = []
    for (listed_parent, quarter), group in frame.groupby(["listed_parent", "quarter"], sort=True):
        valid = group["abstract_clean"].astype(str).str.len().gt(0)
        row: dict[str, object] = {
            "listed_parent": listed_parent,
            "quarter": quarter,
            "patent_count": int(group["application_no"].nunique()),
            "abstract_valid_count": int(valid.sum()),
            "abstract_coverage_rate": float(valid.mean()),
            "low_sample_flag": int(group["application_no"].nunique() < minimum_patents),
        }
        for column in metric_columns:
            row.update(_aggregate_metric(group.loc[valid], column))
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "aggregate_group_quarter",
    "aggregate_market_quarter",
    "build_pit_text_metrics",
    "canonicalize_text_records",
    "canonicalize_text_parts",
    "char_ngram_counts",
    "clean_abstract",
    "iter_full_logical_records",
    "run_text_extraction",
    "run_text_metrics",
    "score_text_batch",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build point-in-time patent abstract text indicators")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config") / "text_default.json",
    )
    args = parser.parse_args()
    result = run_text_pipeline(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
