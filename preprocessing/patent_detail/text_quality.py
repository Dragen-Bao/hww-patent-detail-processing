"""Quality checks and period summaries for the patent abstract text outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE_RELATIONS = {"上市公司本身", "上市公司的子公司"}
PERIODS = {
    "1994-1999": (1994, 1999),
    "2000-2009": (2000, 2009),
    "2010-2019": (2010, 2019),
    "2020-2026": (2020, 2026),
}


def _metric_files(root: Path) -> list[Path]:
    return sorted((root / "text_patent_metrics").glob("metrics-*.parquet"))


def _load_metrics(root: Path) -> pd.DataFrame:
    columns = [
        "application_no",
        "application_date",
        "relation",
        "patent_type_group",
        "baseline_eligible",
        "abstract_length_chars",
        "abstract_history_document_count",
        "abstract_unseen_ngram_share",
        "abstract_prior_centroid_novelty",
        "abstract_conflict_flag",
        "application_date_conflict",
    ]
    frames = [pd.read_parquet(path, columns=columns) for path in _metric_files(root)]
    metrics = pd.concat(frames, ignore_index=True)
    metrics["application_date"] = pd.to_datetime(metrics["application_date"])
    return metrics


def _period_for_year(year: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=year.index, dtype="string")
    for label, (start, end) in PERIODS.items():
        result.loc[year.between(start, end)] = label
    return result


def build_period_summary(root: Path) -> pd.DataFrame:
    """Build the four requested period summaries from patent-level outputs."""
    metrics = _load_metrics(root)
    metrics["period"] = _period_for_year(metrics["application_date"].dt.year)
    metrics = metrics.loc[metrics["period"].notna()].copy()
    valid = metrics["abstract_length_chars"].fillna(0).gt(0)

    summary = (
        metrics.assign(abstract_valid=valid)
        .groupby("period", observed=False)
        .agg(
            patent_count=("application_no", "size"),
            abstract_valid_count=("abstract_valid", "sum"),
            abstract_coverage_rate=("abstract_valid", "mean"),
            abstract_prior_centroid_novelty_mean=("abstract_prior_centroid_novelty", "mean"),
            abstract_prior_centroid_novelty_median=("abstract_prior_centroid_novelty", "median"),
            abstract_prior_centroid_novelty_p90=("abstract_prior_centroid_novelty", lambda s: s.quantile(0.9)),
            abstract_unseen_ngram_share_mean=("abstract_unseen_ngram_share", "mean"),
            abstract_unseen_ngram_share_median=("abstract_unseen_ngram_share", "median"),
            abstract_unseen_ngram_share_p90=("abstract_unseen_ngram_share", lambda s: s.quantile(0.9)),
            abstract_length_chars_mean=("abstract_length_chars", "mean"),
            abstract_length_chars_median=("abstract_length_chars", "median"),
            abstract_length_chars_p90=("abstract_length_chars", lambda s: s.quantile(0.9)),
            cold_start_share=("abstract_history_document_count", lambda s: s.eq(0).mean()),
        )
        .reset_index()
    )
    market = pd.read_parquet(root / "quarterly_text_innovation_metrics.parquet")
    market["year"] = market["quarter"].str[:4].astype(int)
    market["period"] = _period_for_year(market["year"])
    quarter_counts = market.dropna(subset=["period"]).groupby("period", observed=False).size()
    summary["quarter_count"] = summary["period"].map(quarter_counts).astype("Int64")
    order = list(PERIODS)
    summary["_order"] = summary["period"].map({v: i for i, v in enumerate(order)})
    return summary.sort_values("_order").drop(columns="_order")


def audit_outputs(root: Path) -> dict[str, object]:
    """Run structural, PIT, and range checks and write a JSON-friendly report."""
    metrics = _load_metrics(root)
    market = pd.read_parquet(root / "quarterly_text_innovation_metrics.parquet")
    firm = pd.read_parquet(root / "firm_quarter_text_innovation_metrics.parquet")
    extraction = json.loads((root / "text_extraction_summary.json").read_text(encoding="utf-8"))

    metrics["abstract_valid"] = metrics["abstract_length_chars"].fillna(0).gt(0)
    metrics["patent_type_group"] = metrics["patent_type_group"].astype("string")
    baseline = metrics["baseline_eligible"].fillna(False)
    date_counts = metrics.loc[metrics["abstract_valid"]].groupby("application_date").size().sort_index()
    history_by_date = metrics.groupby("application_date")["abstract_history_document_count"].agg(["min", "max"])
    valid_by_all_date = date_counts.reindex(history_by_date.index, fill_value=0)
    history_by_date["expected"] = valid_by_all_date.cumsum() - valid_by_all_date
    history_mismatch_dates = int((history_by_date["min"] != history_by_date["max"]).sum())
    history_mismatch_dates += int((history_by_date["min"] != history_by_date["expected"]).sum())

    market_expected_rate = market["abstract_valid_count"] / market["patent_count"].replace(0, np.nan)
    audit = {
        **extraction,
        "metric_rows": int(len(metrics)),
        "metric_duplicate_application_nos": int(metrics["application_no"].duplicated().sum()),
        "market_quarter_rows": int(len(market)),
        "market_duplicate_quarters": int(market["quarter"].duplicated().sum()),
        "group_rows": int(len(firm)),
        "group_duplicate_keys": int(firm.duplicated(subset=["listed_parent", "quarter"]).sum()),
        "baseline_type_bad_rows": int((baseline & ~metrics["patent_type_group"].isin(["invention", "utility_model"])).sum()),
        "baseline_relation_bad_rows": int((baseline & ~metrics["relation"].isin(BASELINE_RELATIONS)).sum()),
        "application_date_conflict_rows": int(metrics["application_date_conflict"].fillna(False).sum()),
        "abstract_valid_count": int(metrics["abstract_valid"].sum()),
        "abstract_coverage_rate": float(metrics["abstract_valid"].mean()),
        "same_day_history_max_unique_values": int(metrics.groupby("application_date")["abstract_history_document_count"].nunique().max()),
        "pit_history_mismatch_dates": history_mismatch_dates,
        "history_count_min": int(metrics["abstract_history_document_count"].min()),
        "history_count_max": int(metrics["abstract_history_document_count"].max()),
        "unseen_ngram_min": float(metrics["abstract_unseen_ngram_share"].min()),
        "unseen_ngram_max": float(metrics["abstract_unseen_ngram_share"].max()),
        "novelty_min": float(metrics["abstract_prior_centroid_novelty"].min()),
        "novelty_max": float(metrics["abstract_prior_centroid_novelty"].max()),
        "coverage_bad_quarters": int((market["abstract_valid_count"] > market["patent_count"]).sum()),
        "coverage_rate_mismatch_quarters": int((market["abstract_coverage_rate"] - market_expected_rate).abs().gt(1e-12).sum()),
        "group_low_sample_share_mean": float(firm["low_sample_flag"].mean()) if "low_sample_flag" in firm else None,
    }
    return audit


def write_quality_outputs(root: Path) -> tuple[dict[str, object], pd.DataFrame]:
    """Write the audit JSON and the requested period summary files."""
    diagnostics = root / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    audit = audit_outputs(root)
    summary = build_period_summary(root)
    (diagnostics / "text_quality_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary.to_csv(diagnostics / "text_period_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_parquet(diagnostics / "text_period_summary.parquet", index=False, compression="zstd")
    return audit, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    audit, summary = write_quality_outputs(args.root)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))
