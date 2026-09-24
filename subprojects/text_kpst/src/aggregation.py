"""Aggregate patent-level KPST scores to market-quarter and firm-quarter panels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


QUALITY_COLUMNS = [
    "average_backward_similarity",
    "average_forward_similarity",
    "kpst_ratio_unadjusted",
    "field_similarity_adjustment",
    "kpst_quality_adjusted",
    "kpst_quality_adjusted_complete",
]


def _read_partitioned_year(root: Path, year: int) -> pd.DataFrame:
    paths = sorted((root / f"application_year={year}").glob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for group_key, group in frame.groupby(keys, sort=True, observed=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row: dict[str, object] = dict(zip(keys, group_key, strict=True))
        row["patent_count"] = int(group["application_no"].nunique())
        row["text_score_count"] = int(group["kpst_quality_adjusted"].notna().sum())
        row["complete_window_score_count"] = int(
            group["kpst_quality_adjusted_complete"].notna().sum()
        )
        for column in QUALITY_COLUMNS:
            values = pd.to_numeric(group[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            row[f"{column}_mean"] = float(values.mean()) if not values.empty else float("nan")
            row[f"{column}_median"] = float(values.median()) if not values.empty else float("nan")
            row[f"{column}_sum"] = float(values.sum()) if not values.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_outputs(
    canonical_root: Path,
    score_root: Path,
    output_root: Path,
    *,
    formal_sample_start_year: int = 1994,
) -> dict[str, object]:
    """Build listed-company market and firm panels from patent-level quality."""
    market_parts: list[pd.DataFrame] = []
    firm_parts: list[pd.DataFrame] = []
    score_files = sorted(
        score_root.glob("kpst-*.parquet"), key=lambda p: int(p.stem.split("-")[-1])
    )
    for score_path in score_files:
        year = int(score_path.stem.split("-")[-1])
        if year < formal_sample_start_year:
            continue
        scores = pd.read_parquet(score_path)
        scores["quarter"] = pd.to_datetime(scores["application_date"]).dt.to_period("Q").astype(str)

        market_input = scores.loc[scores["baseline_firm_eligible"].eq(1)].copy()
        market_parts.append(_aggregate(market_input, ["quarter"]))

        edges = _read_partitioned_year(canonical_root / "firm_edges", year)
        if edges.empty:
            continue
        edges = edges.loc[edges["baseline_firm_eligible"].eq(1)].drop_duplicates(
            ["listed_parent", "application_no"]
        )
        firm_input = edges[["listed_parent", "application_no"]].merge(
            scores, on="application_no", how="inner"
        )
        firm_parts.append(_aggregate(firm_input, ["listed_parent", "quarter"]))

    market = pd.concat(market_parts, ignore_index=True) if market_parts else pd.DataFrame()
    firm = pd.concat(firm_parts, ignore_index=True) if firm_parts else pd.DataFrame()
    output_root.mkdir(parents=True, exist_ok=True)
    market.to_parquet(output_root / "market_quarter_kpst.parquet", index=False, compression="zstd")
    firm.to_parquet(output_root / "firm_quarter_kpst.parquet", index=False, compression="zstd")
    market.to_csv(output_root / "market_quarter_kpst.csv", index=False, encoding="utf-8-sig")
    firm.to_csv(output_root / "firm_quarter_kpst.csv", index=False, encoding="utf-8-sig")
    summary = {
        "market_quarter_rows": len(market),
        "firm_quarter_rows": len(firm),
        "formal_sample_start_year": formal_sample_start_year,
    }
    (output_root / "aggregation_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
