"""Quarterly aggregation of patent-level KPST scores."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read_edges(root: Path, year: int) -> pd.DataFrame:
    paths = sorted((root / f"application_year={year}").glob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _summary(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(keys, as_index=False)
        .agg(
            patent_count=("application_no", "nunique"),
            kpst_quality_mean=("kpst_quality", "mean"),
            kpst_quality_median=("kpst_quality", "median"),
            bs_mean=("bs", "mean"),
            fs_mean=("fs", "mean"),
        )
    )


def aggregate(
    canonical_root: Path,
    score_root: Path,
    output_root: Path,
    *,
    start_year: int = 1994,
) -> None:
    market_parts = []
    firm_parts = []

    for path in sorted(score_root.glob("*.parquet"), key=lambda p: int(p.stem)):
        year = int(path.stem)
        if year < start_year:
            continue

        scores = pd.read_parquet(path)
        scores["quarter"] = pd.to_datetime(scores["application_date"]).dt.to_period("Q").astype(str)

        market_parts.append(
            _summary(scores.loc[scores["baseline_firm"].eq(1)], ["quarter"])
        )

        edges = _read_edges(canonical_root / "firm_edges", year)
        edges = edges.loc[edges["baseline_firm"].eq(1)].drop_duplicates(
            ["listed_parent", "application_no"]
        )
        firm = edges[["listed_parent", "application_no"]].merge(
            scores, on="application_no", how="inner"
        )
        firm_parts.append(_summary(firm, ["listed_parent", "quarter"]))

    output_root.mkdir(parents=True, exist_ok=True)
    pd.concat(market_parts, ignore_index=True).to_parquet(
        output_root / "market_quarter_kpst.parquet", index=False
    )
    pd.concat(firm_parts, ignore_index=True).to_parquet(
        output_root / "firm_quarter_kpst.parquet", index=False
    )
