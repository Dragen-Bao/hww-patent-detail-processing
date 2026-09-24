"""Patent-level BS, FS, field adjustment and final KPST quality."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


def _window_year_count(
    counts_by_year: Mapping[int, int], start_year: int, end_year: int
) -> int:
    return int(sum(int(counts_by_year.get(year, 0)) for year in range(start_year, end_year + 1)))


def compute_kpst_metrics(
    matrix: sparse.csr_matrix,
    *,
    application_years: Sequence[int],
    applicant_keys: Sequence[str],
    ipc_domain_keys: Sequence[str | None],
    window_years: int = 5,
    exclude_same_applicant: bool = True,
    backward_scope: str = "same_ipc",
    forward_scope: str = "all_ipc",
    denominator_counts_by_year: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """Compute equations (7)-(8) exactly for an in-memory sample.

    The denominator is the total number of granted invention patents in the
    corresponding year window, as stated in the appendix, rather than the
    number of surviving comparison candidates after exclusions.
    """
    n = matrix.shape[0]
    if not (len(application_years) == len(applicant_keys) == len(ipc_domain_keys) == n):
        raise ValueError("metadata lengths must equal matrix rows")
    if backward_scope not in {"same_ipc", "all_ipc"}:
        raise ValueError("invalid backward_scope")
    if forward_scope not in {"same_ipc", "all_ipc"}:
        raise ValueError("invalid forward_scope")

    years = np.asarray(application_years, dtype=np.int64)
    applicants = np.asarray(applicant_keys, dtype=object)
    domains = np.asarray(ipc_domain_keys, dtype=object)
    counts = (
        {int(year): int((years == year).sum()) for year in np.unique(years)}
        if denominator_counts_by_year is None
        else {int(k): int(v) for k, v in denominator_counts_by_year.items()}
    )

    rows: list[dict[str, object]] = []
    for i in range(n):
        year = int(years[i])
        backward_mask = (years >= year - window_years) & (years <= year - 1)
        forward_mask = (years >= year + 1) & (years <= year + window_years)
        if backward_scope == "same_ipc":
            backward_mask &= domains == domains[i]
        if forward_scope == "same_ipc":
            forward_mask &= domains == domains[i]
        if exclude_same_applicant:
            backward_mask &= applicants != applicants[i]
            forward_mask &= applicants != applicants[i]

        backward_indices = np.flatnonzero(backward_mask)
        forward_indices = np.flatnonzero(forward_mask)
        row = matrix.getrow(i)
        backward_sum = (
            float((row @ matrix[backward_indices].T).sum()) if backward_indices.size else 0.0
        )
        forward_sum = (
            float((row @ matrix[forward_indices].T).sum()) if forward_indices.size else 0.0
        )
        backward_denominator = _window_year_count(
            counts, year - window_years, year - 1
        )
        forward_denominator = _window_year_count(
            counts, year + 1, year + window_years
        )
        bs = backward_sum / backward_denominator if backward_denominator > 0 else float("nan")
        fs = forward_sum / forward_denominator if forward_denominator > 0 else float("nan")
        rows.append(
            {
                "patent_index": i,
                "application_year": year,
                "average_backward_similarity": bs,
                "average_forward_similarity": fs,
                "backward_similarity_sum": backward_sum,
                "forward_similarity_sum": forward_sum,
                "backward_comparison_count": int(backward_indices.size),
                "forward_comparison_count": int(forward_indices.size),
                "backward_denominator_patents": backward_denominator,
                "forward_denominator_patents": forward_denominator,
                "kpst_ratio_unadjusted": (
                    fs / bs if np.isfinite(fs) and np.isfinite(bs) and bs != 0 else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def add_field_adjustment(
    metrics: pd.DataFrame,
    *,
    application_year_col: str = "application_year",
    domain_col: str = "ipc_domain_key",
    backward_col: str = "average_backward_similarity",
) -> pd.DataFrame:
    """Apply equations (9)-(10): xi_t(T) and adjusted patent quality."""
    frame = metrics.copy()
    frame["field_similarity_adjustment"] = np.nan
    frame["kpst_quality_adjusted"] = np.nan
    for _, year_group in frame.groupby(application_year_col, observed=True):
        total_n = len(year_group)
        total_bs = pd.to_numeric(year_group[backward_col], errors="coerce").sum(min_count=1)
        if total_n <= 0 or not np.isfinite(total_bs) or total_bs == 0:
            continue
        for _, group in year_group.groupby(domain_col, dropna=False, observed=True):
            n_t = len(group)
            field_bs = pd.to_numeric(group[backward_col], errors="coerce").sum(min_count=1)
            if n_t <= 0 or not np.isfinite(field_bs):
                continue
            xi = (total_n / n_t) * (field_bs / total_bs)
            frame.loc[group.index, "field_similarity_adjustment"] = xi
    ratio = pd.to_numeric(frame["kpst_ratio_unadjusted"], errors="coerce")
    frame["kpst_quality_adjusted"] = frame["field_similarity_adjustment"] * ratio
    return frame
