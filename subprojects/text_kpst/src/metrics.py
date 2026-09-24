"""KPST-style backward/forward patent similarity metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .similarity import mean_cosine_to_candidates


def compute_kpst_metrics(
    matrix: sparse.csr_matrix,
    *,
    application_years: Sequence[int],
    applicant_keys: Sequence[str],
    ipc_domain_keys: Sequence[str | None],
    window_years: int = 5,
    backward_scope: str = "same_ipc",
    forward_scope: str = "all_ipc",
    exclude_same_applicant: bool = True,
    block_size: int = 50_000,
) -> pd.DataFrame:
    """Compute average backward and forward similarities."""
    n = matrix.shape[0]
    if not (len(application_years) == len(applicant_keys) == len(ipc_domain_keys) == n):
        raise ValueError("metadata lengths must equal the matrix row count")
    if backward_scope not in {"same_ipc", "all_ipc"}:
        raise ValueError("backward_scope must be 'same_ipc' or 'all_ipc'")
    if forward_scope not in {"same_ipc", "all_ipc"}:
        raise ValueError("forward_scope must be 'same_ipc' or 'all_ipc'")

    years = np.asarray(application_years, dtype=np.int64)
    applicants = np.asarray(applicant_keys, dtype=object)
    domains = np.asarray(ipc_domain_keys, dtype=object)

    rows: list[dict[str, object]] = []
    all_indices = np.arange(n, dtype=np.int64)
    for i in range(n):
        year = int(years[i])
        backward = (years >= year - window_years) & (years <= year - 1)
        forward = (years >= year + 1) & (years <= year + window_years)
        if exclude_same_applicant:
            backward &= applicants != applicants[i]
            forward &= applicants != applicants[i]
        if backward_scope == "same_ipc":
            backward &= domains == domains[i]
        if forward_scope == "same_ipc":
            forward &= domains == domains[i]

        bs, backward_count = mean_cosine_to_candidates(
            matrix, i, all_indices[backward], block_size=block_size
        )
        fs, forward_count = mean_cosine_to_candidates(
            matrix, i, all_indices[forward], block_size=block_size
        )
        quality = (
            fs / bs
            if np.isfinite(fs) and np.isfinite(bs) and bs > 0
            else float("nan")
        )
        rows.append(
            {
                "patent_index": i,
                "application_year": year,
                "average_backward_similarity": bs,
                "average_forward_similarity": fs,
                "backward_comparison_count": backward_count,
                "forward_comparison_count": forward_count,
                "kpst_ratio_unadjusted": quality,
            }
        )
    return pd.DataFrame(rows)
