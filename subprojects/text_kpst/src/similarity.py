"""Memory-bounded cosine-similarity helpers for sparse patent vectors."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import sparse


def mean_cosine_to_candidates(
    normalized_matrix: sparse.csr_matrix,
    patent_index: int,
    candidate_indices: Iterable[int],
    *,
    block_size: int = 50_000,
) -> tuple[float, int]:
    """Return mean cosine similarity without materializing a full N x N matrix."""
    indices = np.fromiter((int(i) for i in candidate_indices), dtype=np.int64)
    if indices.size == 0:
        return float("nan"), 0
    row = normalized_matrix.getrow(int(patent_index))
    total_similarity = 0.0
    for start in range(0, indices.size, block_size):
        block = indices[start : start + block_size]
        similarities = row @ normalized_matrix[block].T
        total_similarity += float(similarities.sum())
    return total_similarity / int(indices.size), int(indices.size)
