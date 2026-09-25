"""Quarterly TF-BIDF vectors used by the Chen/KPST adaptation."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy import sparse


def bidf(prior_n: int, prior_df: int) -> float:
    return math.log(prior_n / (1 + prior_df)) if prior_n > 0 else float("nan")


def vectorize_quarter(
    documents: Sequence[Sequence[str]],
    vocabulary: dict[str, int],
    prior_n: int,
    prior_df: np.ndarray,
) -> tuple[sparse.csr_matrix, int, np.ndarray]:
    """Use only patents from earlier quarters to construct current-quarter vectors."""
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for i, document in enumerate(documents):
        if prior_n == 0:
            continue
        counts = Counter(word for word in document if word in vocabulary)
        total = sum(counts.values())
        values: list[tuple[int, float]] = []

        for word, count in counts.items():
            j = vocabulary[word]
            value = (count / total) * bidf(prior_n, int(prior_df[j]))
            if value and math.isfinite(value):
                values.append((j, value))

        norm = math.sqrt(sum(value * value for _, value in values))
        if norm:
            values = [(j, value / norm) for j, value in values]

        for j, value in values:
            rows.append(i)
            cols.append(j)
            data.append(value)

    matrix = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(documents), len(vocabulary)), dtype=float
    )

    for document in documents:
        for j in {vocabulary[word] for word in document if word in vocabulary}:
            prior_df[j] += 1
        prior_n += 1

    return matrix, prior_n, prior_df
