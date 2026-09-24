"""TF-BIDF vectors used by the Chen/KPST text measure."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy import sparse


def bidf(prior_n: int, prior_df: int) -> float:
    return math.log(prior_n / (1 + prior_df)) if prior_n > 0 else float("nan")


def build_vocabulary(documents: Sequence[Sequence[str]]) -> dict[str, int]:
    words = sorted({word for document in documents for word in document})
    return {word: i for i, word in enumerate(words)}


def vectorize(
    documents: Sequence[Sequence[str]],
    dates: Sequence[str],
    vocabulary: dict[str, int],
    prior_n: int,
    prior_df: np.ndarray,
) -> tuple[sparse.csr_matrix, int, np.ndarray]:
    """Vectorize one chronological shard using only strictly earlier dates."""
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    order = sorted(range(len(documents)), key=lambda i: (dates[i], i))
    position = 0

    while position < len(order):
        date = dates[order[position]]
        end = position
        while end < len(order) and dates[order[end]] == date:
            end += 1
        batch = order[position:end]

        for i in batch:
            if prior_n == 0:
                continue
            counts = Counter(word for word in documents[i] if word in vocabulary)
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

        for i in batch:
            for j in {vocabulary[w] for w in documents[i] if w in vocabulary}:
                prior_df[j] += 1
            prior_n += 1

        position = end

    matrix = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(documents), len(vocabulary)), dtype=float
    )
    return matrix, prior_n, prior_df
