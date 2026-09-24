"""Point-in-time TF-BIDF sparse representations."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy import sparse


def build_vocabulary(
    tokenized_documents: Sequence[Sequence[str]],
    *,
    min_corpus_frequency: int = 1,
) -> dict[str, int]:
    """Build a deterministic one-word-one-dimension vocabulary."""
    counts: Counter[str] = Counter(
        token for document in tokenized_documents for token in document
    )
    kept = sorted(
        token for token, count in counts.items() if count >= min_corpus_frequency
    )
    return {token: index for index, token in enumerate(kept)}


def _bidf(prior_documents: int, prior_document_frequency: int, smoothing: float) -> float:
    """Smoothed backward IDF: log((N_prior+s)/(df_prior+s))."""
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")
    return math.log(
        (float(prior_documents) + smoothing)
        / (float(prior_document_frequency) + smoothing)
    )


def build_pit_tfbidf(
    tokenized_documents: Sequence[Sequence[str]],
    application_years: Sequence[int],
    *,
    vocabulary: dict[str, int] | None = None,
    min_corpus_frequency: int = 1,
    smoothing: float = 1.0,
    l2_normalize: bool = True,
) -> tuple[sparse.csr_matrix, dict[str, int], np.ndarray]:
    """Build TF-BIDF vectors using strictly earlier application years."""
    if len(tokenized_documents) != len(application_years):
        raise ValueError("tokenized_documents and application_years must have equal length")

    vocabulary = vocabulary or build_vocabulary(
        tokenized_documents, min_corpus_frequency=min_corpus_frequency
    )
    n_documents = len(tokenized_documents)
    n_features = len(vocabulary)
    prior_counts = np.zeros(n_documents, dtype=np.int64)

    document_frequency = np.zeros(n_features, dtype=np.int64)
    prior_documents = 0
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    order = sorted(range(n_documents), key=lambda i: (int(application_years[i]), i))
    position = 0
    while position < len(order):
        year = int(application_years[order[position]])
        end = position
        while end < len(order) and int(application_years[order[end]]) == year:
            end += 1
        batch = order[position:end]

        for document_index in batch:
            prior_counts[document_index] = prior_documents
            counts = Counter(
                token for token in tokenized_documents[document_index] if token in vocabulary
            )
            total = sum(counts.values())
            if total <= 0:
                continue
            row_values: list[tuple[int, float]] = []
            for token, count in counts.items():
                column = vocabulary[token]
                tf = count / total
                weight = tf * _bidf(
                    prior_documents,
                    int(document_frequency[column]),
                    smoothing,
                )
                if weight != 0.0:
                    row_values.append((column, weight))
            if l2_normalize and row_values:
                norm = math.sqrt(sum(value * value for _, value in row_values))
                if norm > 0:
                    row_values = [(column, value / norm) for column, value in row_values]
            for column, value in row_values:
                rows.append(document_index)
                cols.append(column)
                data.append(value)

        # Freeze within year: update history only after every patent in the year is scored.
        for document_index in batch:
            seen_columns = {
                vocabulary[token]
                for token in tokenized_documents[document_index]
                if token in vocabulary
            }
            for column in seen_columns:
                document_frequency[column] += 1
            prior_documents += 1
        position = end

    matrix = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float64), (rows, cols)),
        shape=(n_documents, n_features),
        dtype=np.float64,
    )
    return matrix, vocabulary, prior_counts
