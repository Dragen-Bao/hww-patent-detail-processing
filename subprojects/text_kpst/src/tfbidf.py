"""Word-level TF-BIDF vectors following the supplied paper appendix."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse


def build_vocabulary(
    tokenized_documents: Sequence[Sequence[str]],
    *,
    min_corpus_frequency: int = 1,
) -> dict[str, int]:
    """One real word = one dimension; no feature hashing."""
    counts: Counter[str] = Counter(
        token for document in tokenized_documents for token in document
    )
    kept = sorted(
        token for token, count in counts.items() if count >= min_corpus_frequency
    )
    return {token: index for index, token in enumerate(kept)}


def bidf_weight(prior_patents: int, prior_docs_with_word: int) -> float:
    """Equation (3): log(N_prior / (1 + df_prior))."""
    if prior_patents <= 0:
        return float("nan")
    return math.log(float(prior_patents) / (1.0 + float(prior_docs_with_word)))


@dataclass
class BIDFHistoryState:
    document_count: int
    document_frequency: np.ndarray

    @classmethod
    def empty(cls, n_features: int) -> "BIDFHistoryState":
        return cls(0, np.zeros(n_features, dtype=np.int64))


def vectorize_with_history(
    tokenized_documents: Sequence[Sequence[str]],
    time_keys: Sequence[object],
    *,
    vocabulary: dict[str, int],
    history: BIDFHistoryState,
) -> tuple[sparse.csr_matrix, BIDFHistoryState, np.ndarray]:
    """Vectorize a chronological shard while carrying strict-prior BIDF state."""
    if len(tokenized_documents) != len(time_keys):
        raise ValueError("tokenized_documents and time_keys must have equal length")
    n_documents = len(tokenized_documents)
    n_features = len(vocabulary)
    if len(history.document_frequency) != n_features:
        raise ValueError("history dimension and vocabulary size differ")
    prior_counts = np.zeros(n_documents, dtype=np.int64)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    order = sorted(range(n_documents), key=lambda i: (str(time_keys[i]), i))
    position = 0
    while position < len(order):
        key = str(time_keys[order[position]])
        end = position
        while end < len(order) and str(time_keys[order[end]]) == key:
            end += 1
        batch = order[position:end]
        before = history.document_count

        for document_index in batch:
            prior_counts[document_index] = before
            if before <= 0:
                continue
            counts = Counter(
                token for token in tokenized_documents[document_index] if token in vocabulary
            )
            total_terms = sum(counts.values())
            if total_terms <= 0:
                continue
            values: list[tuple[int, float]] = []
            for token, count in counts.items():
                column = vocabulary[token]
                weight = (count / total_terms) * bidf_weight(
                    before, int(history.document_frequency[column])
                )
                if weight != 0.0 and math.isfinite(weight):
                    values.append((column, weight))
            norm = math.sqrt(sum(value * value for _, value in values))
            if norm > 0:
                values = [(column, value / norm) for column, value in values]
            for column, value in values:
                rows.append(document_index)
                cols.append(column)
                data.append(value)

        for document_index in batch:
            seen = {
                vocabulary[token]
                for token in tokenized_documents[document_index]
                if token in vocabulary
            }
            for column in seen:
                history.document_frequency[column] += 1
            history.document_count += 1
        position = end

    matrix = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float64), (rows, cols)),
        shape=(n_documents, n_features),
        dtype=np.float64,
    )
    return matrix, history, prior_counts


def build_pit_tfbidf(
    tokenized_documents: Sequence[Sequence[str]],
    time_keys: Sequence[object],
    *,
    vocabulary: dict[str, int] | None = None,
    min_corpus_frequency: int = 1,
) -> tuple[sparse.csr_matrix, dict[str, int], np.ndarray]:
    """Convenience in-memory wrapper around vectorize_with_history."""
    if len(tokenized_documents) != len(time_keys):
        raise ValueError("tokenized_documents and time_keys must have equal length")
    vocabulary = vocabulary or build_vocabulary(
        tokenized_documents, min_corpus_frequency=min_corpus_frequency
    )
    history = BIDFHistoryState.empty(len(vocabulary))
    matrix, _, prior_counts = vectorize_with_history(
        tokenized_documents,
        time_keys,
        vocabulary=vocabulary,
        history=history,
    )
    return matrix, vocabulary, prior_counts
