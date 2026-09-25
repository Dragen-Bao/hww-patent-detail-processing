"""Quarterly large-sample vector construction and KPST scoring."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .metrics import add_field_adjustment
from .preprocess import combine_text, tokenize
from .tfbidf import vectorize_quarter

SEP = "\x1f"


def _period(value: str) -> pd.Period:
    return pd.Period(value, freq="Q")


def _quarters(root: Path) -> list[str]:
    values = [
        p.name.split("=", 1)[1]
        for p in root.glob("application_quarter=*")
        if p.name.split("=", 1)[1] != "unknown"
    ]
    return sorted(values, key=lambda x: _period(x).ordinal)


def _window_quarters(quarter: str, window: int) -> tuple[list[str], list[str]]:
    current = _period(quarter)
    backward = [str(current - i) for i in range(window, 0, -1)]
    forward = [str(current + i) for i in range(1, window + 1)]
    return backward, forward


def _read_quarter(root: Path, quarter: str) -> pd.DataFrame:
    paths = sorted((root / f"application_quarter={quarter}").glob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def build_vectors(
    canonical_root: Path,
    vector_root: Path,
    *,
    text_fields: list[str],
) -> None:
    """Build one vocabulary and one sparse matrix per application quarter."""
    if vector_root.exists():
        shutil.rmtree(vector_root)
    token_root = vector_root / "_tokens"
    matrix_root = vector_root / "matrices"
    meta_root = vector_root / "metadata"
    token_root.mkdir(parents=True)
    matrix_root.mkdir()
    meta_root.mkdir()

    word_counts: Counter[str] = Counter()
    granted_counts: dict[str, int] = {}

    for quarter in _quarters(canonical_root):
        frame = _read_quarter(canonical_root, quarter)
        frame = frame.loc[frame["granted_invention"].eq(1)].drop_duplicates("application_no")
        granted_counts[quarter] = len(frame)
        if frame.empty:
            continue

        documents = [
            tokenize(combine_text(row, text_fields))
            for row in frame.to_dict(orient="records")
        ]
        keep = [bool(doc) for doc in documents]
        frame = frame.loc[keep].reset_index(drop=True)
        documents = [doc for doc, flag in zip(documents, keep) if flag]
        word_counts.update(word for doc in documents for word in doc)

        frame = frame[
            [
                "application_no",
                "application_date",
                "application_quarter",
                "applicant_key",
                "ipc_main_group",
                "baseline_firm",
            ]
        ].copy()
        frame["tokens"] = [SEP.join(doc) for doc in documents]
        frame.to_parquet(
            token_root / f"{quarter}.parquet", index=False, compression="zstd"
        )

    words = sorted(word_counts)
    vocabulary = {word: i for i, word in enumerate(words)}
    pd.DataFrame({"word": words, "id": range(len(words))}).to_parquet(
        vector_root / "vocabulary.parquet", index=False
    )
    (vector_root / "granted_counts.json").write_text(
        json.dumps(granted_counts), encoding="utf-8"
    )

    prior_n = 0
    prior_df = np.zeros(len(vocabulary), dtype=np.int64)

    token_paths = sorted(
        token_root.glob("*.parquet"),
        key=lambda p: _period(p.stem).ordinal,
    )
    for path in token_paths:
        quarter = path.stem
        frame = pd.read_parquet(path).sort_values(
            ["application_date", "application_no"]
        ).reset_index(drop=True)
        documents = [str(x).split(SEP) for x in frame["tokens"]]
        matrix, prior_n, prior_df = vectorize_quarter(
            documents, vocabulary, prior_n, prior_df
        )
        sparse.save_npz(matrix_root / f"{quarter}.npz", matrix, compressed=True)
        frame.drop(columns="tokens").to_parquet(
            meta_root / f"{quarter}.parquet", index=False, compression="zstd"
        )

    shutil.rmtree(token_root)


def _load(vector_root: Path, quarter: str) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    matrix = sparse.load_npz(vector_root / "matrices" / f"{quarter}.npz").tocsr()
    meta = pd.read_parquet(vector_root / "metadata" / f"{quarter}.parquet")
    if matrix.shape[0] != len(meta):
        raise ValueError(f"matrix/metadata mismatch in {quarter}")
    return matrix, meta


def _group_sum(
    matrix: sparse.csr_matrix,
    candidate_keys: pd.Series,
    target_keys: pd.Series,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    keys = list(dict.fromkeys(target_keys.astype(str)))
    lookup = {key: i for i, key in enumerate(keys)}
    rows, cols = [], []

    for j, key in enumerate(candidate_keys.fillna("").astype(str)):
        if key and key in lookup:
            rows.append(lookup[key])
            cols.append(j)

    incidence = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(len(keys), matrix.shape[0]),
    )
    return (incidence @ matrix).tocsr(), np.array([lookup[x] for x in target_keys.astype(str)])


def _aligned_dot(
    target: sparse.csr_matrix,
    sums: sparse.csr_matrix,
    codes: np.ndarray,
) -> np.ndarray:
    return np.asarray(target.multiply(sums[codes]).sum(axis=1)).ravel()


def _applicant_field_key(applicant: pd.Series, field: pd.Series) -> pd.Series:
    applicant = applicant.fillna("").astype(str)
    field = field.fillna("").astype(str)
    return pd.Series(
        np.where(
            applicant.ne("") & field.ne(""),
            applicant + "||" + field,
            "",
        ),
        index=applicant.index,
    )


def score_all_quarters(
    vector_root: Path,
    score_root: Path,
    *,
    window: int = 8,
) -> None:
    """Method 6 with previous/next 8 quarters."""
    if score_root.exists():
        shutil.rmtree(score_root)
    score_root.mkdir(parents=True)

    counts = {
        k: int(v)
        for k, v in json.loads(
            (vector_root / "granted_counts.json").read_text(encoding="utf-8")
        ).items()
    }
    quarters = sorted(
        (p.stem for p in (vector_root / "matrices").glob("*.npz")),
        key=lambda x: _period(x).ordinal,
    )
    available = set(quarters)
    first = _period(quarters[0])
    last = _period(quarters[-1])

    for quarter in quarters:
        target_matrix, target = _load(vector_root, quarter)
        n = len(target)
        bs_sum = np.zeros(n)
        fs_sum = np.zeros(n)
        target_field = target["ipc_main_group"].fillna("").astype(str)
        target_applicant = target["applicant_key"].fillna("").astype(str)
        backward, forward = _window_quarters(quarter, window)

        for other in backward:
            if other not in available:
                continue
            matrix, meta = _load(vector_root, other)

            field_sum, field_code = _group_sum(
                matrix,
                meta["ipc_main_group"].fillna("").astype(str),
                target_field,
            )
            bs_sum += _aligned_dot(target_matrix, field_sum, field_code)

            same_sum, same_code = _group_sum(
                matrix,
                _applicant_field_key(meta["applicant_key"], meta["ipc_main_group"]),
                _applicant_field_key(target_applicant, target_field),
            )
            bs_sum -= _aligned_dot(target_matrix, same_sum, same_code)

        for other in forward:
            if other not in available:
                continue
            matrix, meta = _load(vector_root, other)
            fs_sum += (
                target_matrix @ sparse.csr_matrix(matrix.sum(axis=0)).T
            ).toarray().ravel()

            same_sum, same_code = _group_sum(
                matrix,
                meta["applicant_key"].fillna("").astype(str),
                target_applicant,
            )
            fs_sum -= _aligned_dot(target_matrix, same_sum, same_code)

        bs_den = sum(counts.get(q, 0) for q in backward)
        fs_den = sum(counts.get(q, 0) for q in forward)

        out = target.copy()
        out["bs"] = bs_sum / bs_den if bs_den else np.nan
        out["fs"] = fs_sum / fs_den if fs_den else np.nan
        out.loc[target_field.eq(""), "bs"] = np.nan
        out["window_complete"] = int(
            _period(quarter) - window >= first
            and _period(quarter) + window <= last
        )
        out = add_field_adjustment(out)
        out.to_parquet(
            score_root / f"{quarter}.parquet", index=False, compression="zstd"
        )
