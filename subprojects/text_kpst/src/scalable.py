"""Large-sample vector construction and KPST scoring."""

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
from .tfbidf import vectorize

SEP = "\x1f"


def _years(root: Path) -> list[int]:
    return sorted(
        int(p.name.split("=")[1])
        for p in root.glob("application_year=*")
        if p.name.split("=")[1].isdigit()
    )


def _read_year(root: Path, year: int) -> pd.DataFrame:
    paths = sorted((root / f"application_year={year}").glob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def build_vectors(
    canonical_root: Path,
    vector_root: Path,
    *,
    text_fields: list[str],
) -> None:
    """Tokenize once, build one vocabulary, then write yearly sparse matrices."""
    if vector_root.exists():
        shutil.rmtree(vector_root)
    token_root = vector_root / "_tokens"
    matrix_root = vector_root / "matrices"
    meta_root = vector_root / "metadata"
    token_root.mkdir(parents=True)
    matrix_root.mkdir()
    meta_root.mkdir()

    word_counts: Counter[str] = Counter()
    granted_counts: dict[int, int] = {}

    for year in _years(canonical_root):
        frame = _read_year(canonical_root, year)
        frame = frame.loc[frame["granted_invention"].eq(1)].drop_duplicates("application_no")
        granted_counts[year] = len(frame)
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
                "application_year",
                "applicant_key",
                "ipc_main_group",
                "baseline_firm",
            ]
        ].copy()
        frame["tokens"] = [SEP.join(doc) for doc in documents]
        frame.to_parquet(token_root / f"{year}.parquet", index=False, compression="zstd")

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

    for path in sorted(token_root.glob("*.parquet"), key=lambda p: int(p.stem)):
        year = int(path.stem)
        frame = pd.read_parquet(path).sort_values(
            ["application_date", "application_no"]
        ).reset_index(drop=True)
        documents = [str(x).split(SEP) for x in frame["tokens"]]
        matrix, prior_n, prior_df = vectorize(
            documents,
            frame["application_date"].astype(str).tolist(),
            vocabulary,
            prior_n,
            prior_df,
        )
        sparse.save_npz(matrix_root / f"{year}.npz", matrix, compressed=True)
        frame.drop(columns="tokens").to_parquet(
            meta_root / f"{year}.parquet", index=False, compression="zstd"
        )

    shutil.rmtree(token_root)


def _load(vector_root: Path, year: int) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    matrix = sparse.load_npz(vector_root / "matrices" / f"{year}.npz").tocsr()
    meta = pd.read_parquet(vector_root / "metadata" / f"{year}.parquet")
    if matrix.shape[0] != len(meta):
        raise ValueError(f"matrix/metadata mismatch in {year}")
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


def score_all_years(
    vector_root: Path,
    score_root: Path,
    *,
    window: int = 5,
) -> None:
    """Method 6: backward same IPC, forward all IPC, exclude same applicant."""
    if score_root.exists():
        shutil.rmtree(score_root)
    score_root.mkdir(parents=True)

    counts = {
        int(k): int(v)
        for k, v in json.loads(
            (vector_root / "granted_counts.json").read_text(encoding="utf-8")
        ).items()
    }
    years = sorted(
        int(p.stem) for p in (vector_root / "matrices").glob("*.npz")
    )

    for year in years:
        target_matrix, target = _load(vector_root, year)
        n = len(target)
        bs_sum = np.zeros(n)
        fs_sum = np.zeros(n)
        target_field = target["ipc_main_group"].fillna("").astype(str)
        target_applicant = target["applicant_key"].fillna("").astype(str)

        for other in range(year - window, year):
            if other not in years:
                continue
            matrix, meta = _load(vector_root, other)

            field_sum, field_code = _group_sum(
                matrix,
                meta["ipc_main_group"].fillna("").astype(str),
                target_field,
            )
            bs_sum += _aligned_dot(target_matrix, field_sum, field_code)

            candidate_pair = (
                meta["applicant_key"].fillna("").astype(str)
                + "||"
                + meta["ipc_main_group"].fillna("").astype(str)
            )
            target_pair = target_applicant + "||" + target_field
            same_sum, same_code = _group_sum(matrix, candidate_pair, target_pair)
            bs_sum -= _aligned_dot(target_matrix, same_sum, same_code)

        for other in range(year + 1, year + window + 1):
            if other not in years:
                continue
            matrix, meta = _load(vector_root, other)
            fs_sum += (target_matrix @ sparse.csr_matrix(matrix.sum(axis=0)).T).toarray().ravel()

            same_sum, same_code = _group_sum(
                matrix,
                meta["applicant_key"].fillna("").astype(str),
                target_applicant,
            )
            fs_sum -= _aligned_dot(target_matrix, same_sum, same_code)

        bs_den = sum(counts.get(y, 0) for y in range(year - window, year))
        fs_den = sum(counts.get(y, 0) for y in range(year + 1, year + window + 1))

        out = target.copy()
        out["bs"] = bs_sum / bs_den if bs_den else np.nan
        out["fs"] = fs_sum / fs_den if fs_den else np.nan
        out.loc[target_field.eq(""), "bs"] = np.nan
        out["application_year"] = year
        out["window_complete"] = int(
            year - window >= min(years) and year + window <= max(years)
        )
        out = add_field_adjustment(out)
        out.to_parquet(score_root / f"{year}.parquet", index=False, compression="zstd")
