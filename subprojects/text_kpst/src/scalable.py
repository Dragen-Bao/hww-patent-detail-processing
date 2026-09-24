"""Year-sharded TF-BIDF vector construction and KPST scoring.

The scorer uses the identity
    sum_q cosine(u_p, u_q) = u_p dot (sum_q u_q)
for L2-normalized patent vectors. This reproduces the paper's similarity sums
without materializing an N x N patent-pair matrix.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from .metrics import add_field_adjustment
from .preprocess import PatentTextPreprocessor, combine_patent_text
from .tfbidf import BIDFHistoryState, vectorize_with_history

TOKEN_SEPARATOR = "\x1f"


def _year_directories(root: Path) -> list[tuple[int, Path]]:
    rows: list[tuple[int, Path]] = []
    for directory in root.glob("application_year=*"):
        label = directory.name.split("=", 1)[1]
        if label.isdigit():
            rows.append((int(label), directory))
    return sorted(rows)


def _read_year(root: Path, year: int) -> pd.DataFrame:
    paths = sorted((root / f"application_year={year}").glob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def tokenize_and_build_vocabulary(
    canonical_root: Path,
    vector_root: Path,
    *,
    text_fields: list[str],
    stopwords_path: Path | None = None,
    user_dictionary_path: Path | None = None,
    keep_single_char: bool = True,
    min_corpus_frequency: int = 1,
) -> tuple[dict[str, int], dict[int, int]]:
    """First pass: tokenize granted inventions, persist tokens, build vocabulary.

    The BS/FS denominator universe is explicitly defined as granted invention
    patents that can form a non-empty vector under the configured text fields
    and vocabulary. Coverage against all granted inventions is written to an
    audit table rather than silently changing the denominator.
    """
    if vector_root.exists():
        shutil.rmtree(vector_root)
    token_root = vector_root / "tokens"
    token_root.mkdir(parents=True, exist_ok=True)
    processor = PatentTextPreprocessor.from_files(
        stopwords_path=stopwords_path,
        user_dictionary=user_dictionary_path,
        keep_single_char=keep_single_char,
    )
    corpus_counts: Counter[str] = Counter()
    audit_rows: list[dict[str, object]] = []

    for year, _ in _year_directories(canonical_root):
        frame = _read_year(canonical_root, year)
        if frame.empty:
            continue
        frame = frame.loc[
            frame["is_granted_invention"].eq(1)
            & frame["application_date"].notna()
        ].drop_duplicates("application_no").copy()
        if frame.empty:
            continue

        granted_count = len(frame)
        texts = [combine_patent_text(row, text_fields) for row in frame.to_dict(orient="records")]
        tokens = [processor.tokenize(text) for text in texts]
        has_tokens = np.fromiter((bool(token_list) for token_list in tokens), dtype=bool)
        tokenized = frame.loc[has_tokens].reset_index(drop=True)
        token_lists = [
            token_list
            for token_list, flag in zip(tokens, has_tokens, strict=True)
            if flag
        ]
        for token_list in token_lists:
            corpus_counts.update(token_list)

        out = tokenized[
            [
                "application_no",
                "application_date",
                "application_year",
                "applicant_identity_key",
                "ipc_domain_key",
                "baseline_firm_eligible",
            ]
        ].copy()
        out["applicant_identity_missing_flag"] = (
            tokenized["applicant_identity_key"].fillna("").astype(str).str.strip().eq("")
        ).astype("int8")
        out["applicant_address_missing_flag"] = (
            tokenized["applicant_address"].fillna("").astype(str).str.strip().eq("")
        ).astype("int8")
        out["ipc_domain_missing_flag"] = (
            tokenized["ipc_domain_key"].fillna("").astype(str).str.strip().eq("")
        ).astype("int8")
        out["tokens"] = [TOKEN_SEPARATOR.join(token_list) for token_list in token_lists]
        out.to_parquet(
            token_root / f"tokens-{year}.parquet", index=False, compression="zstd"
        )
        audit_rows.append(
            {
                "application_year": year,
                "granted_invention_count": granted_count,
                "tokenized_text_count": len(out),
                "vectorizable_patent_count": 0,
                "missing_or_empty_text_count": granted_count - len(out),
                "applicant_identity_missing_count": 0,
                "applicant_address_missing_count": 0,
                "ipc_domain_missing_count": 0,
            }
        )

    kept = sorted(
        token for token, count in corpus_counts.items() if count >= min_corpus_frequency
    )
    vocabulary = {token: index for index, token in enumerate(kept)}
    vocab_frame = pd.DataFrame(
        {"token": kept, "token_id": np.arange(len(kept), dtype=np.int64)}
    )
    vocab_frame.to_parquet(
        vector_root / "vocabulary.parquet", index=False, compression="zstd"
    )

    counts_by_year: dict[int, int] = {}
    audit_lookup = {int(row["application_year"]): row for row in audit_rows}
    for token_path in sorted(
        token_root.glob("tokens-*.parquet"),
        key=lambda path: int(path.stem.split("-")[-1]),
    ):
        year = int(token_path.stem.split("-")[-1])
        frame = pd.read_parquet(token_path)
        token_lists = [
            str(value).split(TOKEN_SEPARATOR) if str(value) else []
            for value in frame["tokens"]
        ]
        vectorizable = np.fromiter(
            (
                any(token in vocabulary for token in token_list)
                for token_list in token_lists
            ),
            dtype=bool,
        )
        frame = frame.loc[vectorizable].reset_index(drop=True)
        frame.to_parquet(token_path, index=False, compression="zstd")
        counts_by_year[year] = len(frame)

        audit = audit_lookup[year]
        audit["vectorizable_patent_count"] = len(frame)
        audit["applicant_identity_missing_count"] = int(
            frame["applicant_identity_missing_flag"].sum()
        )
        audit["applicant_address_missing_count"] = int(
            frame["applicant_address_missing_flag"].sum()
        )
        audit["ipc_domain_missing_count"] = int(
            frame["ipc_domain_missing_flag"].sum()
        )
        granted = int(audit["granted_invention_count"])
        audit["text_coverage_rate"] = len(frame) / granted if granted else float("nan")

    coverage = pd.DataFrame(audit_rows).sort_values("application_year")
    coverage.to_parquet(
        vector_root / "text_coverage_by_year.parquet", index=False, compression="zstd"
    )
    coverage.to_csv(
        vector_root / "text_coverage_by_year.csv", index=False, encoding="utf-8-sig"
    )
    (vector_root / "year_counts.json").write_text(
        json.dumps(counts_by_year, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return vocabulary, counts_by_year


def build_vector_shards(
    vector_root: Path,
    vocabulary: dict[str, int] | None = None,
) -> dict[str, object]:
    """Second pass: create year-sharded strict-prior TF-BIDF matrices."""
    if vocabulary is None:
        vocab_frame = pd.read_parquet(vector_root / "vocabulary.parquet")
        vocabulary = dict(zip(vocab_frame["token"].astype(str), vocab_frame["token_id"].astype(int)))
    token_root = vector_root / "tokens"
    matrix_root = vector_root / "matrices"
    meta_root = vector_root / "metadata"
    for directory in (matrix_root, meta_root):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    history = BIDFHistoryState.empty(len(vocabulary))
    years: list[int] = []
    row_count = 0
    for token_path in sorted(token_root.glob("tokens-*.parquet"), key=lambda p: int(p.stem.split("-")[-1])):
        year = int(token_path.stem.split("-")[-1])
        frame = pd.read_parquet(token_path).sort_values(
            ["application_date", "application_no"], kind="stable"
        ).reset_index(drop=True)
        documents = [
            str(value).split(TOKEN_SEPARATOR) if str(value) else []
            for value in frame["tokens"]
        ]
        matrix, history, prior_counts = vectorize_with_history(
            documents,
            frame["application_date"].astype(str).tolist(),
            vocabulary=vocabulary,
            history=history,
        )
        sparse.save_npz(matrix_root / f"tfbidf-{year}.npz", matrix, compressed=True)
        meta = frame.drop(columns="tokens").copy()
        meta["bidf_prior_patent_count"] = prior_counts
        meta.to_parquet(meta_root / f"metadata-{year}.parquet", index=False, compression="zstd")
        years.append(year)
        row_count += len(meta)
    manifest = {
        "years": years,
        "row_count": row_count,
        "vocabulary_size": len(vocabulary),
        "bidf_formula": "log(N_prior/(1+df_prior))",
        "same_day_freeze": True,
    }
    (vector_root / "vector_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _group_sums_for_target_keys(
    candidate_matrix: sparse.csr_matrix,
    candidate_keys: Iterable[object],
    target_keys: Iterable[object],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    target_values = [str(value) if value is not None else "" for value in target_keys]
    unique = list(dict.fromkeys(target_values))
    lookup = {key: idx for idx, key in enumerate(unique)}
    target_codes = np.asarray([lookup[key] for key in target_values], dtype=np.int64)
    rows: list[int] = []
    cols: list[int] = []
    for candidate_index, raw_key in enumerate(candidate_keys):
        key = str(raw_key) if raw_key is not None else ""
        group = lookup.get(key)
        if group is not None and key:
            rows.append(group)
            cols.append(candidate_index)
    if rows:
        incidence = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float64), (rows, cols)),
            shape=(len(unique), candidate_matrix.shape[0]),
        )
        sums = (incidence @ candidate_matrix).tocsr()
    else:
        sums = sparse.csr_matrix((len(unique), candidate_matrix.shape[1]), dtype=np.float64)
    return sums, target_codes


def _rowwise_dot_with_group_sums(
    target_matrix: sparse.csr_matrix,
    group_sums: sparse.csr_matrix,
    target_codes: np.ndarray,
    *,
    chunk_size: int = 20_000,
) -> np.ndarray:
    result = np.zeros(target_matrix.shape[0], dtype=np.float64)
    for start in range(0, target_matrix.shape[0], chunk_size):
        end = min(start + chunk_size, target_matrix.shape[0])
        aligned = group_sums[target_codes[start:end]]
        result[start:end] = np.asarray(
            target_matrix[start:end].multiply(aligned).sum(axis=1)
        ).ravel()
    return result


def _load_matrix_meta(vector_root: Path, year: int) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    matrix = sparse.load_npz(vector_root / "matrices" / f"tfbidf-{year}.npz").tocsr()
    meta = pd.read_parquet(vector_root / "metadata" / f"metadata-{year}.parquet")
    if matrix.shape[0] != len(meta):
        raise ValueError(f"matrix/metadata row mismatch for {year}")
    return matrix, meta


def _applicant_domain_keys(
    applicants: Iterable[object], domains: Iterable[object]
) -> pd.Series:
    """Build a composite exclusion key only when applicant identity is known.

    Missing applicant identity must not cause unrelated patents to be treated
    as the same applicant. This keeps backward and forward exclusion rules
    symmetric.
    """
    applicant_series = pd.Series(applicants).fillna("").astype(str).str.strip()
    domain_series = pd.Series(domains).fillna("").astype(str).str.strip()
    valid = applicant_series.ne("") & domain_series.ne("")
    return pd.Series(
        np.where(
            valid,
            applicant_series + "||DOMAIN||" + domain_series,
            "",
        ),
        index=applicant_series.index,
        dtype="object",
    )


def score_year(
    vector_root: Path,
    year: int,
    *,
    counts_by_year: dict[int, int],
    window_years: int = 5,
) -> pd.DataFrame:
    """Score one application year using method-6 comparison pools."""
    target_matrix, target = _load_matrix_meta(vector_root, year)
    n = len(target)
    backward_sum = np.zeros(n, dtype=np.float64)
    forward_sum = np.zeros(n, dtype=np.float64)
    target_domains = target["ipc_domain_key"].fillna("").astype(str)
    target_applicants = target["applicant_identity_key"].fillna("").astype(str).str.strip()
    target_applicant_domain = _applicant_domain_keys(
        target_applicants, target_domains
    )

    available_years = {
        int(path.stem.split("-")[-1])
        for path in (vector_root / "matrices").glob("tfbidf-*.npz")
    }

    for candidate_year in range(year - window_years, year):
        if candidate_year not in available_years:
            continue
        candidate_matrix, candidate = _load_matrix_meta(vector_root, candidate_year)
        domain_sums, domain_codes = _group_sums_for_target_keys(
            candidate_matrix,
            candidate["ipc_domain_key"].fillna("").astype(str),
            target_domains,
        )
        backward_sum += _rowwise_dot_with_group_sums(
            target_matrix, domain_sums, domain_codes
        )
        candidate_pair_keys = _applicant_domain_keys(
            candidate["applicant_identity_key"],
            candidate["ipc_domain_key"],
        )
        same_sums, same_codes = _group_sums_for_target_keys(
            candidate_matrix, candidate_pair_keys, target_applicant_domain
        )
        backward_sum -= _rowwise_dot_with_group_sums(
            target_matrix, same_sums, same_codes
        )

    for candidate_year in range(year + 1, year + window_years + 1):
        if candidate_year not in available_years:
            continue
        candidate_matrix, candidate = _load_matrix_meta(vector_root, candidate_year)
        total_sum = sparse.csr_matrix(candidate_matrix.sum(axis=0))
        forward_sum += (target_matrix @ total_sum.T).toarray().ravel()
        same_sums, same_codes = _group_sums_for_target_keys(
            candidate_matrix,
            candidate["applicant_identity_key"].fillna("").astype(str),
            target_applicants,
        )
        forward_sum -= _rowwise_dot_with_group_sums(
            target_matrix, same_sums, same_codes
        )

    backward_denominator = sum(
        int(counts_by_year.get(candidate_year, 0))
        for candidate_year in range(year - window_years, year)
    )
    forward_denominator = sum(
        int(counts_by_year.get(candidate_year, 0))
        for candidate_year in range(year + 1, year + window_years + 1)
    )
    bs = backward_sum / backward_denominator if backward_denominator else np.full(n, np.nan)
    fs = forward_sum / forward_denominator if forward_denominator else np.full(n, np.nan)
    bs = np.asarray(bs, dtype=float)
    fs = np.asarray(fs, dtype=float)
    bs[target_domains.eq("").to_numpy()] = np.nan

    result = target.copy()
    result["backward_similarity_sum"] = backward_sum
    result["forward_similarity_sum"] = forward_sum
    result["backward_denominator_patents"] = backward_denominator
    result["forward_denominator_patents"] = forward_denominator
    result["average_backward_similarity"] = bs
    result["average_forward_similarity"] = fs
    result["kpst_ratio_unadjusted"] = np.divide(
        fs,
        bs,
        out=np.full(n, np.nan),
        where=np.isfinite(bs) & (bs != 0) & np.isfinite(fs),
    )
    min_year = min(available_years) if available_years else year
    max_year = max(available_years) if available_years else year
    result["backward_window_complete"] = int(year - window_years >= min_year)
    result["forward_window_complete"] = int(year + window_years <= max_year)
    result["application_year"] = year
    result = add_field_adjustment(result)
    complete = result["backward_window_complete"].eq(1) & result["forward_window_complete"].eq(1)
    result["kpst_quality_adjusted_complete"] = result["kpst_quality_adjusted"].where(complete)
    return result


def score_all_years(
    vector_root: Path,
    score_root: Path,
    *,
    window_years: int = 5,
) -> dict[str, object]:
    if score_root.exists():
        shutil.rmtree(score_root)
    score_root.mkdir(parents=True, exist_ok=True)
    raw_counts = json.loads((vector_root / "year_counts.json").read_text(encoding="utf-8"))
    counts = {int(year): int(count) for year, count in raw_counts.items()}
    years = sorted(counts)
    rows = 0
    for year in years:
        scored = score_year(
            vector_root,
            year,
            counts_by_year=counts,
            window_years=window_years,
        )
        scored.to_parquet(score_root / f"kpst-{year}.parquet", index=False, compression="zstd")
        rows += len(scored)
    coverage_path = vector_root / "text_coverage_by_year.csv"
    manifest = {
        "years": years,
        "row_count": rows,
        "window_years": window_years,
        "denominator_universe": "granted inventions with non-empty configured text vector",
        "text_coverage_audit": str(coverage_path),
        "same_applicant_rule": "exclude only when applicant identity key is non-empty",
        "score_directory_cleaned_before_build": True,
    }
    (score_root / "score_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
